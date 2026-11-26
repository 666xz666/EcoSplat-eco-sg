// 以同一份屏幕空间排序列表进行解析预滤波与自适应子像素合成。
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

constexpr int kTile = 16;
constexpr int kThreads = kTile * kTile;
constexpr float kAlphaCutoff = 1.0f / 255.0f;
constexpr float kTransmittanceCutoff = 1.0e-4f;
constexpr float kGaussNode = 0.28867513459f;

struct Layer {
    float2 mean;
    float4 center;
    float4 filtered;
    float3 color;
};

__device__ __forceinline__ bool advance(float alpha, float& transmittance) {
    const float after = transmittance * (1.0f - alpha);
    if (after <= kTransmittanceCutoff) {
        return false;
    }
    transmittance = after;
    return true;
}

__device__ __forceinline__ float alpha_at(float4 quad, float x, float y) {
    const float exponent = 0.5f * (quad.x * x * x + quad.z * y * y) + quad.y * x * y;
    const float alpha = fminf(0.999f, quad.w * __expf(-exponent));
    return (exponent >= 0.0f && alpha >= kAlphaCutoff) ? alpha : 0.0f;
}

__device__ __forceinline__ float3 accumulate(float3 pixel, float weight, float3 color) {
    return make_float3(fmaf(weight, color.x, pixel.x),
                       fmaf(weight, color.y, pixel.y),
                       fmaf(weight, color.z, pixel.z));
}

__device__ __forceinline__ Layer read_layer(int id, const float2* means, const float3* colors,
                                            const float4* footprints) {
    return Layer{means[id], footprints[2 * id], footprints[2 * id + 1], colors[id]};
}

__global__ void prepare_kernel(const float3* conics, const float* opacity, const int2* radii,
                               float4* footprint, int2* padded, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int2 radius = radii[i];
    if (radius.x <= 0 || radius.y <= 0 || opacity[i] < kAlphaCutoff) {
        padded[i] = make_int2(0, 0);
        return;
    }
    const float3 q = conics[i];
    const float determinant = fmaxf(0.0f, q.x * q.z - q.y * q.y);
    const float variance = 1.0f / 12.0f;
    const float volume = 1.0f + variance * (q.x + q.z) + variance * variance * determinant;
    const float inverse = 1.0f / volume;
    footprint[2 * i] = make_float4(q.x, q.y, q.z, opacity[i]);
    footprint[2 * i + 1] = make_float4((q.x + variance * determinant) * inverse,
                                      q.y * inverse,
                                      (q.z + variance * determinant) * inverse,
                                      opacity[i] * rsqrtf(volume));
    padded[i] = make_int2(radius.x + 1, radius.y + 1);
}

__global__ void coarse_kernel(const float2* means, const float3* colors, const float4* footprints,
                              const int* offsets, const int* ids, int count, int width, int height,
                              int tile_width, int tiles, bool corrected, float3* output, float* scores) {
    const int tile_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int x = (tile_id % tile_width) * kTile + tid % kTile;
    const int y = (tile_id / tile_width) * kTile + tid / kTile;
    const bool inside = x < width && y < height;
    const float px = x + 0.5f;
    const float py = y + 0.5f;
    const int begin = offsets[tile_id];
    const int end = tile_id + 1 < tiles ? offsets[tile_id + 1] : count;
    __shared__ Layer batch[kThreads];
    __shared__ float reduction[kThreads];

    bool center_live = inside;
    bool filtered_live = inside;
    float center_trans = 1.0f;
    float filtered_trans = 1.0f;
    float gradient_x = 0.0f;
    float gradient_y = 0.0f;
    float cross_trans = 0.0f;
    float3 cross_color = make_float3(0.0f, 0.0f, 0.0f);
    float3 filtered_color = make_float3(0.0f, 0.0f, 0.0f);

    for (int next = begin; next < end; next += kThreads) {
        if (__syncthreads_count(!center_live && !filtered_live) == kThreads) break;
        const int index = next + tid;
        if (index < end) batch[tid] = read_layer(ids[index], means, colors, footprints);
        __syncthreads();
        const int limit = min(kThreads, end - next);
        for (int j = 0; j < limit && (center_live || filtered_live); ++j) {
            const Layer g = batch[j];
            const float dx = g.mean.x - px;
            const float dy = g.mean.y - py;
            if (filtered_live) {
                const float alpha = alpha_at(g.filtered, dx, dy);
                if (alpha > 0.0f) {
                    const float before = filtered_trans;
                    filtered_live = advance(alpha, filtered_trans);
                    if (filtered_live) filtered_color = accumulate(filtered_color, before * alpha, g.color);
                }
            }
            if (center_live) {
                const float alpha = alpha_at(g.center, dx, dy);
                if (alpha > 0.0f) {
                    const float before = center_trans;
                    center_live = advance(alpha, center_trans);
                    if (center_live) {
                        const float gx = alpha < 0.999f ? alpha * (g.center.x * dx + g.center.y * dy) : 0.0f;
                        const float gy = alpha < 0.999f ? alpha * (g.center.y * dx + g.center.z * dy) : 0.0f;
                        // 维护混合透过率梯度与仅由不同高斯产生的拉普拉斯交叉项。
                        const float cross = alpha * cross_trans + 2.0f * (gx * gradient_x + gy * gradient_y);
                        cross_color = accumulate(cross_color, cross, g.color);
                        cross_trans -= cross;
                        gradient_x = (1.0f - alpha) * gradient_x - before * gx;
                        gradient_y = (1.0f - alpha) * gradient_y - before * gy;
                    }
                }
            }
        }
    }

    cross_color.x *= 1.0f / 24.0f;
    cross_color.y *= 1.0f / 24.0f;
    cross_color.z *= 1.0f / 24.0f;
    if (inside) {
        output[y * width + x] = corrected ? accumulate(filtered_color, 1.0f, cross_color)
                                          : filtered_color;
    }
    reduction[tid] = inside ? cross_color.x * cross_color.x + cross_color.y * cross_color.y +
                               cross_color.z * cross_color.z : 0.0f;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    if (tid == 0) scores[tile_id] = reduction[0] / max(end - begin, 1);
}

__global__ void quad_kernel(const float2* means, const float3* colors, const float4* footprints,
                            const int* offsets, const int* ids, int count, int width, int height,
                            int tile_width, int tiles, const int64_t* selected, float3* output) {
    const int tile_id = static_cast<int>(selected[blockIdx.x]);
    const int tid = threadIdx.x;
    const int x = (tile_id % tile_width) * kTile + tid % kTile;
    const int y = (tile_id / tile_width) * kTile + tid / kTile;
    const bool inside = x < width && y < height;
    const float px = x + 0.5f;
    const float py = y + 0.5f;
    const int begin = offsets[tile_id];
    const int end = tile_id + 1 < tiles ? offsets[tile_id + 1] : count;
    __shared__ float2 shared_mean[kThreads];
    __shared__ float4 shared_quad[kThreads];
    __shared__ float3 shared_color[kThreads];
    float trans[4];
    float3 light[4];
    unsigned live = inside ? 15u : 0u;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        light[k] = make_float3(0.0f, 0.0f, 0.0f);
        trans[k] = 1.0f;
    }
    for (int next = begin; next < end; next += kThreads) {
        if (__syncthreads_count(live == 0) == kThreads) break;
        const int index = next + tid;
        if (index < end) {
            const int id = ids[index];
            shared_mean[tid] = means[id];
            shared_quad[tid] = footprints[2 * id];
            shared_color[tid] = colors[id];
        }
        __syncthreads();
        const int limit = min(kThreads, end - next);
        for (int j = 0; j < limit && live != 0; ++j) {
            const float dx = shared_mean[j].x - px;
            const float dy = shared_mean[j].y - py;
            const float4 q = shared_quad[j];
            const float3 c = shared_color[j];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                if ((live & (1u << k)) == 0) continue;
                const float u = (k & 1) ? kGaussNode : -kGaussNode;
                const float v = (k & 2) ? kGaussNode : -kGaussNode;
                const float alpha = alpha_at(q, dx - u, dy - v);
                if (alpha == 0.0f) continue;
                const float before = trans[k];
                if (advance(alpha, trans[k])) {
                    light[k] = accumulate(light[k], before * alpha, c);
                } else {
                    live &= ~(1u << k);
                }
            }
        }
    }
    if (inside) {
        float3 result = make_float3(0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int k = 0; k < 4; ++k) result = accumulate(result, 0.25f, light[k]);
        output[y * width + x] = result;
    }
}

void require_tensor(const at::Tensor& tensor, at::ScalarType dtype) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == dtype && tensor.is_contiguous(),
                "Expected a contiguous CUDA tensor with the declared dtype");
}

void require_raster(const at::Tensor& means, const at::Tensor& color, const at::Tensor& footprints,
                    const at::Tensor& offsets, const at::Tensor& ids, int width, int height) {
    require_tensor(means, at::kFloat);
    require_tensor(color, at::kFloat);
    require_tensor(footprints, at::kFloat);
    require_tensor(offsets, at::kInt);
    require_tensor(ids, at::kInt);
    const int64_t count = means.numel() / 2;
    const int64_t tiles = ((width + kTile - 1) / kTile) * ((height + kTile - 1) / kTile);
    TORCH_CHECK(width > 0 && height > 0 && means.numel() == count * 2 &&
                color.numel() == count * 3 && footprints.numel() == count * 8 && offsets.numel() == tiles,
                "Incompatible view layout");
    TORCH_CHECK(means.device() == color.device() && means.device() == offsets.device() &&
                means.device() == ids.device() && means.device() == footprints.device(),
                "All view tensors must be on the same device");
}

}  // namespace

std::vector<at::Tensor> prepare_footprints(const at::Tensor& conics, const at::Tensor& opacity,
                                           const at::Tensor& radii) {
    require_tensor(conics, at::kFloat);
    require_tensor(opacity, at::kFloat);
    require_tensor(radii, at::kInt);
    const int count = opacity.numel();
    TORCH_CHECK(count > 0 && conics.numel() == count * 3 && radii.numel() == count * 2 &&
                conics.device() == radii.device() && conics.device() == opacity.device(),
                "Incompatible projected Gaussians");
    c10::cuda::CUDAGuard guard(conics.device());
    auto footprint = at::empty({count, 8}, conics.options());
    auto padded = at::empty_like(radii);
    prepare_kernel<<<(count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float3*>(conics.data_ptr<float>()), opacity.data_ptr<float>(),
        reinterpret_cast<const int2*>(radii.data_ptr<int>()), reinterpret_cast<float4*>(footprint.data_ptr<float>()),
        reinterpret_cast<int2*>(padded.data_ptr<int>()), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {footprint, padded};
}

std::vector<at::Tensor> render_base(const at::Tensor& means, const at::Tensor& color,
                                    const at::Tensor& footprints, const at::Tensor& offsets,
                                    const at::Tensor& ids, int width, int height, bool corrected) {
    require_raster(means, color, footprints, offsets, ids, width, height);
    c10::cuda::CUDAGuard guard(means.device());
    const int tw = (width + kTile - 1) / kTile;
    const int tiles = offsets.numel();
    auto image = at::empty({height, width, 3}, means.options());
    auto scores = at::empty({tiles}, means.options());
    coarse_kernel<<<tiles, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float2*>(means.data_ptr<float>()),
        reinterpret_cast<const float3*>(color.data_ptr<float>()),
        reinterpret_cast<const float4*>(footprints.data_ptr<float>()), offsets.data_ptr<int>(), ids.data_ptr<int>(),
        ids.numel(), width, height, tw, tiles, corrected, reinterpret_cast<float3*>(image.data_ptr<float>()),
        scores.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {image, scores};
}

void render_selected(const at::Tensor& means, const at::Tensor& color, const at::Tensor& footprints,
                     const at::Tensor& offsets, const at::Tensor& ids, const at::Tensor& selected,
                     at::Tensor& image, int width, int height) {
    require_raster(means, color, footprints, offsets, ids, width, height);
    require_tensor(selected, at::kLong);
    require_tensor(image, at::kFloat);
    TORCH_CHECK(selected.device() == means.device() && image.device() == means.device() &&
                image.numel() == static_cast<int64_t>(width) * height * 3 &&
                selected.numel() <= offsets.numel(), "Incompatible selected tiles or output");
    if (selected.numel() == 0) return;
    c10::cuda::CUDAGuard guard(means.device());
    quad_kernel<<<selected.numel(), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float2*>(means.data_ptr<float>()),
        reinterpret_cast<const float3*>(color.data_ptr<float>()),
        reinterpret_cast<const float4*>(footprints.data_ptr<float>()), offsets.data_ptr<int>(), ids.data_ptr<int>(),
        ids.numel(), width, height, (width + kTile - 1) / kTile, offsets.numel(), selected.data_ptr<int64_t>(),
        reinterpret_cast<float3*>(image.data_ptr<float>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

__global__ void downsample_kernel(const float3* input, float3* output,
                                  int source_w, int source_h, int width, int height) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    const int x = index % width;
    const int y = index / width;
    const float sx = static_cast<float>(source_w) / width;
    const float sy = static_cast<float>(source_h) / height;
    const float left = x * sx;
    const float right = (x + 1) * sx;
    const float top = y * sy;
    const float bottom = (y + 1) * sy;
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = static_cast<int>(top); j < min(source_h, static_cast<int>(ceilf(bottom))); ++j) {
        const float wy = fminf(bottom, j + 1.0f) - fmaxf(top, static_cast<float>(j));
        for (int i = static_cast<int>(left); i < min(source_w, static_cast<int>(ceilf(right))); ++i) {
            const float wx = fminf(right, i + 1.0f) - fmaxf(left, static_cast<float>(i));
            sum = accumulate(sum, wx * wy, input[j * source_w + i]);
        }
    }
    const float scale = 1.0f / (sx * sy);
    output[index] = make_float3(sum.x * scale, sum.y * scale, sum.z * scale);
}

}  // namespace

at::Tensor downsample_box(const at::Tensor& input, int width, int height) {
    require_tensor(input, at::kFloat);
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 3 && width > 0 && height > 0 &&
                input.size(0) >= height && input.size(1) >= width, "Expected HWC RGB downsampling");
    if (input.size(0) == height && input.size(1) == width) return input;
    c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty({height, width, 3}, input.options());
    const int count = width * height;
    downsample_kernel<<<(count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const float3*>(input.data_ptr<float>()),
        reinterpret_cast<float3*>(output.data_ptr<float>()),
        input.size(1), input.size(0), width, height);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
