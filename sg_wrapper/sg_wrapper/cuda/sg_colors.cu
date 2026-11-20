// Original SG inference kernel; no gsplat source is copied.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

__global__ void sg_color_kernel(
    const float* means, const float* base, const float* dirs,
    const float* sharp, const float* amp, const float* cameras,
    float* output, int64_t total, int64_t G, int64_t V, int64_t L,
    bool clamp_colors, float eps) {
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t g = idx % G, camera = idx / G, b = camera / V;
    const int64_t point = b * G + g;
    float vx = cameras[camera * 16 + 3] - means[point * 3];
    float vy = cameras[camera * 16 + 7] - means[point * 3 + 1];
    float vz = cameras[camera * 16 + 11] - means[point * 3 + 2];
    const float vn = sqrtf(vx*vx + vy*vy + vz*vz) + eps;
    vx /= vn; vy /= vn; vz /= vn;
    float r = 0.f, gcol = 0.f, blue = 0.f;
    for (int64_t l = 0; l < L; ++l) {
        const int64_t k = point * L + l;
        const float ar = amp[k*3], ag = amp[k*3+1], ab = amp[k*3+2];
        // 任意位置的零振幅表示删除；不能把原始 lobe 顺序误当作保留前缀。
        if (ar == 0.f && ag == 0.f && ab == 0.f) continue;
        float x = dirs[k*3], y = dirs[k*3+1], z = dirs[k*3+2];
        const float norm = fmaxf(sqrtf(x*x + y*y + z*z), eps);
        x /= norm; y /= norm; z /= norm;
        const float weight = expf(sharp[k] * (x*vx + y*vy + z*vz - 1.f));
        r += ar*weight; gcol += ag*weight; blue += ab*weight;
    }
    float rgb[3] = {r, gcol, blue};
    for (int c=0; c<3; ++c) {
        float value = (base[point*3+c] + rgb[c]) * 0.28209479177387814f + 0.5f;
        if (clamp_colors) value = fminf(1.f, fmaxf(0.f, value));
        output[idx*3+c] = value;
    }
}

torch::Tensor sg_colors_forward(
    torch::Tensor means, torch::Tensor base, torch::Tensor dirs,
    torch::Tensor sharp, torch::Tensor amp, torch::Tensor cameras,
    bool clamp_colors, double eps) {
    for (const auto& t : {means, base, dirs, sharp, amp, cameras}) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.is_contiguous(),
                    "SG CUDA requires contiguous CUDA float32 tensors");
        TORCH_CHECK(t.device() == means.device(), "SG tensors must share a device");
        TORCH_CHECK(!t.requires_grad(), "SG CUDA is inference-only; detach inputs or use reference");
    }
    TORCH_CHECK(std::isfinite(eps) && eps > 0, "eps must be positive and finite");
    TORCH_CHECK(means.dim()==3 && means.size(2)==3, "means must be [B,G,3]");
    const auto B=means.size(0), G=means.size(1);
    TORCH_CHECK(base.sizes()==means.sizes(), "base must match means");
    TORCH_CHECK(dirs.dim()==4 && dirs.size(0)==B && dirs.size(1)==G && dirs.size(3)==3,
                "dirs must be [B,G,L,3]");
    const auto L=dirs.size(2);
    TORCH_CHECK(amp.sizes()==dirs.sizes(), "amp must match dirs");
    TORCH_CHECK(sharp.dim()==4 && sharp.size(0)==B && sharp.size(1)==G &&
                sharp.size(2)==L && sharp.size(3)==1, "sharp must be [B,G,L,1]");
    TORCH_CHECK(cameras.dim()==4 && cameras.size(0)==B &&
                cameras.size(2)==4 && cameras.size(3)==4, "cameras must be [B,V,4,4]");
    const auto V=cameras.size(1);
    const c10::cuda::CUDAGuard guard(means.device());
    auto out=torch::empty({B*V,G,int64_t(3)}, means.options());
    const auto total=B*V*G;
    if (total) {
        sg_color_kernel<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
            means.data_ptr<float>(),base.data_ptr<float>(),dirs.data_ptr<float>(),
            sharp.data_ptr<float>(),amp.data_ptr<float>(),cameras.data_ptr<float>(),
            out.data_ptr<float>(),total,G,V,L,clamp_colors,float(eps));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sg_colors_forward, "SG inference colors (CUDA)");
}
