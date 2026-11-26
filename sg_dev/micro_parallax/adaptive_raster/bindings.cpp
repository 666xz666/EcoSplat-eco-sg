// WSL 真源中的自适应亚像素推理扩展入口。
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> prepare_footprints(
    const at::Tensor& conics, const at::Tensor& opacity, const at::Tensor& radii);

std::vector<at::Tensor> render_base(
    const at::Tensor& means, const at::Tensor& color, const at::Tensor& footprints,
    const at::Tensor& offsets, const at::Tensor& ids, int width, int height, bool corrected);

void render_selected(
    const at::Tensor& means, const at::Tensor& color, const at::Tensor& footprints,
    const at::Tensor& offsets, const at::Tensor& ids, const at::Tensor& selected,
    at::Tensor& image, int width, int height);

at::Tensor downsample_box(const at::Tensor& input, int width, int height);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("prepare_footprints", &prepare_footprints);
    module.def("render_base", &render_base);
    module.def("render_selected", &render_selected);
    module.def("downsample_box", &downsample_box);
}
