#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> patchmatch_cuda(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor weights,
    torch::Tensor initial_nnf,
    int64_t patch_radius,
    int64_t iterations,
    int64_t seed);

torch::Tensor vote_cuda(
    torch::Tensor source_edit,
    torch::Tensor nnf,
    int64_t patch_radius);

namespace {

void check_features(const torch::Tensor& source, const torch::Tensor& target) {
  TORCH_CHECK(source.is_cuda() && target.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(source.scalar_type() == torch::kFloat32,
              "source must be float32");
  TORCH_CHECK(target.scalar_type() == torch::kFloat32,
              "target must be float32");
  TORCH_CHECK(source.dim() == 3 && target.dim() == 3,
              "features must have shape (C,H,W)");
  TORCH_CHECK(source.sizes() == target.sizes(),
              "source and target feature shapes must match");
  TORCH_CHECK(source.device() == target.device(),
              "source and target must be on the same CUDA device");
}

}  // namespace

std::vector<torch::Tensor> patchmatch(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor weights,
    torch::Tensor initial_nnf,
    int64_t patch_radius,
    int64_t iterations,
    int64_t seed) {
  check_features(source, target);
  TORCH_CHECK(weights.is_cuda() && weights.scalar_type() == torch::kFloat32,
              "weights must be a CUDA float32 tensor");
  TORCH_CHECK(weights.dim() == 1 && weights.size(0) == source.size(0),
              "weights must contain one value per feature channel");
  TORCH_CHECK(initial_nnf.is_cuda() && initial_nnf.scalar_type() == torch::kInt32,
              "initial_nnf must be a CUDA int32 tensor");
  TORCH_CHECK(initial_nnf.dim() == 3 && initial_nnf.size(2) == 2,
              "initial_nnf must have shape (H,W,2)");
  TORCH_CHECK(initial_nnf.size(0) == source.size(1) &&
                  initial_nnf.size(1) == source.size(2),
              "NNF spatial shape must match the features");
  TORCH_CHECK(patch_radius >= 0, "patch_radius must be non-negative");
  TORCH_CHECK(iterations >= 1, "iterations must be positive");
  return patchmatch_cuda(source.contiguous(), target.contiguous(),
                         weights.contiguous(), initial_nnf.contiguous(),
                         patch_radius, iterations, seed);
}

torch::Tensor vote(
    torch::Tensor source_edit,
    torch::Tensor nnf,
    int64_t patch_radius) {
  TORCH_CHECK(source_edit.is_cuda() && source_edit.scalar_type() == torch::kFloat32,
              "source_edit must be a CUDA float32 tensor");
  TORCH_CHECK(source_edit.dim() == 3, "source_edit must have shape (C,H,W)");
  TORCH_CHECK(nnf.is_cuda() && nnf.scalar_type() == torch::kInt32,
              "nnf must be a CUDA int32 tensor");
  TORCH_CHECK(nnf.dim() == 3 && nnf.size(2) == 2,
              "nnf must have shape (H,W,2)");
  TORCH_CHECK(nnf.size(0) == source_edit.size(1) &&
                  nnf.size(1) == source_edit.size(2),
              "NNF spatial shape must match source_edit");
  TORCH_CHECK(patch_radius >= 0, "patch_radius must be non-negative");
  return vote_cuda(source_edit.contiguous(), nnf.contiguous(), patch_radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("patchmatch", &patchmatch,
             "guided jump-flood PatchMatch (CUDA)");
  module.def("vote", &vote, "overlapping patch voting (CUDA)");
}
