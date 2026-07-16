#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void vote_kernel(
    const float* __restrict__ source,
    const int32_t* __restrict__ nnf,
    float* __restrict__ output,
    int channels,
    int height,
    int width,
    int radius) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx >= width || ty >= height) return;

  int votes = 0;
  for (int channel = 0; channel < channels; ++channel) {
    float value = 0.0f;
    int channel_votes = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        const int patch_x = tx + dx;
        const int patch_y = ty + dy;
        if (patch_x < 0 || patch_x >= width ||
            patch_y < 0 || patch_y >= height) continue;
        const int patch_index = patch_y * width + patch_x;
        const int source_x = nnf[patch_index * 2] - dx;
        const int source_y = nnf[patch_index * 2 + 1] - dy;
        if (source_x < 0 || source_x >= width ||
            source_y < 0 || source_y >= height) continue;
        value += source[(channel * height + source_y) * width + source_x];
        ++channel_votes;
      }
    }
    const int output_index = (channel * height + ty) * width + tx;
    output[output_index] = channel_votes > 0
        ? value / static_cast<float>(channel_votes)
        : 0.0f;
    votes = channel_votes;
  }
  (void)votes;
}

}  // namespace

torch::Tensor vote_cuda(
    torch::Tensor source_edit,
    torch::Tensor nnf,
    int64_t patch_radius) {
  const c10::cuda::CUDAGuard device_guard(source_edit.device());
  const int channels = static_cast<int>(source_edit.size(0));
  const int height = static_cast<int>(source_edit.size(1));
  const int width = static_cast<int>(source_edit.size(2));
  auto output = torch::zeros_like(source_edit);
  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  const auto stream = at::cuda::getCurrentCUDAStream(source_edit.device().index());
  vote_kernel<<<grid, block, 0, stream>>>(
      source_edit.data_ptr<float>(), nnf.data_ptr<int32_t>(),
      output.data_ptr<float>(), channels, height, width,
      static_cast<int>(patch_radius));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
