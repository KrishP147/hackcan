#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

__device__ __forceinline__ float patch_distance(
    const float* __restrict__ source,
    const float* __restrict__ target,
    const float* __restrict__ weights,
    int channels,
    int source_x,
    int source_y,
    int target_x,
    int target_y,
    int height,
    int width,
    int radius) {
  float cost = 0.0f;
  int valid_count = 0;
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const int sx = source_x + dx;
      const int sy = source_y + dy;
      const int tx = target_x + dx;
      const int ty = target_y + dy;
      if (sx < 0 || sx >= width || sy < 0 || sy >= height ||
          tx < 0 || tx >= width || ty < 0 || ty >= height) {
        cost += 10.0f;
        continue;
      }
      ++valid_count;
      for (int channel = 0; channel < channels; ++channel) {
        const float a = source[(channel * height + sy) * width + sx];
        const float b = target[(channel * height + ty) * width + tx];
        const float delta = a - b;
        cost += weights[channel] * delta * delta;
      }
    }
  }
  return cost / static_cast<float>(max(valid_count, 1));
}

__global__ void initialize_cost_kernel(
    const float* __restrict__ source,
    const float* __restrict__ target,
    const float* __restrict__ weights,
    const int32_t* __restrict__ nnf,
    float* __restrict__ cost,
    int channels,
    int height,
    int width,
    int radius) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx >= width || ty >= height) return;
  const int index = ty * width + tx;
  const int sx = nnf[index * 2];
  const int sy = nnf[index * 2 + 1];
  cost[index] = patch_distance(source, target, weights, channels,
                               sx, sy, tx, ty, height, width, radius);
}

__global__ void jump_flood_kernel(
    const float* __restrict__ source,
    const float* __restrict__ target,
    const float* __restrict__ weights,
    const int32_t* __restrict__ input_nnf,
    const float* __restrict__ input_cost,
    int32_t* __restrict__ output_nnf,
    float* __restrict__ output_cost,
    int channels,
    int height,
    int width,
    int radius,
    int step) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx >= width || ty >= height) return;
  const int index = ty * width + tx;

  int best_x = input_nnf[index * 2];
  int best_y = input_nnf[index * 2 + 1];
  float best_cost = input_cost[index];
  const int offsets[8][2] = {
      {-step, 0}, {step, 0}, {0, -step}, {0, step},
      {-step, -step}, {step, step}, {-step, step}, {step, -step}};

#pragma unroll
  for (int candidate_index = 0; candidate_index < 8; ++candidate_index) {
    const int nx = tx + offsets[candidate_index][0];
    const int ny = ty + offsets[candidate_index][1];
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
    const int neighbor = ny * width + nx;
    const int sx = input_nnf[neighbor * 2] + (tx - nx);
    const int sy = input_nnf[neighbor * 2 + 1] + (ty - ny);
    if (sx < 0 || sx >= width || sy < 0 || sy >= height) continue;
    const float candidate_cost = patch_distance(
        source, target, weights, channels, sx, sy, tx, ty,
        height, width, radius);
    if (candidate_cost < best_cost) {
      best_cost = candidate_cost;
      best_x = sx;
      best_y = sy;
    }
  }

  output_nnf[index * 2] = best_x;
  output_nnf[index * 2 + 1] = best_y;
  output_cost[index] = best_cost;
}

__device__ __forceinline__ uint32_t mix_bits(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__global__ void random_search_kernel(
    const float* __restrict__ source,
    const float* __restrict__ target,
    const float* __restrict__ weights,
    int32_t* __restrict__ nnf,
    float* __restrict__ cost,
    int channels,
    int height,
    int width,
    int radius,
    uint32_t seed) {
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx >= width || ty >= height) return;
  const int index = ty * width + tx;
  int best_x = nnf[index * 2];
  int best_y = nnf[index * 2 + 1];
  float best_cost = cost[index];

  for (int search_radius = max(width, height); search_radius >= 1;
       search_radius >>= 1) {
    const uint32_t hx = mix_bits(seed ^ static_cast<uint32_t>(index) ^
                                 static_cast<uint32_t>(search_radius * 0x9e3779b9U));
    const uint32_t hy = mix_bits(hx ^ 0x85ebca6bU);
    const int span = search_radius * 2 + 1;
    const int sx = min(max(best_x + static_cast<int>(hx % span) - search_radius, 0),
                       width - 1);
    const int sy = min(max(best_y + static_cast<int>(hy % span) - search_radius, 0),
                       height - 1);
    const float candidate_cost = patch_distance(
        source, target, weights, channels, sx, sy, tx, ty,
        height, width, radius);
    if (candidate_cost < best_cost) {
      best_cost = candidate_cost;
      best_x = sx;
      best_y = sy;
    }
  }

  nnf[index * 2] = best_x;
  nnf[index * 2 + 1] = best_y;
  cost[index] = best_cost;
}

}  // namespace

std::vector<torch::Tensor> patchmatch_cuda(
    torch::Tensor source,
    torch::Tensor target,
    torch::Tensor weights,
    torch::Tensor initial_nnf,
    int64_t patch_radius,
    int64_t iterations,
    int64_t seed) {
  const c10::cuda::CUDAGuard device_guard(source.device());
  const int channels = static_cast<int>(source.size(0));
  const int height = static_cast<int>(source.size(1));
  const int width = static_cast<int>(source.size(2));
  auto nnf = initial_nnf.clone();
  auto cost = torch::empty({height, width}, source.options());

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  const auto stream = at::cuda::getCurrentCUDAStream(source.device().index());

  initialize_cost_kernel<<<grid, block, 0, stream>>>(
      source.data_ptr<float>(), target.data_ptr<float>(), weights.data_ptr<float>(),
      nnf.data_ptr<int32_t>(), cost.data_ptr<float>(), channels, height, width,
      static_cast<int>(patch_radius));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  for (int iteration = 0; iteration < iterations; ++iteration) {
    int step = 1;
    while (step * 2 < std::max(height, width)) step *= 2;
    for (; step >= 1; step >>= 1) {
      auto next_nnf = torch::empty_like(nnf);
      auto next_cost = torch::empty_like(cost);
      jump_flood_kernel<<<grid, block, 0, stream>>>(
          source.data_ptr<float>(), target.data_ptr<float>(), weights.data_ptr<float>(),
          nnf.data_ptr<int32_t>(), cost.data_ptr<float>(),
          next_nnf.data_ptr<int32_t>(), next_cost.data_ptr<float>(),
          channels, height, width, static_cast<int>(patch_radius), step);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      nnf = next_nnf;
      cost = next_cost;
    }
    random_search_kernel<<<grid, block, 0, stream>>>(
        source.data_ptr<float>(), target.data_ptr<float>(), weights.data_ptr<float>(),
        nnf.data_ptr<int32_t>(), cost.data_ptr<float>(), channels, height, width,
        static_cast<int>(patch_radius),
        static_cast<uint32_t>(seed + iteration * 104729));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {nnf, cost};
}
