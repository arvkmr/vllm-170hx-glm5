#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void moe_gemv_marlin_launch(
    const void* x, const int* w_packed, const void* scales,
    const int* zp_packed, float* y, const int* sorted_token_ids,
    const int* expert_ids, const float* topk_w,
    int K, int N, int G, int tok_block, int num_valid_tokens,
    long stride_we, long stride_se, long stride_ze,
    int mul_routed, int x_row_div, int slots, void* stream);

void moe_gemv_marlin(
    torch::Tensor x, torch::Tensor w_packed, torch::Tensor scales,
    torch::Tensor zp_packed, torch::Tensor y,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor topk_w, int64_t K, int64_t N, int64_t G,
    int64_t tok_block, int64_t num_valid_tokens,
    bool mul_routed, int64_t x_row_div)
{
  moe_gemv_marlin_launch(
      x.data_ptr(), w_packed.data_ptr<int>(), scales.data_ptr(),
      zp_packed.data_ptr<int>(), y.data_ptr<float>(),
      sorted_token_ids.data_ptr<int>(), expert_ids.data_ptr<int>(),
      topk_w.data_ptr<float>(), (int)K, (int)N, (int)G, (int)tok_block,
      (int)num_valid_tokens, w_packed.stride(0), scales.stride(0),
      zp_packed.stride(0), mul_routed ? 1 : 0, (int)x_row_div,
      (int)expert_ids.size(0),
      (void*)c10::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_gemv_marlin", &moe_gemv_marlin, "MoE GEMV on marlin layout");
}
