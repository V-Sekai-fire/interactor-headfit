struct SomaCorrL1Params {
  uint D;
  uint K;
  uint useTanh;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SomaCorrL1Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> feat;
[[vk::binding(2, 0)]]
StructuredBuffer<float> w1;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> z;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint k = tid.x;
  if ((k >= params.K)) {
    return;
  }
  float a = 0.000000;
  for (uint d = 0u; d < params.D; ++d) {
    a = (a + (feat[d] * w1[((d * params.K) + k)]));
  }
  a = max(a, 0.000000);
  if ((params.useTanh != 0u)) {
    a = tanh(a);
  }
  z[k] = a;
}
