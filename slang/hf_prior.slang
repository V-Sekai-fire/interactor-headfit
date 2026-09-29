struct HfPriorParams {
  uint N;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfPriorParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> x;
[[vk::binding(2, 0)]]
StructuredBuffer<float> x0;
[[vk::binding(3, 0)]]
StructuredBuffer<float> lam;
[[vk::binding(4, 0)]]
RWStructuredBuffer<float> e;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> g;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint j = tid.x;
  if ((j >= params.N)) {
    return;
  }
  float d = (x[j] - x0[j]);
  float l = lam[j];
  e[j] = ((l * d) * d);
  g[j] = ((2.000000 * l) * d);
}
