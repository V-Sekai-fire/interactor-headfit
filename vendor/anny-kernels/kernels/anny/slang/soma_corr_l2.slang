struct SomaCorrL2Params {
  uint V;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SomaCorrL2Params> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> rest;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> rowPtr;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> col;
[[vk::binding(4, 0)]]
StructuredBuffer<float> val;
[[vk::binding(5, 0)]]
StructuredBuffer<float> z;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> bind;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.V)) {
    return;
  }
  float x = 0.000000;
  float y = 0.000000;
  float w = 0.000000;
  for (uint e = rowPtr[i]; e < rowPtr[(i + 1u)]; ++e) {
    float zk = z[col[e]];
    x = (x + (zk * val[(e * 3u)]));
    y = (y + (zk * val[((e * 3u) + 1u)]));
    w = (w + (zk * val[((e * 3u) + 2u)]));
  }
  bind[(i * 3u)] = (rest[(i * 3u)] + x);
  bind[((i * 3u) + 1u)] = (rest[((i * 3u) + 1u)] + y);
  bind[((i * 3u) + 2u)] = (rest[((i * 3u) + 2u)] + w);
}
