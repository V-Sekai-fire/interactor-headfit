struct HfMarkParams {
  uint K;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfMarkParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> pos;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> mid;
[[vk::binding(3, 0)]]
StructuredBuffer<float> tgt;
[[vk::binding(4, 0)]]
StructuredBuffer<float> lw;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> e;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> dpos;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint k = tid.x;
  if ((k >= params.K)) {
    return;
  }
  uint o = (mid[k] * 3u);
  float dx = (pos[o] - tgt[(k * 3u)]);
  float dy = (pos[(o + 1u)] - tgt[((k * 3u) + 1u)]);
  float dz = (pos[(o + 2u)] - tgt[((k * 3u) + 2u)]);
  float w = lw[k];
  e[k] = (w * (((dx * dx) + (dy * dy)) + (dz * dz)));
  float b = (2.000000 * w);
  dpos[o] = (dpos[o] + (b * dx));
  dpos[(o + 1u)] = (dpos[(o + 1u)] + (b * dy));
  dpos[(o + 2u)] = (dpos[(o + 2u)] + (b * dz));
}
