struct AnnyCapsuleDepthParams {
  uint V;
  uint C;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyCapsuleDepthParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> points;
[[vk::binding(2, 0)]]
StructuredBuffer<float> capsules;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> depth;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.V)) {
    return;
  }
  float px = points[(i * 3u)];
  float py = points[((i * 3u) + 1u)];
  float pz = points[((i * 3u) + 2u)];
  float best = (0.000000 - asfloat(2139095039u));
  for (uint c = 0u; c < params.C; ++c) {
    uint b = (c * 7u);
    float dx = (capsules[(b + 3u)] - capsules[(b + 0u)]);
    float dy = (capsules[(b + 4u)] - capsules[(b + 1u)]);
    float dz = (capsules[(b + 5u)] - capsules[(b + 2u)]);
    float len2 = (((dx * dx) + (dy * dy)) + (dz * dz));
    float t = 0.000000;
    if ((len2 > 0.000000)) {
      t = min(1.000000, max(0.000000, (((((px - capsules[(b + 0u)]) * dx) + ((py - capsules[(b + 1u)]) * dy)) + ((pz - capsules[(b + 2u)]) * dz)) / len2)));
    }
    float ex = (px - (capsules[(b + 0u)] + (t * dx)));
    float ey = (py - (capsules[(b + 1u)] + (t * dy)));
    float ez = (pz - (capsules[(b + 2u)] + (t * dz)));
    best = max(best, (capsules[(b + 6u)] - sqrt((((ex * ex) + (ey * ey)) + (ez * ez)))));
  }
  depth[i] = best;
}
