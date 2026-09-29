struct HfSurfParams {
  uint P;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfSurfParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> pos;
[[vk::binding(2, 0)]]
StructuredBuffer<float> q;
[[vk::binding(3, 0)]]
StructuredBuffer<float> n;
[[vk::binding(4, 0)]]
StructuredBuffer<float> w;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> e;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> dpos;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
  float dx = (pos[((i * 3u) + 0u)] - q[((i * 3u) + 0u)]);
  float dy = (pos[((i * 3u) + 1u)] - q[((i * 3u) + 1u)]);
  float dz = (pos[((i * 3u) + 2u)] - q[((i * 3u) + 2u)]);
  float nx = n[((i * 3u) + 0u)];
  float ny = n[((i * 3u) + 1u)];
  float nz = n[((i * 3u) + 2u)];
  float wp = w[(i * 2u)];
  float wq = w[((i * 2u) + 1u)];
  float pl = (((nx * dx) + (ny * dy)) + (nz * dz));
  float dd = (((dx * dx) + (dy * dy)) + (dz * dz));
  e[i] = (((wp * pl) * pl) + (wq * dd));
  float a = ((2.000000 * wp) * pl);
  float b = (2.000000 * wq);
  dpos[((i * 3u) + 0u)] = ((a * nx) + (b * dx));
  dpos[((i * 3u) + 1u)] = ((a * ny) + (b * dy));
  dpos[((i * 3u) + 2u)] = ((a * nz) + (b * dz));
}
