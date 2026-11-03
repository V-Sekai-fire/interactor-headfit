struct AnnyGroundParams {
  uint N;
  uint reset;
  uint anchor0;
  uint anchor1;
  uint anchor2;
  uint has_seat;
  float floor_y;
  float hyst;
  float seat_y;
  float seat_cx;
  float seat_cz;
  float seat_ux;
  float seat_uz;
  float half_d;
  float half_w;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyGroundParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> world;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> point_bone;
[[vk::binding(3, 0)]]
StructuredBuffer<float> point_off;
[[vk::binding(4, 0)]]
StructuredBuffer<uint> kind;
[[vk::binding(5, 0)]]
RWStructuredBuffer<float> state;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> out;

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  float rx0 = ((params.reset == 1u) ? 0.000000 : state[3u]);
  float rz0 = ((params.reset == 1u) ? 0.000000 : state[5u]);
  float y0 = asfloat(2139095039u);
  float y1 = asfloat(2139095039u);
  float yb = asfloat(2139095039u);
  float ys = asfloat(2139095039u);
  for (uint i = 0u; i < params.N; ++i) {
    uint b = (point_bone[i] * 12u);
    float px = ((((world[(b + 0u)] * point_off[((i * 3u) + 0u)]) + (world[(b + 1u)] * point_off[((i * 3u) + 1u)])) + (world[(b + 2u)] * point_off[((i * 3u) + 2u)])) + world[(b + 9u)]);
    float py = ((((world[(b + 3u)] * point_off[((i * 3u) + 0u)]) + (world[(b + 4u)] * point_off[((i * 3u) + 1u)])) + (world[(b + 5u)] * point_off[((i * 3u) + 2u)])) + world[(b + 10u)]);
    float pz = ((((world[(b + 6u)] * point_off[((i * 3u) + 0u)]) + (world[(b + 7u)] * point_off[((i * 3u) + 1u)])) + (world[(b + 8u)] * point_off[((i * 3u) + 2u)])) + world[(b + 11u)]);
    out[(4u + (i * 3u))] = px;
    out[(5u + (i * 3u))] = py;
    out[(6u + (i * 3u))] = pz;
    if ((kind[i] == 0u)) {
      y0 = min(y0, py);
    } else {
      if ((kind[i] == 1u)) {
        y1 = min(y1, py);
      } else {
        yb = min(yb, py);
        float qx = ((px + rx0) - params.seat_cx);
        float qz = ((pz + rz0) - params.seat_cz);
        float along = ((qx * params.seat_ux) + (qz * params.seat_uz));
        float across = ((qx * params.seat_uz) - (qz * params.seat_ux));
        if (((params.has_seat != 0u) && ((abs(along) <= params.half_d) && (abs(across) <= params.half_w)))) {
          ys = min(ys, py);
        }
      }
    }
  }
  float dyf = (params.floor_y - min(min(y0, y1), yb));
  float dys = ((ys < asfloat(2139095039u)) ? (params.seat_y - ys) : (params.floor_y - asfloat(2139095039u)));
  float dy = max(dyf, dys);
  float prev = state[0u];
  float foot = ((params.reset != 0u) ? ((y1 < y0) ? 1.000000 : 0.000000) : state[6u]);
  if (((foot < 0.500000) && (y1 < (y0 - params.hyst)))) {
    foot = 1.000000;
  } else {
    if (((foot > 0.500000) && (y0 < (y1 - params.hyst)))) {
      foot = 0.000000;
    }
  }
  float s = ((dys > dyf) ? 2.000000 : foot);
  uint a = ((s > 1.500000) ? params.anchor2 : ((s > 0.500000) ? params.anchor1 : params.anchor0));
  float ax = out[(4u + (a * 3u))];
  float az = out[(6u + (a * 3u))];
  if (((params.reset != 0u) || (s != prev))) {
    state[1u] = (ax + rx0);
    state[2u] = (az + rz0);
  }
  float rootx = (state[1u] - ax);
  float rootz = (state[2u] - az);
  state[0u] = s;
  state[3u] = rootx;
  state[4u] = dy;
  state[5u] = rootz;
  state[6u] = foot;
  out[0u] = rootx;
  out[1u] = dy;
  out[2u] = rootz;
  out[3u] = s;
  for (uint i = 0u; i < params.N; ++i) {
    out[(4u + (i * 3u))] = (out[(4u + (i * 3u))] + rootx);
    out[(5u + (i * 3u))] = (out[(5u + (i * 3u))] + dy);
    out[(6u + (i * 3u))] = (out[(6u + (i * 3u))] + rootz);
  }
}
