struct HfResampleParams {
  uint N;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfResampleParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> prm;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> tri;
[[vk::binding(3, 0)]]
StructuredBuffer<uint> hit;
[[vk::binding(4, 0)]]
StructuredBuffer<float> bary;
[[vk::binding(5, 0)]]
StructuredBuffer<float> dm;
[[vk::binding(6, 0)]]
RWStructuredBuffer<float> out;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint j = tid.x;
  if ((j >= params.N)) {
    return;
  }
  uint h = hit[j];
  if ((h == 4294967295u)) {
    out[(j * 3u)] = 0.000000;
    out[((j * 3u) + 1u)] = 0.000000;
    out[((j * 3u) + 2u)] = 0.000000;
    return;
  }
  uint a = (tri[(h * 3u)] * 3u);
  uint b = (tri[((h * 3u) + 1u)] * 3u);
  uint c = (tri[((h * 3u) + 2u)] * 3u);
  float ba = bary[(j * 3u)];
  float bb = bary[((j * 3u) + 1u)];
  float bc = bary[((j * 3u) + 2u)];
  float d0 = (((ba * dm[(a + 0u)]) + (bb * dm[(b + 0u)])) + (bc * dm[(c + 0u)]));
  float d1 = (((ba * dm[(a + 1u)]) + (bb * dm[(b + 1u)])) + (bc * dm[(c + 1u)]));
  float d2 = (((ba * dm[(a + 2u)]) + (bb * dm[(b + 2u)])) + (bc * dm[(c + 2u)]));
  float R_a10 = prm[(0u + 0u)];
  float R_a11 = prm[(0u + 1u)];
  float R_a12 = prm[(0u + 2u)];
  float R_a20 = prm[(0u + 3u)];
  float R_a21 = prm[(0u + 4u)];
  float R_a22 = prm[(0u + 5u)];
  float R_n1 = max(sqrt((((R_a10 * R_a10) + (R_a11 * R_a11)) + (R_a12 * R_a12))), 1.0e-20f);
  float R_r10 = (R_a10 / R_n1);
  float R_r11 = (R_a11 / R_n1);
  float R_r12 = (R_a12 / R_n1);
  float R_d = (((R_r10 * R_a20) + (R_r11 * R_a21)) + (R_r12 * R_a22));
  float R_u0 = (R_a20 - (R_d * R_r10));
  float R_u1 = (R_a21 - (R_d * R_r11));
  float R_u2 = (R_a22 - (R_d * R_r12));
  float R_nu = max(sqrt((((R_u0 * R_u0) + (R_u1 * R_u1)) + (R_u2 * R_u2))), 1.0e-20f);
  float R_r20 = (R_u0 / R_nu);
  float R_r21 = (R_u1 / R_nu);
  float R_r22 = (R_u2 / R_nu);
  float R_r30 = ((R_r11 * R_r22) - (R_r12 * R_r21));
  float R_r31 = ((R_r12 * R_r20) - (R_r10 * R_r22));
  float R_r32 = ((R_r10 * R_r21) - (R_r11 * R_r20));
  float R0 = R_r10;
  float R1 = R_r11;
  float R2 = R_r12;
  float R3 = R_r20;
  float R4 = R_r21;
  float R5 = R_r22;
  float R6 = R_r30;
  float R7 = R_r31;
  float R8 = R_r32;
  float rd0 = (((R0 * d0) + (R1 * d1)) + (R2 * d2));
  float rd1 = (((R3 * d0) + (R4 * d1)) + (R5 * d2));
  float rd2 = (((R6 * d0) + (R7 * d1)) + (R8 * d2));
  float sc = prm[9u];
  out[((j * 3u) + 0u)] = (sc * rd0);
  out[((j * 3u) + 1u)] = (sc * rd1);
  out[((j * 3u) + 2u)] = (sc * rd2);
}
