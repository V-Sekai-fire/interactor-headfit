struct HfSimParams {
  uint P;
};

[[vk::binding(0, 0)]]
ConstantBuffer<HfSimParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> prm;
[[vk::binding(2, 0)]]
StructuredBuffer<float> dpos;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> dv;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if ((i >= params.P)) {
    return;
  }
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
  float g0 = dpos[((i * 3u) + 0u)];
  float g1 = dpos[((i * 3u) + 1u)];
  float g2 = dpos[((i * 3u) + 2u)];
  float rg0 = (((R0 * g0) + (R3 * g1)) + (R6 * g2));
  float rg1 = (((R1 * g0) + (R4 * g1)) + (R7 * g2));
  float rg2 = (((R2 * g0) + (R5 * g1)) + (R8 * g2));
  float sc = prm[9u];
  dv[((i * 3u) + 0u)] = (sc * rg0);
  dv[((i * 3u) + 1u)] = (sc * rg1);
  dv[((i * 3u) + 2u)] = (sc * rg2);
}
