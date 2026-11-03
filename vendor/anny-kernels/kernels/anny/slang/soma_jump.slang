struct SomaJumpParams {
  uint N;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SomaJumpParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<float> accIn;
[[vk::binding(2, 0)]]
StructuredBuffer<uint> ancIn;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> accOut;
[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> ancOut;

[shader("compute")] [numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint b = tid.x;
  if ((b >= params.N)) {
    return;
  }
  uint a = ancIn[b];
  uint ob = (b * 12u);
  float A0 = accIn[(ob + 0u)];
  float A1 = accIn[(ob + 1u)];
  float A2 = accIn[(ob + 2u)];
  float A3 = accIn[(ob + 4u)];
  float A4 = accIn[(ob + 5u)];
  float A5 = accIn[(ob + 6u)];
  float A6 = accIn[(ob + 8u)];
  float A7 = accIn[(ob + 9u)];
  float A8 = accIn[(ob + 10u)];
  float at0 = accIn[(ob + 3u)];
  float at1 = accIn[(ob + 7u)];
  float at2 = accIn[(ob + 11u)];
  if ((a == 4294967295u)) {
    accOut[(ob + 0u)] = A0;
    accOut[(ob + 1u)] = A1;
    accOut[(ob + 2u)] = A2;
    accOut[(ob + 4u)] = A3;
    accOut[(ob + 5u)] = A4;
    accOut[(ob + 6u)] = A5;
    accOut[(ob + 8u)] = A6;
    accOut[(ob + 9u)] = A7;
    accOut[(ob + 10u)] = A8;
    accOut[(ob + 3u)] = at0;
    accOut[(ob + 7u)] = at1;
    accOut[(ob + 11u)] = at2;
    ancOut[b] = 4294967295u;
  } else {
    uint oa = (a * 12u);
    float P0 = accIn[(oa + 0u)];
    float P1 = accIn[(oa + 1u)];
    float P2 = accIn[(oa + 2u)];
    float P3 = accIn[(oa + 4u)];
    float P4 = accIn[(oa + 5u)];
    float P5 = accIn[(oa + 6u)];
    float P6 = accIn[(oa + 8u)];
    float P7 = accIn[(oa + 9u)];
    float P8 = accIn[(oa + 10u)];
    float pt0 = accIn[(oa + 3u)];
    float pt1 = accIn[(oa + 7u)];
    float pt2 = accIn[(oa + 11u)];
    float C0 = (((P0 * A0) + (P1 * A3)) + (P2 * A6));
    float C1 = (((P0 * A1) + (P1 * A4)) + (P2 * A7));
    float C2 = (((P0 * A2) + (P1 * A5)) + (P2 * A8));
    float C3 = (((P3 * A0) + (P4 * A3)) + (P5 * A6));
    float C4 = (((P3 * A1) + (P4 * A4)) + (P5 * A7));
    float C5 = (((P3 * A2) + (P4 * A5)) + (P5 * A8));
    float C6 = (((P6 * A0) + (P7 * A3)) + (P8 * A6));
    float C7 = (((P6 * A1) + (P7 * A4)) + (P8 * A7));
    float C8 = (((P6 * A2) + (P7 * A5)) + (P8 * A8));
    float Pa0 = (((P0 * at0) + (P1 * at1)) + (P2 * at2));
    float Pa1 = (((P3 * at0) + (P4 * at1)) + (P5 * at2));
    float Pa2 = (((P6 * at0) + (P7 * at1)) + (P8 * at2));
    float c0 = (Pa0 + pt0);
    float c1 = (Pa1 + pt1);
    float c2 = (Pa2 + pt2);
    accOut[(ob + 0u)] = C0;
    accOut[(ob + 1u)] = C1;
    accOut[(ob + 2u)] = C2;
    accOut[(ob + 4u)] = C3;
    accOut[(ob + 5u)] = C4;
    accOut[(ob + 6u)] = C5;
    accOut[(ob + 8u)] = C6;
    accOut[(ob + 9u)] = C7;
    accOut[(ob + 10u)] = C8;
    accOut[(ob + 3u)] = c0;
    accOut[(ob + 7u)] = c1;
    accOut[(ob + 11u)] = c2;
    ancOut[b] = ancIn[a];
  }
}
