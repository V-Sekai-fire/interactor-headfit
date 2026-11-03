struct AnnyFkLocalParams {
  uint J;
};

[[vk::binding(0, 0)]]
ConstantBuffer<AnnyFkLocalParams> params;
[[vk::binding(1, 0)]]
StructuredBuffer<uint> parents;
[[vk::binding(2, 0)]]
StructuredBuffer<float> local;
[[vk::binding(3, 0)]]
RWStructuredBuffer<float> world;

[shader("compute")] [numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  if ((tid.x != 0u)) {
    return;
  }
  for (uint j = 0u; j < params.J; ++j) {
    uint b = (j * 12u);
    uint par = parents[j];
    if ((par == 4294967295u)) {
      for (uint k = 0u; k < 12u; ++k) {
        world[(b + k)] = local[(b + k)];
      }
    } else {
      uint p = (par * 12u);
      world[(b + 0u)] = (((world[(p + 0u)] * local[(b + 0u)]) + (world[(p + 1u)] * local[(b + 3u)])) + (world[(p + 2u)] * local[(b + 6u)]));
      world[(b + 1u)] = (((world[(p + 0u)] * local[(b + 1u)]) + (world[(p + 1u)] * local[(b + 4u)])) + (world[(p + 2u)] * local[(b + 7u)]));
      world[(b + 2u)] = (((world[(p + 0u)] * local[(b + 2u)]) + (world[(p + 1u)] * local[(b + 5u)])) + (world[(p + 2u)] * local[(b + 8u)]));
      world[(b + 3u)] = (((world[(p + 3u)] * local[(b + 0u)]) + (world[(p + 4u)] * local[(b + 3u)])) + (world[(p + 5u)] * local[(b + 6u)]));
      world[(b + 4u)] = (((world[(p + 3u)] * local[(b + 1u)]) + (world[(p + 4u)] * local[(b + 4u)])) + (world[(p + 5u)] * local[(b + 7u)]));
      world[(b + 5u)] = (((world[(p + 3u)] * local[(b + 2u)]) + (world[(p + 4u)] * local[(b + 5u)])) + (world[(p + 5u)] * local[(b + 8u)]));
      world[(b + 6u)] = (((world[(p + 6u)] * local[(b + 0u)]) + (world[(p + 7u)] * local[(b + 3u)])) + (world[(p + 8u)] * local[(b + 6u)]));
      world[(b + 7u)] = (((world[(p + 6u)] * local[(b + 1u)]) + (world[(p + 7u)] * local[(b + 4u)])) + (world[(p + 8u)] * local[(b + 7u)]));
      world[(b + 8u)] = (((world[(p + 6u)] * local[(b + 2u)]) + (world[(p + 7u)] * local[(b + 5u)])) + (world[(p + 8u)] * local[(b + 8u)]));
      world[(b + 9u)] = ((((world[(p + 0u)] * local[(b + 9u)]) + (world[(p + 1u)] * local[(b + 10u)])) + (world[(p + 2u)] * local[(b + 11u)])) + world[(p + 9u)]);
      world[(b + 10u)] = ((((world[(p + 3u)] * local[(b + 9u)]) + (world[(p + 4u)] * local[(b + 10u)])) + (world[(p + 5u)] * local[(b + 11u)])) + world[(p + 10u)]);
      world[(b + 11u)] = ((((world[(p + 6u)] * local[(b + 9u)]) + (world[(p + 7u)] * local[(b + 10u)])) + (world[(p + 8u)] * local[(b + 11u)])) + world[(p + 11u)]);
    }
  }
}
