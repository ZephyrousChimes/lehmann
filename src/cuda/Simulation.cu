#include "cuda/Simulation.cuh"

namespace Lehmann {
  __global__ void SimulateStep(
  const BankState* current,
  BankState* next,
  const int* exposure_targets,
  const float* exposure_weights,
  int K,
  int n)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    BankState& me = next[idx];

    if (me.failed) return;

    float loss = 0.0f;

    // Loop through my K exposures
    for (int j = 0; j < K; ++j) {
      int target_bank_id = exposure_targets[idx * K + j];
      double weight = exposure_weights[idx * K + j];

      if (current[target_bank_id].failed) {
        loss += me.assets * weight;
      }
    }

    me.assets -= loss;

    if (me.assets < me.liabilities) {
      me.failed = true;
    }

  }
}


