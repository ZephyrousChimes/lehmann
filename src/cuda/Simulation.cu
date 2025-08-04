#include "cuda/Simulation.cuh"

namespace Lehmann {
  __global__ void SimulateStep(
    BankState *banks,
    const int *exposure_targets,
    const double *exposure_weights,
    const int K,
    const int n
  ) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    BankState& me = banks[idx];

    if (me.failed) return; // already failed, skip

    float loss = 0.0f;

    // Loop through my K exposures
    for (int j = 0; j < K; ++j) {
      int target_bank_id = exposure_targets[idx * K + j];
      double weight = exposure_weights[idx * K + j];

      if (banks[target_bank_id].failed) {
        loss += me.assets * weight;  // lose proportional value
      }
    }

    me.assets -= loss;

    if (me.assets < me.liabilities) {
      me.failed = true;
    }

  }
}


