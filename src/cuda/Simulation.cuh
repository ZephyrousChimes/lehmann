//
// Created by kronos on 8/3/25.
//

#pragma once

#include "cuda/Bank.cuh"

namespace Lehmann {
__global__ void SimulateStep(
    const BankState* current,
    BankState* next,
    const int* exposure_targets,
    const float* exposure_weights,
    int K,
    int n);
}


