//
// Created by kronos on 8/3/25.
//

#pragma once

#include "cuda/Bank.cuh"

namespace Lehmann {
__global__ void simulate_step(
    BankState* banks,
    int* exposure_targets,
    double* exposure_weights,
    int K,
    int n);
}


