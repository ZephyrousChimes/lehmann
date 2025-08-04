//
// Created by kronos on 8/3/25.
//

#include <cstdio>
#include <iostream>

#include "core/config.h"
#include "cuda/Bank.cuh"

int main() {
  Lehmann::BankState* h_banks = new Lehmann::BankState[Lehmann::NUM_BANKS];

  int *h_exposure_targets = new int[Lehmann::NUM_BANKS * Lehmann::EXPOSURE];
  float *h_exposure_weights = new float[Lehmann::NUM_BANKS * Lehmann::EXPOSURE];

  for (int i = 0; i < Lehmann::NUM_BANKS; ++i) {
    for (int j = 0; j < Lehmann::EXPOSURE; ++j) {
      int neighbor = (i + j + 1) % Lehmann::NUM_BANKS; // next few banks in sequence
      h_exposure_targets[i * Lehmann::EXPOSURE + j] = neighbor;
      h_exposure_weights[i * Lehmann::EXPOSURE + j] = 0.05f; // each exposure is 5%
    }
  }

  int* d_exposure_targets;
  float* d_exposure_weights;

  cudaMalloc(&d_exposure_targets, Lehmann::NUM_BANKS * Lehmann::EXPOSURE * sizeof(int));
  cudaMalloc(&d_exposure_weights, Lehmann::NUM_BANKS * Lehmann::EXPOSURE * sizeof(float));

  cudaMemcpy(d_exposure_targets, h_exposure_targets, Lehmann::NUM_BANKS * Lehmann::EXPOSURE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_exposure_weights, h_exposure_weights, Lehmann::NUM_BANKS * Lehmann::EXPOSURE * sizeof(float), cudaMemcpyHostToDevice);

  for (int i = 0; i < 5; ++i) {
    std::cout << "Bank " << i << " exposed to: ";
    for (int j = 0; j < Lehmann::EXPOSURE; ++j) {
      std::cout << h_exposure_targets[i * Lehmann::EXPOSURE + j] << " ";
    }
    std::cout << "\n";
  }


  delete[] h_banks;
  delete[] d_exposure_targets;
  delete[] d_exposure_weights;
  cudaFree(d_exposure_targets);
  cudaFree(d_exposure_weights);

  return 0;
}