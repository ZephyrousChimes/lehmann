
#pragma once

namespace Lehmann {
class BankState {
public:
  BankState(): assets(1000.0f), liabilities(800.0f), liquidity(200.0f), failed(false) {}

  BankState(const double assets, const double liabilities, const double liquidity, const bool failed)
    : assets(assets), liabilities(liabilities), liquidity(liquidity), failed(failed) {}

  double assets;
  double liabilities;
  double liquidity;
  bool failed;
};

__global__ void MarkAllFailed(Lehmann::BankState* banks, int n);
__host__ void LaunchMarkAllFailed(Lehmann::BankState* d_banks, int n);
}
