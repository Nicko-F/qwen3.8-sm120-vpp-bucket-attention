// SPDX-License-Identifier: Apache-2.0
#include "host_region.h"
#include <cerrno>
#include <climits>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unistd.h>

using namespace vpp;
template<class F> void rejects(F fn) {
  try { fn(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid host-region operation accepted");
}
int main() {
  try {
    const auto bytes = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
    rejects([&]{ HostRegion::create(0); });
    rejects([&]{ HostRegion::create(bytes + 1); });
    rejects([&]{ HostRegion::create(bytes, INT_MAX); });
    rejects([&]{ HostRegion::create(bytes, -2); });
    rejects([&]{ HostRegion::create(bytes, -1, std::numeric_limits<std::uint64_t>::max()); });
    auto region = HostRegion::create(bytes);  // Inherit placement; reserve defaults to zero.
    if (region->handle().numa_node != -1) throw std::runtime_error("inherited placement changed");
    auto peer = HostRegion::attach(region->handle());
    auto* source = static_cast<unsigned char*>(region->range(0, bytes));
    auto* destination = static_cast<unsigned char*>(peer->range(0, bytes));
    source[0] = 71;
    if (destination[0] != 71) throw std::runtime_error("shared backing changed");
    rejects([&]{ region->range(bytes, 1); });
    std::cout << "shared host allocation, inherited NUMA and optional reserve passed\n";
  } catch (const std::system_error& error) {
    if (error.code().value() == EPERM || error.code().value() == ENOMEM) {
      std::cerr << "mlock unavailable under this host policy: " << error.what() << '\n';
      return 77;
    }
    throw;
  }
}
