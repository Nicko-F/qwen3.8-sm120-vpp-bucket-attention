// SPDX-License-Identifier: Apache-2.0
#include "interface.h"
#include "slot_pool.h"
#include <iostream>
#include <stdexcept>

using namespace vpp;
template<class F> void rejects(F fn) {
  try { fn(); } catch (const std::exception&) { return; }
  throw std::runtime_error("invalid lifecycle operation accepted");
}
void require(bool value) {
  if (!value) throw std::runtime_error("boundary lifecycle assertion failed");
}
int main() {
  BoundaryLayout layout{2048, 2};
  for (auto stage : kSourceStages) {
    auto desc = make_boundary(layout, 1, 12, 0, stage, 513);
    auto wire = encode_boundary(layout, desc);
    auto decoded = match_boundary(layout, wire, 1, 12, 0, stage);
    require(decoded.valid_tokens == 513 && wire.size() == 96);
    rejects([&]{ match_boundary(layout, wire, 1, 13, 0, stage); });
    rejects([&]{ match_boundary(layout, wire, 2, 12, 0, stage); });
  }
  SlotPool source(SlotRole::Source, layout, 1, 0);
  auto first = source.try_acquire(make_boundary(layout, 1, 0, 0, 0, 1024));
  require(first.has_value());
  rejects([&]{ source.source_read_complete(*first); });
  source.producer_recorded(*first);
  source.transfer_recorded(*first);
  source.source_read_complete(*first);
  require(source.live() == 0);
  rejects([&]{ source.descriptor(*first); });
  auto second = source.try_acquire(make_boundary(layout, 1, 1, 1, 0, 1024));
  require(second.has_value());
  source.producer_recorded(*second);
  source.transfer_recorded(*second);
  source.source_read_complete(*second);
  auto reused = source.try_acquire(make_boundary(layout, 1, 2, 2, 0, 1024));
  require(reused.has_value() && reused->generation != first->generation);
  rejects([&]{ source.producer_recorded(*first); });
  source.stop_admission();
  source.release_unsubmitted_on_stop(*reused);
  require(source.live() == 0);
  std::cout << "boundary wire and slot-generation lifecycle passed\n";
}
