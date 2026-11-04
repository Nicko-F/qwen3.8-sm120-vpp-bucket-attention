// SPDX-License-Identifier: Apache-2.0
#include "interface.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace vpp {
namespace {
constexpr std::uint32_t kMagic = 0x31505056;  // VPP1
constexpr std::uint16_t kVersion = 1;
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
void put(WireBoundary& wire, std::size_t offset, std::uint64_t value,
         unsigned bytes) {
  for (unsigned i = 0; i != bytes; ++i)
    wire[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
}
std::uint64_t get(const WireBoundary& wire, std::size_t offset, unsigned bytes) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i != bytes; ++i)
    value |= std::uint64_t(wire[offset + i]) << (8 * i);
  return value;
}
}  // namespace

void BoundaryLayout::validate() const {
  require(token_capacity > 0 && token_capacity <= 16384, "invalid token capacity");
  require(slot_count > 0 && slot_count <= 7, "invalid admitted boundary capacity");
}
std::uint64_t BoundaryLayout::plane_bytes() const {
  validate();
  return std::uint64_t(token_capacity) * kHiddenSize * 2;
}
std::uint64_t BoundaryLayout::slot_bytes() const { return 2 * plane_bytes(); }

std::uint32_t stage_rank(std::uint32_t stage) {
  require(stage < 8, "invalid virtual stage");
  return stage < 4 ? stage : 7 - stage;
}
std::uint32_t edge_index(std::uint32_t src, std::uint32_t dst) {
  for (std::uint32_t i = 0; i < kSourceStages.size(); ++i)
    if (src == kSourceStages[i] && dst == src + 1) return i;
  throw std::invalid_argument("not a remote folded edge");
}
void validate_boundary(const BoundaryLayout& layout, const BoundaryDesc& d) {
  layout.validate();
  edge_index(d.src_stage, d.dst_stage);
  require(d.epoch != 0, "role epoch must be nonzero");
  require(d.edge_sequence != std::numeric_limits<std::uint64_t>::max(),
          "edge sequence exhausted; restart role epoch, never wrap");
  require(d.slot_id == d.edge_sequence % layout.slot_count, "wrong ring slot");
  require(d.generation == d.edge_sequence / layout.slot_count + 1,
          "wrong slot generation");
  require(d.valid_tokens <= layout.token_capacity, "boundary exceeds capacity");
  const auto bytes = std::uint64_t(d.valid_tokens) * kHiddenSize * 2;
  require(d.hidden.offset_bytes == 0 && d.hidden.valid_bytes == bytes,
          "invalid hidden span");
  require(d.residual.offset_bytes == layout.plane_bytes() &&
          d.residual.valid_bytes == bytes, "invalid residual span");
}
BoundaryDesc make_boundary(const BoundaryLayout& layout, std::uint64_t epoch,
                          std::int64_t batch, std::uint64_t sequence,
                          std::uint32_t src, std::uint32_t rows) {
  layout.validate();
  const auto bytes = std::uint64_t(rows) * kHiddenSize * 2;
  BoundaryDesc d{epoch, batch, sequence, sequence / layout.slot_count + 1,
    src, src + 1, static_cast<std::uint32_t>(sequence % layout.slot_count),
    rows, {0, bytes}, {layout.plane_bytes(), bytes}};
  validate_boundary(layout, d);
  return d;
}
WireBoundary encode_boundary(const BoundaryLayout& layout, const BoundaryDesc& d) {
  validate_boundary(layout, d);
  WireBoundary wire{};
  put(wire, 0, kMagic, 4); put(wire, 4, kVersion, 2); put(wire, 6, kDualBF16, 2);
  put(wire, 8, d.epoch, 8);
  std::uint64_t batch_bits;
  std::memcpy(&batch_bits, &d.batch_id, sizeof(batch_bits));
  put(wire, 16, batch_bits, 8); put(wire, 24, d.edge_sequence, 8);
  put(wire, 32, d.generation, 8); put(wire, 40, d.src_stage, 4);
  put(wire, 44, d.dst_stage, 4); put(wire, 48, d.slot_id, 4);
  put(wire, 52, d.valid_tokens, 4); put(wire, 56, d.hidden.offset_bytes, 8);
  put(wire, 64, d.hidden.valid_bytes, 8); put(wire, 72, d.residual.offset_bytes, 8);
  put(wire, 80, d.residual.valid_bytes, 8);
  return wire;
}
BoundaryDesc decode_boundary(const BoundaryLayout& layout, const WireBoundary& w) {
  require(get(w, 0, 4) == kMagic && get(w, 4, 2) == kVersion &&
          get(w, 6, 2) == kDualBF16 && get(w, 88, 8) == 0,
          "unsupported boundary wire header");
  const auto bits = get(w, 16, 8);
  std::int64_t batch;
  std::memcpy(&batch, &bits, sizeof(batch));
  BoundaryDesc d{get(w, 8, 8), batch, get(w, 24, 8), get(w, 32, 8),
    static_cast<std::uint32_t>(get(w, 40, 4)),
    static_cast<std::uint32_t>(get(w, 44, 4)),
    static_cast<std::uint32_t>(get(w, 48, 4)),
    static_cast<std::uint32_t>(get(w, 52, 4)),
    {get(w, 56, 8), get(w, 64, 8)}, {get(w, 72, 8), get(w, 80, 8)}};
  validate_boundary(layout, d);
  return d;
}
BoundaryDesc match_boundary(const BoundaryLayout& layout, const WireBoundary& wire,
    std::uint64_t epoch, std::int64_t batch, std::uint64_t sequence, std::uint32_t stage) {
  const auto d = decode_boundary(layout, wire);
  require(d.epoch == epoch && d.batch_id == batch &&
          d.edge_sequence == sequence && d.src_stage == stage,
          "boundary does not match the admitted stage identity");
  return d;
}
std::array<std::uint64_t, 4> rank_boundary_budget(const BoundaryLayout& layout) {
  const auto family = layout.slot_count * layout.slot_bytes();
  std::array<std::uint64_t, 4> bytes{};
  for (auto src : kSourceStages) {
    bytes[stage_rank(src)] += family;
    bytes[stage_rank(src + 1)] += family;
  }
  return bytes;
}
}  // namespace vpp
