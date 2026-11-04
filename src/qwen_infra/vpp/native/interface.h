// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>

namespace vpp {

// Local view; NEVER transmit sizeof(BoundaryDesc) or a process-local pointer.
// Wire encoding is explicit little-endian, independent of struct padding.
struct TensorSpan {
  std::uint64_t offset_bytes;
  std::uint64_t valid_bytes;
};

struct BoundaryDesc {
  std::uint64_t epoch;
  std::int64_t batch_id;  // Negative native warmup IDs are valid.
  std::uint64_t edge_sequence;
  std::uint64_t generation;
  std::uint32_t src_stage;
  std::uint32_t dst_stage;
  std::uint32_t slot_id;
  std::uint32_t valid_tokens;
  TensorSpan hidden;
  TensorSpan residual;
};

struct BoundaryLayout {
  std::uint32_t token_capacity;
  std::uint32_t slot_count;
  void validate() const;
  std::uint64_t plane_bytes() const;
  std::uint64_t slot_bytes() const;
};

using WireBoundary = std::array<std::uint8_t, 96>;
constexpr std::uint32_t kHiddenSize = 5120;
constexpr std::uint16_t kDualBF16 = 1;
constexpr std::array<std::uint32_t, 6> kSourceStages{{0, 1, 2, 4, 5, 6}};

std::uint32_t stage_rank(std::uint32_t stage);
std::uint32_t edge_index(std::uint32_t src, std::uint32_t dst);
BoundaryDesc make_boundary(const BoundaryLayout&, std::uint64_t epoch,
                          std::int64_t batch_id, std::uint64_t edge_sequence,
                          std::uint32_t src_stage, std::uint32_t valid_tokens);
void validate_boundary(const BoundaryLayout&, const BoundaryDesc&);
WireBoundary encode_boundary(const BoundaryLayout&, const BoundaryDesc&);
BoundaryDesc decode_boundary(const BoundaryLayout&, const WireBoundary&);
// Match the admitted stage identity BEFORE allocating/submitting its payload.
BoundaryDesc match_boundary(const BoundaryLayout&, const WireBoundary&,
                            std::uint64_t epoch, std::int64_t batch_id,
                            std::uint64_t sequence, std::uint32_t src_stage);
// Conservative source+destination allocations, no model/metadata/Graph memory.
std::array<std::uint64_t, 4> rank_boundary_budget(const BoundaryLayout&);

}  // namespace vpp
