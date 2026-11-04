// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "device_pool.h"
#include "host_region.h"
#include <memory>
#include <string>

namespace vpp {

class DmaProgress;

// Same boundary ABI, whole reference or bounded segmented host ring.
// Neither mode uses peer-memory IPC or host-host payload copies.
class DmaEdge {
 public:
  static constexpr std::uint32_t kSegments = 8;
  static std::shared_ptr<HostRegion> create_region(BoundaryLayout, int numa_node,
                                                  std::uint64_t segment_bytes = 0,
                                                  std::uint64_t minimum_available_bytes = 0);
  DmaEdge(std::shared_ptr<DeviceSlotPool>, std::shared_ptr<HostRegion>,
          c10::cuda::CUDAStream direction_stream, std::uint64_t segment_bytes = 0,
          DmaProgress* progress = nullptr);
  ~DmaEdge() noexcept;
  DmaEdge(const DmaEdge&) = delete;
  DmaEdge& operator=(const DmaEdge&) = delete;
  std::string export_event() const;
  void connect(const std::string& peer_event);
  SlotTicket reserve(const WireBoundary&, std::uint32_t timeout_ms);
  std::optional<WireBoundary> peek_header();
  void submit(const SlotTicket&, bool independent_ready = false);
  // Wait only for this ticket's final EventRecord submission, not GPU completion.
  void wait_recorded(const SlotTicket&, std::uint32_t timeout_ms);
  // Cold policy change only: no live slots/jobs or pending work on either stream.
  // Allows matched shared-vs-per-edge receive queues without reallocating payloads.
  void set_receive_stream(c10::cuda::CUDAStream);
  void mark(c10::cuda::CUDAEvent&);
  std::uint32_t progress();
  // Cold two-phase shutdown: drain ALL peers, STOP the rank progress engine,
  // disconnect imported events on
  // ALL peers, caller barrier, then close local exports/registrations/pools.
  void disconnect();
  void close();
  void abort() noexcept;
  std::uint64_t host_bytes() const;
  std::string numa_receipt() const;

 private:
  friend class DmaProgress;
  struct State;
  void check() const;
  void harvest(bool wait);
  void wait_sequence(std::uint32_t* counter, std::uint32_t expected);
  std::unique_ptr<State> state_;
  std::uint64_t next_sequence_ = 0;
  bool connected_ = false, disconnected_ = false, failed_ = false;
};

// Exactly one optional progress thread per rank. Pool metadata remains on its
// original host owner; the worker only sees immutable jobs and CUDA handles.
class DmaProgress {
 public:
  struct Stats {
    std::uint64_t segments = 0, idle_waits = 0, peak_jobs = 0;
    double cpu_seconds = 0;
    int tid = 0;
  };
  explicit DmaProgress(c10::DeviceIndex);
  ~DmaProgress() noexcept;
  void start();
  void stop() noexcept;
  std::uint32_t progress(); // Harvest every endpoint sharing rank-level credits.
  Stats stats() const;  // after stop, outside measured rounds
 private:
  friend class DmaEdge;
  struct Impl;
  void attach(DmaEdge::State*);
  void submit(DmaEdge::State&, const SlotTicket&, const BoundaryDesc&,
              const WireBoundary&, char*, cudaEvent_t, cudaEvent_t,
              bool independent, std::array<cudaEvent_t,2>, const std::atomic<std::uint32_t>*);
  std::array<std::optional<SlotTicket>,7> collect(DmaEdge::State&, bool wait,
                                                const SlotTicket* target = nullptr,
                                                std::uint32_t timeout_ms = 30000);
  std::unique_ptr<Impl> impl_;
};

}  // namespace vpp
