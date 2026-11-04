// SPDX-License-Identifier: Apache-2.0
#include "device_pool.h"
#include "nccl_edge.h"
#include "dma_edge.h"
#include "kv_copy.h"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <climits>
#include <sys/syscall.h>
#include <unistd.h>

namespace py = pybind11;
using namespace vpp;
using namespace pybind11::literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<KvCopyResult>(m,"KvCopyResult")
      .def_readonly("job_id",&KvCopyResult::job_id)
      .def_readonly("generation",&KvCopyResult::generation)
      .def_readonly("bytes",&KvCopyResult::bytes)
      .def_readonly("segments",&KvCopyResult::segments)
      .def_readonly("cancelled",&KvCopyResult::cancelled)
      .def_readonly("device_ms",&KvCopyResult::device_ms)
      .def_readonly("first_issue_ns",&KvCopyResult::first_issue_ns)
      .def_readonly("last_done_ns",&KvCopyResult::last_done_ns);
  py::class_<KvCopy>(m,"KvCopy")
      .def(py::init<c10::DeviceIndex,std::vector<at::Tensor>,at::Tensor,
                    std::uint64_t,std::uint32_t,std::uint32_t,bool>(),
           "device"_a,"gpu_tensors"_a,"host_payload"_a,"segment_bytes"_a=8ULL<<20,
           "max_jobs"_a=8,"max_spans"_a=1U<<20,"pitched"_a=true)
      .def("submit",[](KvCopy& e,std::uint64_t id,std::uint64_t generation,
                       const at::Tensor& spans,bool store,std::uint64_t stream,c10::DeviceIndex device) {
        const auto queue=c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream),device);
        py::gil_scoped_release release; return e.submit(id,generation,spans,store,queue);
      })
      .def("set_ready",&KvCopy::set_ready,py::call_guard<py::gil_scoped_release>())
      .def("submit_ready",[](KvCopy& e,std::uint64_t id,std::uint64_t generation,
                       const at::Tensor& spans,bool store,std::uint64_t stream,c10::DeviceIndex device,
                       const at::Tensor& gates) {
        const auto queue=c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream),device);
        py::gil_scoped_release release; return e.submit(id,generation,spans,store,queue,gates);
      })
      .def("cancel",&KvCopy::cancel,py::call_guard<py::gil_scoped_release>())
      .def("poll",&KvCopy::poll,py::call_guard<py::gil_scoped_release>())
      .def("snapshot",&KvCopy::snapshot,py::call_guard<py::gil_scoped_release>())
      .def("wait",&KvCopy::wait,"ids"_a,"timeout_ms"_a=30000,py::call_guard<py::gil_scoped_release>())
      .def("drain",&KvCopy::drain,"timeout_ms"_a=30000,py::call_guard<py::gil_scoped_release>())
      .def("close",&KvCopy::close,py::call_guard<py::gil_scoped_release>());
  py::enum_<SlotRole>(m, "SlotRole")
      .value("Source", SlotRole::Source).value("Destination", SlotRole::Destination);
  py::class_<SlotTicket>(m, "SlotTicket")
      .def_readonly("slot_id", &SlotTicket::slot_id)
      .def_readonly("generation", &SlotTicket::generation)
      .def_readonly("sequence", &SlotTicket::sequence);
  m.def("boundary", [](std::uint32_t capacity, std::uint32_t slots, std::uint64_t epoch,
                       std::int64_t batch, std::uint64_t sequence, std::uint32_t stage,
                       std::uint32_t rows) {
    BoundaryLayout layout{capacity, slots};
    const auto bytes = encode_boundary(layout, make_boundary(layout, epoch, batch, sequence, stage, rows));
    return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  });
  m.def("check_boundary", [](std::uint32_t capacity, std::uint32_t slots,
                            std::uint64_t epoch, std::int64_t batch,
                            std::uint64_t sequence, std::uint32_t stage, py::bytes input) {
    const std::string data = input;
    TORCH_CHECK(data.size() == WireBoundary{}.size(), "expected fixed96-byte boundary");
    WireBoundary wire;
    std::copy(data.begin(), data.end(), wire.begin());
    return match_boundary({capacity, slots}, wire, epoch, batch, sequence, stage).valid_tokens;
  });
  py::class_<DeviceSlotPool, std::shared_ptr<DeviceSlotPool>>(m, "DeviceSlotPool")
      .def(py::init([](SlotRole role, std::uint32_t capacity, std::uint32_t slots,
                      std::uint64_t epoch, std::uint32_t stage, c10::DeviceIndex device) {
        return std::make_shared<DeviceSlotPool>(role, BoundaryLayout{capacity, slots}, epoch, stage, device);
      }))
      .def("reserve", [](DeviceSlotPool& pool, py::bytes input) {
        const std::string data = input;
        TORCH_CHECK(data.size() == WireBoundary{}.size(), "expected fixed96-byte boundary");
        WireBoundary wire;
        std::copy(data.begin(), data.end(), wire.begin());
        return pool.reserve(wire);
      })
      .def("publish", &DeviceSlotPool::publish)
      .def("producer_views", &DeviceSlotPool::producer_views)
      .def("publish_written", &DeviceSlotPool::publish_written)
      .def("publication_token", &DeviceSlotPool::publication_token)
      .def("loopback", &DeviceSlotPool::loopback)
      .def("consume", &DeviceSlotPool::consume)
      .def("record_consumer_done", &DeviceSlotPool::record_consumer_done)
      .def("discard_received", &DeviceSlotPool::discard_received)
      .def("progress", &DeviceSlotPool::progress)
      .def("close", &DeviceSlotPool::close)
      .def_property_readonly("live", &DeviceSlotPool::live)
      .def_property_readonly("peak_live", &DeviceSlotPool::peak_live)
      .def_property_readonly("payload_bytes", &DeviceSlotPool::payload_bytes);
  m.def("record_plane", &DeviceSlotPool::record_plane);
  m.def("publication_elapsed_ms", &DeviceSlotPool::publication_elapsed_ms);
  m.def("nccl_library_info", &nccl_library_info);
  m.def("current_numa_node", [] {
    unsigned cpu=0,node=0;
    if (::syscall(SYS_getcpu,&cpu,&node,nullptr)!=0 || node>INT_MAX) return -1;
    const auto path="/sys/devices/system/node/node"+std::to_string(node)+"/cpulist";
    return ::access(path.c_str(),R_OK)==0 ? static_cast<int>(node) : -1;
  });
  py::class_<HostRegion,std::shared_ptr<HostRegion>>(m,"HostRegion")
      .def_static("create_dma", [](std::uint32_t capacity,std::uint32_t slots,int node,std::uint64_t segment,
                                   std::uint64_t reserve) {
        py::gil_scoped_release release;
        return DmaEdge::create_region({capacity,slots},node,segment,reserve);
      },py::arg("capacity"),py::arg("slots"),py::arg("node")=-1,py::arg("segment_bytes")=0,
         py::arg("minimum_available_bytes")=0)
      .def_static("attach", [](const std::array<std::int64_t,7>& h) {
        TORCH_CHECK(h[0]>0 && h[1]>=0 && h[2]>=0 && h[3]>=0 && h[4]>0 && h[4]<=INT_MAX &&
                    h[5]>=0 && h[5]<INT_MAX && h[6]>=-1 && h[6]<=INT_MAX,"invalid shared-region handle");
        HostRegionHandle handle{std::uint64_t(h[0]),std::uint64_t(h[1]),std::uint64_t(h[2]),
          std::uint64_t(h[3]),int(h[4]),int(h[5]),int(h[6])};
        py::gil_scoped_release release; return HostRegion::attach(handle);
      })
      .def_property_readonly("handle", [](HostRegion& r) {
        const auto& h=r.handle();
        TORCH_CHECK(h.bytes<=INT64_MAX && h.device<=INT64_MAX && h.inode<=INT64_MAX &&
                    h.process_birth<=INT64_MAX,"shared-region handle exceeds signed wire range");
        return std::array<std::int64_t,7>{std::int64_t(h.bytes),std::int64_t(h.device),
          std::int64_t(h.inode),std::int64_t(h.process_birth),h.owner_pid,h.owner_fd,h.numa_node};
      });
  py::class_<DmaProgress>(m,"DmaProgress")
      .def(py::init<c10::DeviceIndex>())
      .def("start",&DmaProgress::start,py::call_guard<py::gil_scoped_release>())
      .def("stop",&DmaProgress::stop,py::call_guard<py::gil_scoped_release>())
      .def("stats",[](DmaProgress& p) {
        const auto s=p.stats();
        return py::dict("segments"_a=s.segments,"idle_waits"_a=s.idle_waits,
          "peak_jobs"_a=s.peak_jobs,"cpu_seconds"_a=s.cpu_seconds,"tid"_a=s.tid);
      });
  py::class_<DmaEdge>(m,"DmaEdge")
      .def(py::init([](std::shared_ptr<DeviceSlotPool> pool,std::shared_ptr<HostRegion> region,
                      std::uint64_t stream,c10::DeviceIndex device,DmaProgress& progress,
                      std::uint64_t segment) {
        auto queue=c10::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream),device);
        py::gil_scoped_release release;
        return std::make_unique<DmaEdge>(std::move(pool),std::move(region),queue,segment,&progress);
      }),py::keep_alive<1,6>())
      .def("export_event",[](DmaEdge& e) { return py::bytes(e.export_event()); })
      .def("connect",[](DmaEdge& e,py::bytes event) {
        const std::string handle=event; py::gil_scoped_release release; e.connect(handle);
      })
      .def("reserve",[](DmaEdge& e,py::bytes input,std::uint32_t timeout) {
        const std::string data=input; TORCH_CHECK(data.size()==96,"expected fixed96-byte boundary");
        WireBoundary wire; std::copy(data.begin(),data.end(),wire.begin());
        py::gil_scoped_release release; return e.reserve(wire,timeout);
      })
      .def("submit",&DmaEdge::submit,"ticket"_a,"independent_ready"_a=false,
           py::call_guard<py::gil_scoped_release>())
      .def("wait_recorded",&DmaEdge::wait_recorded,py::call_guard<py::gil_scoped_release>())
      .def("progress",&DmaEdge::progress,py::call_guard<py::gil_scoped_release>())
      .def("disconnect",&DmaEdge::disconnect,py::call_guard<py::gil_scoped_release>())
      .def("close",&DmaEdge::close,py::call_guard<py::gil_scoped_release>())
      .def("abort",&DmaEdge::abort,py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("host_bytes",&DmaEdge::host_bytes)
      .def_property_readonly("numa_receipt",&DmaEdge::numa_receipt);
  m.def("nccl_unique_id", [] { return py::bytes(nccl_unique_id()); });
  py::class_<NcclEdge>(m, "NcclEdge")
      .def(py::init([](std::shared_ptr<DeviceSlotPool> pool, py::bytes input) {
        const std::string uid = input;
        py::gil_scoped_release release;
        return std::make_unique<NcclEdge>(std::move(pool), uid);
      }))
      .def("reserve", [](NcclEdge& edge, py::bytes input, std::uint32_t timeout_ms) {
        const std::string data = input;
        TORCH_CHECK(data.size() == WireBoundary{}.size(), "expected fixed96-byte boundary");
        WireBoundary wire;
        std::copy(data.begin(), data.end(), wire.begin());
        py::gil_scoped_release release;
        return edge.reserve(wire, timeout_ms);
      })
      .def("submit", &NcclEdge::submit, py::call_guard<py::gil_scoped_release>())
      .def("progress", &NcclEdge::progress, py::call_guard<py::gil_scoped_release>())
      .def("close", &NcclEdge::close, py::call_guard<py::gil_scoped_release>())
      .def("abort", &NcclEdge::abort, py::call_guard<py::gil_scoped_release>());
}
