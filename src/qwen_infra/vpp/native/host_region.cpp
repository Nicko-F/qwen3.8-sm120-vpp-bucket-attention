// SPDX-License-Identifier: Apache-2.0
#include "host_region.h"

#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <linux/memfd.h>
#include <limits>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace vpp {
namespace {
constexpr int kSeals = F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
[[noreturn]] void os_error(const char* action) {
  throw std::system_error(errno, std::generic_category(), action);
}
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
std::uint64_t page_size() {
  const auto page = ::sysconf(_SC_PAGESIZE);
  require(page > 0, "cannot determine host page size");
  return static_cast<std::uint64_t>(page);
}
void validate(std::uint64_t bytes, int node) {
  require(bytes > 0 && bytes <= std::numeric_limits<std::size_t>::max() &&
          bytes <= static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) &&
          bytes % page_size() == 0, "shared region must be page aligned and addressable");
  require(node >= -1, "NUMA node must be nonnegative or -1 for inherited placement");
  if (node >= 0) {
    std::ifstream cpus("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    require(cpus.good(), "requested NUMA node does not exist");
  }
}
std::uint64_t birth(int pid) {
  std::ifstream input("/proc/" + std::to_string(pid) + "/stat");
  std::string line, field;
  require(static_cast<bool>(std::getline(input, line)), "owner process is missing");
  const auto end = line.rfind(')');
  require(end != std::string::npos, "invalid owner process record");
  std::istringstream fields(line.substr(end + 1));
  // Field3 starts after comm; starttime is field22.
  for (int i = 0; i < 20; ++i)
    require(static_cast<bool>(fields >> field), "incomplete owner process record");
  return std::stoull(field);
}
void headroom(std::uint64_t bytes, std::uint64_t reserve) {
  require(reserve <= std::numeric_limits<std::uint64_t>::max() - bytes,
          "allocation and optional memory reserve overflow");
  std::ifstream input("/proc/meminfo");
  std::string line, key, unit;
  std::uint64_t kib = 0;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    if ((fields >> key >> kib >> unit) && key == "MemAvailable:") {
      require(unit == "kB", "unexpected MemAvailable units");
      const auto needed = reserve + bytes;
      require(kib >= needed / 1024 + (needed % 1024 != 0),
              "shared region exceeds available memory and its configured reserve");
      return;
    }
  }
  throw std::runtime_error("MemAvailable missing; no allocation permitted");
}
class LocalAffinity {
 public:
  explicit LocalAffinity(int node) {
    if (node < 0) return;
    if (::sched_getaffinity(0, sizeof(saved_), &saved_)) os_error("getaffinity");
    cpu_set_t selected;
    CPU_ZERO(&selected);
    std::ifstream input("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
    std::string list, group;
    require(static_cast<bool>(std::getline(input, list)), "missing NUMA CPU list");
    std::istringstream groups(list);
    while (std::getline(groups, group, ',')) {
      const auto dash = group.find('-');
      const int first = std::stoi(group.substr(0, dash));
      const int last = dash == std::string::npos ? first : std::stoi(group.substr(dash + 1));
      require(first >= 0 && last >= first && last < CPU_SETSIZE, "invalid NUMA CPU range");
      for (int cpu = first; cpu <= last; ++cpu)
        if (CPU_ISSET(cpu, &saved_)) CPU_SET(cpu, &selected);
    }
    require(CPU_COUNT(&selected) > 0, "no allowed CPU on selected NUMA node");
    if (::sched_setaffinity(0, sizeof(selected), &selected)) os_error("setaffinity");
    changed_ = true;
  }
  void restore() {
    if (changed_ && ::sched_setaffinity(0, sizeof(saved_), &saved_)) os_error("restore affinity");
    changed_ = false;
  }
  ~LocalAffinity() noexcept {
    if (changed_) (void)::sched_setaffinity(0, sizeof(saved_), &saved_);
  }
 private:
  cpu_set_t saved_{};
  bool changed_ = false;
};
}  // namespace

void HostRegion::map() {
  void* value = ::mmap(nullptr, handle_.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (value == MAP_FAILED) os_error("mmap shared region");
  base_ = value;
}

std::shared_ptr<HostRegion> HostRegion::create(std::uint64_t bytes, int node,
                                             std::uint64_t minimum_available_bytes) {
  validate(bytes, node);
  headroom(bytes, minimum_available_bytes);
  auto region = std::shared_ptr<HostRegion>(new HostRegion);
  region->fd_ = static_cast<int>(::syscall(SYS_memfd_create, "qwen-vpp-activation",
                                          MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (region->fd_ < 0) os_error("memfd_create");
  if (::ftruncate(region->fd_, static_cast<off_t>(bytes))) os_error("ftruncate shared region");
  if (::fcntl(region->fd_, F_ADD_SEALS, kSeals)) os_error("seal shared region size");
  struct stat identity{};
  if (::fstat(region->fd_, &identity)) os_error("fstat shared region");
  region->handle_ = {bytes, static_cast<std::uint64_t>(identity.st_dev),
                    static_cast<std::uint64_t>(identity.st_ino), birth(::getpid()),
                    ::getpid(), region->fd_, node};
  region->map();
  LocalAffinity affinity(node);
  // One byte per fresh zero-filled page; no duplicate payload buffer/memcpy.
  auto* data = static_cast<volatile char*>(region->base_);
  const auto page = page_size();
  for (std::uint64_t offset = 0; offset < bytes; offset += page) data[offset] = 0;
  if (::mlock(region->base_, bytes)) os_error("mlock shared region");
  region->locked_ = true;
  affinity.restore();
  return region;
}

std::shared_ptr<HostRegion> HostRegion::attach(const HostRegionHandle& handle) {
  validate(handle.bytes, handle.numa_node);
  require(handle.owner_pid > 0 && handle.owner_fd >= 0 && handle.owner_fd < (1 << 20),
          "invalid shared-region owner descriptor");
  require(birth(handle.owner_pid) == handle.process_birth, "shared-region owner PID was reused");
  auto region = std::shared_ptr<HostRegion>(new HostRegion);
  region->handle_ = handle;
  const auto path = "/proc/" + std::to_string(handle.owner_pid) + "/fd/" +
                    std::to_string(handle.owner_fd);
  region->fd_ = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (region->fd_ < 0) os_error("open shared-region owner fd");
  struct stat identity{};
  if (::fstat(region->fd_, &identity)) os_error("fstat peer region");
  require(S_ISREG(identity.st_mode) && identity.st_uid == ::geteuid() &&
          static_cast<std::uint64_t>(identity.st_size) == handle.bytes &&
          static_cast<std::uint64_t>(identity.st_dev) == handle.device &&
          static_cast<std::uint64_t>(identity.st_ino) == handle.inode,
          "peer region identity/size does not match the owner");
  const int seals = ::fcntl(region->fd_, F_GET_SEALS);
  if (seals < 0) os_error("read region seals");
  require((seals & kSeals) == kSeals, "peer region has no immutable size");
  region->map();
  return region;
}

void* HostRegion::range(std::uint64_t offset, std::uint64_t bytes) const {
  require(base_ && offset <= handle_.bytes && bytes <= handle_.bytes - offset,
          "host span exceeds the admitted region");
  return static_cast<char*>(base_) + offset;
}
std::string HostRegion::numa_receipt() const {
  std::ostringstream prefix;
  prefix << std::hex << reinterpret_cast<std::uintptr_t>(base_);
  std::ifstream input("/proc/self/numa_maps");
  std::string line;
  while (std::getline(input, line))
    if (line.compare(0, prefix.str().size() + 1, prefix.str() + " ") == 0) return line;
  throw std::runtime_error("shared-region NUMA mapping was not observed");
}
void HostRegion::verify_numa() const {
  const auto line = numa_receipt();
  require(line.find("file=/memfd:qwen-vpp-activation") != std::string::npos,
          "NUMA receipt is not this activation memfd");
  std::istringstream fields(line);
  std::string field;
  std::uint64_t local = 0, foreign = 0;
  while (fields >> field) {
    if (field.size() < 4 || field[0] != 'N' || field.find('=') == std::string::npos) continue;
    const auto equal = field.find('=');
    const int node = std::stoi(field.substr(1, equal - 1));
    (handle_.numa_node < 0 || node == handle_.numa_node ? local : foreign) +=
        std::stoull(field.substr(equal + 1));
  }
  if (foreign != 0 || local != handle_.bytes / page_size())
    throw std::runtime_error("shared pages are not entirely on node" +
                             std::to_string(handle_.numa_node) + ": " + line);
}
HostRegion::~HostRegion() noexcept {
  if (base_) {
    if (locked_) (void)::munlock(base_, handle_.bytes);
    (void)::munmap(base_, handle_.bytes);
  }
  if (fd_ >= 0) (void)::close(fd_);
}

}  // namespace vpp
