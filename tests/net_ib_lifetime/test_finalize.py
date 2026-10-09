"""Compile the actual finalizer with mocked verbs; run lifetime gates under UBSan."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
source = (root / 'src/transport/net_ib/init.cc').read_text()
start = source.index('ncclResult_t ncclIbFinalizeDevices(void) {')
end = source.index('\n}\n', start) + 3
actual = source[start:end]
prefix = r'''
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
using ncclResult_t = int;
constexpr int ncclSuccess = 0, ncclInvalidUsage = 2;
#define WARN(...) ((void)0)
#define INFO(...) ((void)0)
#define NCCLCHECK(call) do { auto result = (call); if (result) return result; } while (0)
struct ibv_context { int id; };
struct ncclIbMrCache { void* slots = nullptr; int population = 0; };
struct ncclIbDev {
  int pdRefs = 0; ncclIbMrCache mrCache;
  const char* devName = "mock"; ibv_context* context = nullptr;
  char* pciPath = nullptr; void* pd = nullptr;
};
static ncclIbDev ncclIbDevs[8];
static int netRefCount, ncclNIbDevs, ncclNMergedIbDevs, enabled, closed[8];
static std::mutex ncclIbLifetimeMutex;
static std::vector<std::thread> ncclIbOwnedReaders;
static std::atomic<bool> ncclIbAsyncStop{false};
static std::atomic<bool> readerExited{false};
int ncclParamIbReleaseOnFinalize() { return enabled; }
ncclResult_t wrap_ibv_close_device(ibv_context* c) {
  assert(readerExited.load());
  assert(++closed[c->id] == 1); return ncclSuccess;
}
void reset() {
  assert(ncclIbOwnedReaders.empty());
  for (auto& d : ncclIbDevs) d = {};
  for (auto& c : closed) c = 0;
  netRefCount = 1; ncclNIbDevs = 0; ncclNMergedIbDevs = 0; enabled = 1;
  ncclIbAsyncStop.store(false); readerExited.store(true);
}
'''
cases = r'''
int main() {
  ibv_context a{1}, b{2};
  reset(); enabled = 0; ncclNIbDevs = 1; ncclIbDevs[0].context = &a;
  assert(ncclIbFinalizeDevices() == ncclSuccess);
  assert(closed[1] == 0 && ncclNIbDevs == 1); // Legacy retention stays default.
  reset(); netRefCount = 2; ncclNIbDevs = 1; ncclIbDevs[0].context = &a;
  assert(ncclIbFinalizeDevices() == ncclSuccess);
  assert(netRefCount == 1 && closed[1] == 0); // Other communicators keep contexts.
  reset(); ncclNIbDevs = 1; ncclIbDevs[0].context = &a; ncclIbDevs[0].pdRefs = 1;
  assert(ncclIbFinalizeDevices() == ncclInvalidUsage);
  assert(closed[1] == 0 && !ncclIbAsyncStop.load()); // Refuse active DMA owners.
  reset(); ncclNIbDevs = 1; ncclIbDevs[0].context = &a; ncclIbDevs[0].mrCache.population = 1;
  assert(ncclIbFinalizeDevices() == ncclInvalidUsage);
  assert(closed[1] == 0 && !ncclIbAsyncStop.load()); // Refuse live registrations.
  reset(); netRefCount = 0;
  assert(ncclIbFinalizeDevices() == ncclInvalidUsage); // No counter underflow.
  reset(); ncclNIbDevs = 3; ncclNMergedIbDevs = 3;
  ncclIbDevs[0].context = &a; ncclIbDevs[1].context = &a; ncclIbDevs[2].context = &b;
  char* sharedPath = (char*)calloc(64, 1);
  ncclIbDevs[0].pciPath = sharedPath; ncclIbDevs[1].pciPath = sharedPath;
  ncclIbDevs[2].pciPath = (char*)calloc(64, 1);
  readerExited.store(false);
  ncclIbOwnedReaders.emplace_back([] {
    while (!ncclIbAsyncStop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    readerExited.store(true);
  });
  assert(ncclIbFinalizeDevices() == ncclSuccess);
  assert(closed[1] == 1 && closed[2] == 1); // Shared ports close only once.
  assert(ncclIbOwnedReaders.empty() && readerExited.load()); // Join before close.
  assert(ncclNIbDevs == -1 && ncclNMergedIbDevs == 0); // Next init re-enumerates.
  for (int d = 0; d < 3; ++d) assert(!ncclIbDevs[d].context && !ncclIbDevs[d].pciPath);
}
'''
with tempfile.TemporaryDirectory() as directory:
    p = Path(directory)
    (p / 'test.cc').write_text(prefix + actual + cases)
    subprocess.run(['g++', '-std=c++17', '-pthread', '-O1', '-g', '-fsanitize=undefined',
                    '-fno-sanitize-recover=all', str(p / 'test.cc'), '-o', str(p / 'test')], check=True)
    subprocess.run([str(p / 'test')], check=True)
print('PASS: 6 actual-finalizer lifecycle cases, shared contexts, reader join, UBSan')
