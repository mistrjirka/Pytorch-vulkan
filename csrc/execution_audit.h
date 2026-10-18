#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <string>
namespace torch_vulkan::audit {
inline std::atomic<uint64_t> fallback_calls{0},host_calls{0},host_bytes{0},host_max{0};
inline void host_read(uint64_t bytes){host_calls.fetch_add(1);host_bytes.fetch_add(bytes);auto old=host_max.load();while(old<bytes&&!host_max.compare_exchange_weak(old,bytes)){} }
inline std::map<std::string,uint64_t> stats(){return {{"cpu_fallback_calls",fallback_calls.load()},{"host_read_calls",host_calls.load()},{"host_read_bytes",host_bytes.load()},{"host_read_max_bytes",host_max.load()}};}
inline void reset(){fallback_calls=0;host_calls=0;host_bytes=0;host_max=0;}
}
