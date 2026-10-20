#pragma once

// Device-independent chunk scheduler for GPU tensor-list kernels.
// Each group maps consecutive workgroups to their owning tensor, preserving
// all tensor-specific sizes and accommodating different update coefficients.
// The operator supplies a conflict predicate for its read/write hazards.
// No assumptions about the operator's mathematical content are made here.

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace torch_vulkan::ops::multitensor {

struct ChunkGroup {
  size_t first;
  size_t last; // exclusive
  uint32_t workgroups;
};

template<class Conflicts>
std::vector<ChunkGroup> partition(const std::vector<uint32_t>& chunks,
                                  size_t slots,
                                  uint32_t max_workgroups,
                                  Conflicts conflicts) {
  if (!slots || !max_workgroups)
    throw std::invalid_argument("Multitensor: invalid device group limit");
  std::vector<ChunkGroup> result;
  for (size_t first=0;first<chunks.size();) {
    size_t last=first;
    uint32_t total=0;
    while(last<chunks.size() && last-first<slots) {
      if(!chunks[last] || chunks[last]>max_workgroups-total)
        break;
      bool independent=true;
      for(size_t prev=first;prev<last;++prev)
        if(conflicts(prev,last)) {independent=false;break;}
      if(!independent)break;
      total+=chunks[last++];
    }
    if(last==first)
      throw std::invalid_argument("Multitensor: tensor exceeds device workgroup limit");
    result.push_back(ChunkGroup{first,last,total});
    first=last;
  }
  return result;
}

} // namespace torch_vulkan::ops::multitensor
