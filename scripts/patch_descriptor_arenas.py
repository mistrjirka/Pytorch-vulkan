#!/usr/bin/env python3
"""Patch Kompute v0.9.0 to pool descriptor sets independently of VkPipelines.

Run after scripts/patch_kompute.py. Every Algorithm keeps its unique bindings,
and only releases the set when all OpAlgoDispatch holders have been destroyed.
The descriptor-set layout and backing descriptor pools are reused by all
algorithms requiring the same number of storage-buffer bindings.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from patch_kompute import replace_once


def patch_descriptor_arenas(source: Path) -> None:
    header=source/"src/include/kompute/Algorithm.hpp"
    old=header.read_text()
    s=old
    s=replace_once(s,"struct SharedPipelineBundle;\n",
                   "struct SharedPipelineBundle;\nstruct SharedDescriptorArena;\n",
                   "descriptor arena type declaration")
    s=replace_once(s,"    std::shared_ptr<SharedPipelineBundle> mSharedPipelineBundle;\n    std::string mSharedPipelineCacheKey;",
                   "    std::shared_ptr<SharedPipelineBundle> mSharedPipelineBundle;\n    std::shared_ptr<SharedDescriptorArena> mDescriptorArena;\n    std::string mSharedPipelineCacheKey;",
                   "descriptor arena lifetime lease")
    if s!=old:header.write_text(s)

    file=source/"src/Algorithm.cpp"
    old=file.read_text()
    s=old
    s=replace_once(s, '#include "kompute/Algorithm.hpp"',
                   '#include "kompute/Algorithm.hpp"\n#include <cstdlib>\n#include <cstring>\n#include <algorithm>',
                   "descriptor arena reuse headers")
    anchor="// The bundle owns immutable pipeline handles; Algorithm instances keep the"
    snippet=r"""// Reuse descriptor-set layouts and per-layout pools independently of shader
// pipelines. Every Algorithm has a private descriptor SET, while the shared
// pool is owned by the arena until its last in-flight user has completed.
struct SharedDescriptorArena {
    static constexpr uint32_t kPoolSize = 128;
    static constexpr size_t kMaxIdleSets = 256; // per binding-count arena
    struct Pool {
        vk::DescriptorPool handle;
        uint32_t live=0;      // assigned to Algorithms
        uint32_t allocated=0; // including reusable, unassigned sets
        std::vector<vk::DescriptorSet> idle;
    };
    std::shared_ptr<vk::Device> device;
    vk::DescriptorSetLayout layout;
    uint32_t bindings;
    std::vector<std::shared_ptr<Pool>> pools;
    size_t idle_count=0;
    std::mutex mutex;
    static bool reuse_enabled() {
        static const bool enabled=[]() {
            const char* flag=std::getenv("TORCH_VULKAN_DESCRIPTOR_SET_REUSE");
            return !(flag && std::strcmp(flag,"0")==0);
        }();
        return enabled;
    }

    SharedDescriptorArena(std::shared_ptr<vk::Device> dev, uint32_t n)
        : device(std::move(dev)), bindings(n) {
        std::vector<vk::DescriptorSetLayoutBinding> layout_bindings;
        for (uint32_t i=0;i<n;++i)
            layout_bindings.emplace_back(i,vk::DescriptorType::eStorageBuffer,
                                         1,vk::ShaderStageFlagBits::eCompute);
        vk::DescriptorSetLayoutCreateInfo create({},n,layout_bindings.data());
        layout=device->createDescriptorSetLayout(create);
    }
    ~SharedDescriptorArena() {
        if (device) {
            for (const auto& p:pools) device->destroy(p->handle);
            if(layout) device->destroy(layout);
        }
    }
    // Each descriptor set is private to its Algorithm. Rebind only a set
    // whose previous Algorithm has been destroyed: OpAlgoDispatch retains
    // that Algorithm until the owning Sequence is GPU-complete.
    std::pair<vk::DescriptorPool,vk::DescriptorSet> acquire() {
        std::lock_guard<std::mutex> guard(mutex);
        if(reuse_enabled()) for (auto& p:pools) {
            if(p->idle.empty()) continue;
            auto set=p->idle.back();
            p->idle.pop_back();
            --idle_count;
            ++p->live;
            return {p->handle,set};
        }
        for (auto& p:pools) {
            if(p->allocated >= kPoolSize) continue;
            vk::DescriptorSetAllocateInfo alloc(p->handle,1,&layout);
            try {
                auto sets=device->allocateDescriptorSets(alloc);
                ++p->allocated;
                ++p->live;
                return {p->handle,sets.front()};
            } catch (const vk::SystemError&) {
                // A fragmented pool may require a new pool. Other pools
                // remain valid and do not drop live in-flight descriptors.
            }
        }
        vk::DescriptorPoolSize size(vk::DescriptorType::eStorageBuffer,
                                    std::max(1u,bindings)*kPoolSize);
        vk::DescriptorPoolCreateInfo create(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,kPoolSize,1,&size);
        auto p=std::make_shared<Pool>();
        p->handle=device->createDescriptorPool(create);
        pools.push_back(p);
        vk::DescriptorSetAllocateInfo alloc(p->handle,1,&layout);
        auto sets=device->allocateDescriptorSets(alloc);
        ++p->allocated;
        ++p->live;
        return {p->handle,sets.front()};
    }
    void release(vk::DescriptorPool pool, vk::DescriptorSet set) {
        std::lock_guard<std::mutex> guard(mutex);
        for (auto& p:pools) {
            if (p->handle!=pool) continue;
            if (!p->live) throw std::runtime_error("Descriptor arena double release");
            --p->live;
            if(reuse_enabled() && idle_count<kMaxIdleSets) {
                p->idle.push_back(set);
                ++idle_count;
            } else {
                device->freeDescriptorSets(pool,std::vector<vk::DescriptorSet>{set});
                --p->allocated;
            }
            if(!p->allocated) {
                // A non-live, empty pool no longer retains descriptor memory.
                // In-flight Algorithms always count as live, so its handle
                // cannot be destroyed while recorded GPU work may use it.
                auto handle=p->handle;
                pools.erase(std::remove(pools.begin(),pools.end(),p),pools.end());
                device->destroy(handle);
            }
            return;
        }
        throw std::runtime_error("Unknown descriptor pool in Kompute cache");
    }
};

"""
    s=replace_once(s,anchor,snippet+anchor,"descriptor arena structure")
    anchor="std::mutex shared_pipeline_mutex;"
    replacement=("std::mutex shared_pipeline_mutex;\n"
                 "std::unordered_map<std::string,std::shared_ptr<SharedDescriptorArena>> shared_arenas;")
    s=replace_once(s,anchor,replacement,"descriptor cache")
    anchor="Algorithm::PipelineCacheStats Algorithm::sharedPipelineCacheStats() {"
    snippet=r"""std::shared_ptr<SharedDescriptorArena> get_descriptor_arena(
    const std::shared_ptr<vk::Device>& device,size_t count) {
    std::string key;
    auto dev=reinterpret_cast<uintptr_t>(device.get());
    append_blob(key,dev);
    auto bindings=static_cast<uint32_t>(count);
    append_blob(key,bindings);
    std::lock_guard<std::mutex> lock(shared_pipeline_mutex);
    auto found=shared_arenas.find(key);
    if (found!=shared_arenas.end()) return found->second;
    auto arena=std::make_shared<SharedDescriptorArena>(device,bindings);
    shared_arenas.emplace(std::move(key),arena);
    return arena;
}

"""
    s=replace_once(s,anchor,snippet+anchor,"descriptor arena lookup")
    anchor="    shared_pipelines.clear();\n    shared_pipeline_fifo.clear();"
    replacement="    shared_pipelines.clear();\n    shared_pipeline_fifo.clear();\n    shared_arenas.clear();"
    s=replace_once(s,anchor,replacement,"descriptor arena teardown")
    anchor="    if (this->mFreeDescriptorSetLayout && this->mDescriptorSetLayout) {"
    replacement="""    if (mDescriptorArena && mDescriptorSet && mDescriptorPool) {
        mDescriptorArena->release(*mDescriptorPool,*mDescriptorSet);
        mDescriptorSet.reset();
        mDescriptorPool.reset();
        mDescriptorSetLayout.reset();
        mDescriptorArena.reset();
    }

"""+anchor
    s=replace_once(s,anchor,replacement,"descriptor release")

    start=s.find("    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes = {",
                 s.index("Algorithm::createParameters()"))
    end=s.find('    KP_LOG_DEBUG("Kompute Algorithm updating descriptor sets");', start)
    if start < 0 or end < 0:
        if "mDescriptorArena=get_descriptor_arena(mDevice,mTensors.size());" not in s:
            raise RuntimeError("Could not locate Kompute createParameters")
    else:
        replacement="""    // Per-binding layout is structurally compatible with the compiled
    // pipeline layout; the VkDescriptorSet itself is never shared.
    mDescriptorArena=get_descriptor_arena(mDevice,mTensors.size());
    auto allocation=mDescriptorArena->acquire();
    mDescriptorPool=std::make_shared<vk::DescriptorPool>(allocation.first);
    mDescriptorSetLayout=std::shared_ptr<vk::DescriptorSetLayout>(
        mDescriptorArena,&mDescriptorArena->layout);
    mDescriptorSet=std::make_shared<vk::DescriptorSet>(allocation.second);
    mFreeDescriptorPool=false;
    mFreeDescriptorSetLayout=false;
    mFreeDescriptorSet=false;

"""
        s=s[:start]+replacement+s[end:]
    if s!=old:file.write_text(s)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("kompute_source",type=Path)
    args=p.parse_args()
    patch_descriptor_arenas(args.kompute_source.resolve())
    print("Patched Kompute v0.9.0 descriptor arena")
