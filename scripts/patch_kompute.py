#!/usr/bin/env python3
"""Idempotently patch the pinned Kompute v0.9.0 to share compute pipelines.

Kompute Algorithm historically creates a VkPipeline, shader module, cache,
pipeline layout and descriptor set *per bound buffer tuple*. This patch
caches the immutable objects by Vulkan device, SPIR-V, specialization values,
push-constant layout and binding count, while leaving per-algorithm descriptor
sets and pools bound to the current tensors.

Only the downloaded v0.9.0 source tree is modified, never a system library.
The patch is idempotent and refuses incompatible/unexpected source contents.
This is a temporary compatibility shim until a dedicated Vulkan executor
replaces Kompute's tensor-address-bound Algorithm cache.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(s: str, old: str, new: str, description: str) -> str:
    if new in s:
        return s
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{description}: expected exactly one unpatched anchor; found {n}")
    return s.replace(old, new, 1)


def patch_headers(source: Path) -> None:
    file = source / "src/include/kompute/Algorithm.hpp"
    original = file.read_text()
    s = original
    if ("mDescriptorArena" in s and "sharedPipelineCacheStats" in s and
        "mSharedPipelineCacheKey" in s):
        # The follow-on arena patch introduces an extra declaration/member
        # into our original anchors. Both patches have already been applied.
        return
    s = replace_once(s, "namespace kp {\n",
        "namespace kp {\n\nstruct SharedPipelineBundle;\n",
        "Algorithm forward declaration")
    anchor = "    void destroy();\n\n  private:"
    replacement = """    void destroy();

    // Reusable VkPipelines independent of the bound descriptor buffers.
    struct PipelineCacheStats {
        uint64_t hits = 0, misses = 0, evictions = 0, entries = 0;
    };
    static PipelineCacheStats sharedPipelineCacheStats();
    // Must be called before destroying the Kompute VkDevice.
    static void clearSharedPipelineCache();

  private:"""
    s = replace_once(s, anchor, replacement, "public API")
    anchor = "    std::shared_ptr<vk::Pipeline> mPipeline;\n    bool mFreePipeline = false;"
    replacement = """    std::shared_ptr<vk::Pipeline> mPipeline;
    bool mFreePipeline = false;
    // Shared bundle owns immutable pipeline components, while this Algorithm
    // continues to own its own tensor descriptors and descriptor pool.
    std::shared_ptr<SharedPipelineBundle> mSharedPipelineBundle;
    std::string mSharedPipelineCacheKey;"""
    s = replace_once(s, anchor, replacement, "private pipeline bundle")
    # A rebuild invalidates the previous alias fields and releases its lease.
    anchor = """        this->createParameters();
        this->createShaderModule();
        this->createPipeline();"""
    replacement = """        // Release the old shared pipeline before constructing another one.
        this->mSharedPipelineBundle.reset();
        this->mPipeline.reset();
        this->mPipelineLayout.reset();
        this->mPipelineCache.reset();
        this->mShaderModule.reset();
        this->mSharedPipelineCacheKey.clear();
        this->createParameters();
        this->createShaderModule();
        this->createPipeline();"""
    s = replace_once(s, anchor, replacement, "rebuild lease")
    if s != original:
        file.write_text(s)


def patch_source(source: Path) -> None:
    file = source / "src/Algorithm.cpp"
    original = file.read_text()
    s = original
    if ("struct SharedDescriptorArena {" in s and
        "Algorithm::sharedPipelineCacheStats()" in s and
        "mSharedPipelineBundle = bundle" in s):
        return
    s = replace_once(s, '#include <fstream>\n',
        '#include <fstream>\n#include <deque>\n#include <mutex>\n#include <unordered_map>\n#include <cstring>\n#include <cstdint>\n',
        "Algorithm includes")
    anchor = "namespace kp {\n\nAlgorithm::~Algorithm()"
    replacement = """namespace kp {

// The bundle owns immutable pipeline handles; Algorithm instances keep the
// current descriptor bindings separate. Pipeline layouts constructed against
// *identically defined* descriptor-set layouts are compatible in Vulkan.
struct SharedPipelineBundle {
    std::shared_ptr<vk::Device> device;
    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
    vk::PipelineCache cache;
    vk::ShaderModule module;

    ~SharedPipelineBundle() {
        // Eager and deferred users hold a shared_ptr to this bundle until the
        // command's completion. Destroy Vk objects before the logical device.
        if (device) {
            if (pipeline) device->destroy(pipeline);
            if (layout) device->destroy(layout);
            if (module) device->destroy(module);
            if (cache) device->destroy(cache);
        }
    }
};

namespace {
std::mutex shared_pipeline_mutex;
std::unordered_map<std::string, std::shared_ptr<SharedPipelineBundle>> shared_pipelines;
std::deque<std::string> shared_pipeline_fifo;
constexpr size_t max_shared_pipelines = 256;
Algorithm::PipelineCacheStats shared_pipeline_stats;

template <typename T>
void append_blob(std::string& key, const T& value) {
    key.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::string make_shared_pipeline_key(
    const std::shared_ptr<vk::Device>& device,
    const std::vector<uint32_t>& spirv,
    size_t descriptors,
    uint32_t push_size,
    uint32_t push_element_size,
    const void* specialization_data,
    uint32_t specialization_size,
    uint32_t specialization_element_size) {
    std::string key;
    auto device_id = reinterpret_cast<uintptr_t>(device.get());
    append_blob(key, device_id);
    auto binding_count = static_cast<uint64_t>(descriptors);
    append_blob(key, binding_count);
    append_blob(key, push_size);
    append_blob(key, push_element_size);
    append_blob(key, specialization_size);
    append_blob(key, specialization_element_size);
    auto code_size = static_cast<uint64_t>(spirv.size());
    append_blob(key, code_size);
    key.append(reinterpret_cast<const char*>(spirv.data()),
               spirv.size() * sizeof(uint32_t));
    if (specialization_size) {
        key.append(reinterpret_cast<const char*>(specialization_data),
                   specialization_size * specialization_element_size);
    }
    return key;
}
} // namespace

Algorithm::PipelineCacheStats Algorithm::sharedPipelineCacheStats() {
    std::lock_guard<std::mutex> lock(shared_pipeline_mutex);
    auto s = shared_pipeline_stats;
    s.entries = shared_pipelines.size();
    return s;
}

void Algorithm::clearSharedPipelineCache() {
    std::lock_guard<std::mutex> lock(shared_pipeline_mutex);
    shared_pipelines.clear();
    shared_pipeline_fifo.clear();
}

Algorithm::~Algorithm()"""
    s = replace_once(s, anchor, replacement, "shared cache helpers")
    # Once descriptors exist, look up a matching pipeline before allocating
    # a shader module. Do not couple the cache key to any tensor data pointer.
    anchor = """    KP_LOG_DEBUG("Kompute Algorithm createShaderModule started");

    vk::ShaderModuleCreateInfo shaderModuleInfo"""
    replacement = """    KP_LOG_DEBUG("Kompute Algorithm createShaderModule started");

    mSharedPipelineCacheKey = make_shared_pipeline_key(
        mDevice, mSpirv, mTensors.size(),
        mPushConstantsSize, mPushConstantsDataTypeMemorySize,
        mSpecializationConstantsData, mSpecializationConstantsSize,
        mSpecializationConstantsDataTypeMemorySize);
    {
        std::lock_guard<std::mutex> lock(shared_pipeline_mutex);
        auto it = shared_pipelines.find(mSharedPipelineCacheKey);
        if (it != shared_pipelines.end()) {
            mSharedPipelineBundle = it->second;
            auto& b = *mSharedPipelineBundle;
            // Aliasing shared_ptrs hold the same owning bundle alive.
            mShaderModule = std::shared_ptr<vk::ShaderModule>(
                mSharedPipelineBundle, &b.module);
            mPipelineLayout = std::shared_ptr<vk::PipelineLayout>(
                mSharedPipelineBundle, &b.layout);
            mPipelineCache = std::shared_ptr<vk::PipelineCache>(
                mSharedPipelineBundle, &b.cache);
            mPipeline = std::shared_ptr<vk::Pipeline>(
                mSharedPipelineBundle, &b.pipeline);
            mFreeShaderModule = mFreePipelineLayout = false;
            mFreePipelineCache = mFreePipeline = false;
            ++shared_pipeline_stats.hits;
            return;
        }
    }

    vk::ShaderModuleCreateInfo shaderModuleInfo"""
    s = replace_once(s, anchor, replacement, "shader cache lookup")
    anchor = """    KP_LOG_DEBUG("Kompute Algorithm calling create Pipeline");

    vk::PipelineLayoutCreateInfo"""
    replacement = """    KP_LOG_DEBUG("Kompute Algorithm calling create Pipeline");
    if (mSharedPipelineBundle) return;

    vk::PipelineLayoutCreateInfo"""
    s = replace_once(s, anchor, replacement, "pipeline cache hit")
    anchor = """    KP_LOG_DEBUG("Kompute Algorithm Create Pipeline Success");
}"""
    replacement = """    // Transfer VkPipeline ownership to a reusable immutable bundle.
    // Descriptor layouts and pools remain owned by this bound Algorithm.
    auto bundle = std::make_shared<SharedPipelineBundle>();
    bundle->device = mDevice;
    bundle->pipeline = *mPipeline;
    bundle->layout = *mPipelineLayout;
    bundle->cache = *mPipelineCache;
    bundle->module = *mShaderModule;
    mSharedPipelineBundle = bundle;
    mPipeline = std::shared_ptr<vk::Pipeline>(bundle, &bundle->pipeline);
    mPipelineLayout = std::shared_ptr<vk::PipelineLayout>(bundle, &bundle->layout);
    mPipelineCache = std::shared_ptr<vk::PipelineCache>(bundle, &bundle->cache);
    mShaderModule = std::shared_ptr<vk::ShaderModule>(bundle, &bundle->module);
    mFreePipeline = mFreePipelineLayout = false;
    mFreeShaderModule = mFreePipelineCache = false;

    {
        std::lock_guard<std::mutex> lock(shared_pipeline_mutex);
        if (shared_pipelines.size() >= max_shared_pipelines) {
            shared_pipelines.erase(shared_pipeline_fifo.front());
            shared_pipeline_fifo.pop_front();
            ++shared_pipeline_stats.evictions;
        }
        shared_pipelines[mSharedPipelineCacheKey] = bundle;
        shared_pipeline_fifo.push_back(mSharedPipelineCacheKey);
        ++shared_pipeline_stats.misses;
    }

    KP_LOG_DEBUG("Kompute Algorithm Create Pipeline Success");
}"""
    s = replace_once(s, anchor, replacement, "pipeline cache insertion")
    if s != original:
        file.write_text(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("kompute_source", type=Path)
    args = p.parse_args()
    src = args.kompute_source.resolve()
    if not (src / "src/Algorithm.cpp").is_file():
        p.error("Not a Kompute v0.9.0 source tree: missing src/Algorithm.cpp")
    patch_headers(src)
    patch_source(src)
    print("Patched Kompute v0.9.0 with shared pipeline bundle cache")


if __name__ == "__main__":
    main()
