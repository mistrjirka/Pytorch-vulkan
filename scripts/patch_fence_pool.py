#!/usr/bin/env python3
"""Lease completed Vulkan fences across Kompute v0.9.0 sequences.

A Manager owns the pool; an in-flight Sequence exclusively owns its fence.
Timeouts retain that lease. Teardown waits before releasing command resources.
"""
from pathlib import Path
import argparse
from patch_kompute import replace_once

HEADER=r'''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kompute/Core.hpp"
#include <mutex>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace kp {
class FencePool {
    std::shared_ptr<vk::Device> device_;
    std::mutex mutex_;
    std::vector<vk::Fence> free_;
    bool closed_=false,enabled_=true;
public:
    explicit FencePool(std::shared_ptr<vk::Device> device):device_(device) {
        const char* flag=std::getenv("TORCH_VULKAN_DISABLE_FENCE_POOL");
        enabled_=!flag || std::strcmp(flag,"1")!=0;
        free_.reserve(32);
    }
    ~FencePool(){ shutdown(); }
    vk::Fence acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if(closed_)throw std::runtime_error("Kompute fence pool is closed");
        if(!enabled_ || free_.empty())return device_->createFence(vk::FenceCreateInfo());
        auto fence=free_.back();free_.pop_back();
        if(device_->resetFences(1,&fence)!=vk::Result::eSuccess) {
            device_->destroyFence(fence);
            throw std::runtime_error("Kompute fence reset failed");
        }
        return fence;
    }
    // Only fences with no pending queue work may be returned here.
    void release(vk::Fence fence) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(closed_ || !enabled_ || free_.size()>=32)device_->destroyFence(fence);
        else free_.push_back(fence);
    }
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        for(auto fence:free_)device_->destroyFence(fence);
        free_.clear();closed_=true;
    }
};
} // namespace kp
'''

def patch(source):
    def edit(path,changes):
        file=source/path;s=file.read_text()
        for old,new,name in changes:s=replace_once(s,old,new,name)
        file.write_text(s)
    header=source/'src/include/kompute/FencePool.hpp'
    if header.exists() and header.read_text()!=HEADER:
        raise RuntimeError('Unexpected existing FencePool.hpp')
    header.write_text(HEADER)
    edit('src/include/kompute/Sequence.hpp',[
        ('#include "kompute/Core.hpp"','#include "kompute/Core.hpp"\n#include "kompute/FencePool.hpp"','fence pool include'),
        ('uint32_t totalTimestamps = 0);','uint32_t totalTimestamps = 0,\n             std::shared_ptr<FencePool> fencePool = nullptr);','pool constructor parameter'),
        ('    vk::Fence mFence;','    vk::Fence mFence;\n    std::shared_ptr<FencePool> mFencePool;','sequence fence lease')])
    edit('src/include/kompute/Manager.hpp',[
        ('    std::shared_ptr<vk::Device> mDevice = nullptr;',
         '    std::shared_ptr<vk::Device> mDevice = nullptr;\n    std::shared_ptr<FencePool> mFencePool;\n    std::once_flag mFencePoolInit;','manager pool ownership')])
    edit('src/Manager.cpp',[
        ('    std::shared_ptr<Sequence> sq{ new kp::Sequence(',
         '    std::call_once(this->mFencePoolInit,[this](){\n        this->mFencePool=std::make_shared<FencePool>(this->mDevice);\n    });\n    std::shared_ptr<Sequence> sq{ new kp::Sequence(','initialize device pool'),
        ('      totalTimestamps) };','      totalTimestamps,this->mFencePool) };','share pool with sequence'),
        ('    if (this->mFreeDevice) {\n        KP_LOG_INFO("Destroying device");',
         '    // Managed sequences were destroyed above; release idle fences before the device.\n    if(this->mFencePool)this->mFencePool->shutdown();\n    this->mFencePool.reset();\n    if (this->mFreeDevice) {\n        KP_LOG_INFO("Destroying device");','pool teardown order')])
    edit('src/Sequence.cpp',[
        ('                   uint32_t totalTimestamps)','                   uint32_t totalTimestamps,\n                   std::shared_ptr<FencePool> fencePool)','pool constructor definition'),
        ('    this->mQueueIndex = queueIndex;','    this->mQueueIndex = queueIndex;\n    this->mFencePool = std::move(fencePool);','store pool lease'),
        ('    KP_LOG_DEBUG("Kompute Sequence calling clear");\n    this->mOperations.clear();',
         '    KP_LOG_DEBUG("Kompute Sequence calling clear");\n    if(this->mIsRunning)throw std::runtime_error("Cannot clear an in-flight Kompute sequence");\n    this->mOperations.clear();','preserve in-flight resources'),
        ('    this->mIsRunning = true;\n\n    for (size_t i = 0; i < this->mOperations.size(); i++) {',
         '    // A fence is in flight only after a successful submit.\n    for (size_t i = 0; i < this->mOperations.size(); i++) {','submission state'),
        ('    this->mFence = this->mDevice->createFence(vk::FenceCreateInfo());',
         '    this->mFence = this->mFencePool ? this->mFencePool->acquire() :\n        this->mDevice->createFence(vk::FenceCreateInfo());','acquire completed fence'),
        ('    this->mComputeQueue->submit(1, &submitInfo, this->mFence);',
         '''    const auto submitResult=this->mComputeQueue->submit(1, &submitInfo, this->mFence);
    if(submitResult!=vk::Result::eSuccess) {
        this->mDevice->destroyFence(this->mFence);
        this->mFence=nullptr;
        throw std::runtime_error("Kompute queue submission failed");
    }
    this->mIsRunning=true;''','track successful submission'),
        ('''    this->mDevice->destroy(
      this->mFence, (vk::Optional<const vk::AllocationCallbacks>)nullptr);

    this->mIsRunning = false;

    if (result == vk::Result::eTimeout) {
        KP_LOG_WARN("Kompute Sequence evalAwait reached timeout of {}",
                    waitFor);
        return shared_from_this();
    }''',
         '''    if (result == vk::Result::eTimeout) {
        // The queue still owns the fence and command resources. Keep the lease.
        return shared_from_this();
    }
    if(result!=vk::Result::eSuccess)
        throw std::runtime_error("Kompute fence wait failed");
    this->mIsRunning=false;
    if(this->mFencePool)this->mFencePool->release(this->mFence);
    else this->mDevice->destroyFence(this->mFence);
    this->mFence=nullptr;''','release only after completion'),
        ('    if (this->mFreeCommandBuffer) {',
         '''    // Destruction may follow evalAsync or a timeout. Never recycle a fence
    // or free buffers while the queue can still access them.
    if(this->mIsRunning) {
        const auto result=this->mDevice->waitForFences(1,&this->mFence,VK_TRUE,UINT64_MAX);
        if(result!=vk::Result::eSuccess)
            throw std::runtime_error("Kompute sequence teardown wait failed");
        this->mIsRunning=false;
    }
    if(this->mFence) {
        if(this->mFencePool)this->mFencePool->release(this->mFence);
        else this->mDevice->destroyFence(this->mFence);
        this->mFence=nullptr;
    }
    this->mFencePool.reset();
    if (this->mFreeCommandBuffer) {''','wait before teardown')])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);args=p.parse_args()
    patch(args.source);print('Kompute completed-fence pool patch verified')
