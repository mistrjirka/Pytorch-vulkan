#!/usr/bin/env python3
"""Batch adjacent primary-buffer barriers in pinned Kompute v0.9.0.

Preserves every buffer/range, access mask, stage mask and queue-family setting.
No barriers move across a dispatch/copy, and staging-buffer barriers stay intact.
"""
import argparse
from pathlib import Path
from patch_kompute import replace_once

HEADER = r'''// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kompute/Tensor.hpp"
#include <vector>

namespace kp {
// All entries have identical stage scopes and no intervening commands. Keep
// per-buffer access/range dependencies; only batch their Vulkan API recording.
inline void recordPrimaryBufferBarriers(
    const vk::CommandBuffer& commandBuffer,
    const std::vector<std::shared_ptr<Tensor>>& tensors,
    vk::AccessFlagBits srcAccessMask, vk::AccessFlagBits dstAccessMask,
    vk::PipelineStageFlagBits srcStageMask, vk::PipelineStageFlagBits dstStageMask)
{
    if (tensors.empty()) return; // Preserve the old empty loop's no-op behavior.
    std::vector<vk::BufferMemoryBarrier> barriers;
    barriers.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        const auto info = tensor->constructDescriptorBufferInfo();
        vk::BufferMemoryBarrier barrier;
        barrier.buffer = info.buffer;
        barrier.offset = info.offset;
        barrier.size = info.range;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers.push_back(barrier);
    }
    commandBuffer.pipelineBarrier(srcStageMask, dstStageMask,
                                  vk::DependencyFlags(), nullptr, barriers, nullptr);
}
} // namespace kp
'''


def patch(source):
    edits = {
        'src/OpAlgoDispatch.cpp': [
            ('#include "kompute/operations/OpAlgoDispatch.hpp"',
             '#include "kompute/operations/OpAlgoDispatch.hpp"\n#include "kompute/BarrierBatch.hpp"',
             'dispatch barrier batch include'),
            ('''    for (const std::shared_ptr<Tensor>& tensor :
         this->mAlgorithm->getTensors()) {
        tensor->recordPrimaryBufferMemoryBarrier(
          commandBuffer,
          vk::AccessFlagBits::eTransferWrite,
          vk::AccessFlagBits::eShaderRead,
          vk::PipelineStageFlagBits::eTransfer,
          vk::PipelineStageFlagBits::eComputeShader);
    }''', '''    recordPrimaryBufferBarriers(
      commandBuffer, this->mAlgorithm->getTensors(),
      vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader);''',
             'batch dispatch transfer dependencies')],
        'src/OpMemoryBarrier.cpp': [
            ('#include "kompute/operations/OpMemoryBarrier.hpp"',
             '#include "kompute/operations/OpMemoryBarrier.hpp"\n#include "kompute/BarrierBatch.hpp"',
             'memory barrier batch include'),
            ('''        for (const std::shared_ptr<Tensor>& tensor : this->mTensors) {
            tensor->recordPrimaryBufferMemoryBarrier(commandBuffer,
                                                     this->mSrcAccessMask,
                                                     this->mDstAccessMask,
                                                     this->mSrcStageMask,
                                                     this->mDstStageMask);
        }''', '''        recordPrimaryBufferBarriers(commandBuffer, this->mTensors,
                                    this->mSrcAccessMask, this->mDstAccessMask,
                                    this->mSrcStageMask, this->mDstStageMask);''',
             'batch explicit primary dependencies')],
    }
    header = source/'src/include/kompute/BarrierBatch.hpp'
    if header.exists() and header.read_text() != HEADER:
        raise RuntimeError('Unexpected existing BarrierBatch.hpp')
    # Validate all edits before mutating any file; reject source drift.
    prepared = {}
    for relative, changes in edits.items():
        path = source/relative
        text = path.read_text()
        for old,new,label in changes:
            text = replace_once(text, old, new, label)
        prepared[path] = text
    for path,text in {header:HEADER, **prepared}.items():
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
    print('Kompute primary buffer barrier batching applied (idempotent)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    patch(parser.parse_args().source)
