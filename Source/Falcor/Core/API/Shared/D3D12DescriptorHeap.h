/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "D3D12Handles.h"
#include "Core/Assert.h"
#include <d3d12.h>
#include <memory>
#include <queue>
#include <set>
#include <vector>

namespace Falcor
{
    class D3D12DescriptorHeap : public std::enable_shared_from_this<D3D12DescriptorHeap>
    {
    public:
        using SharedPtr = std::shared_ptr<D3D12DescriptorHeap>;
        using SharedConstPtr = std::shared_ptr<const D3D12DescriptorHeap>;
        using ApiHandle = D3D12DescriptorHeapHandle;
        using CpuHandle = D3D12DescriptorCpuHandle;
        using GpuHandle = D3D12DescriptorGpuHandle;

        ~D3D12DescriptorHeap();
        static const uint32_t kDescPerChunk = 64;

        /** Create a new descriptor heap.
            \param[in] type Descriptor heap type.
            \param[in] descCount Descriptor count.
            \param[in] shaderVisible True if the descriptor heap should be shader visible.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t descCount, bool shaderVisible = true);

        CpuHandle getBaseCpuHandle() const { return mCpuHeapStart; }
        GpuHandle getBaseGpuHandle() const;
        bool getShaderVisible() { return mShaderVisible; }

    private:
        struct Chunk;

    public:
        class Allocation
        {
        public:
            using SharedPtr = std::shared_ptr<Allocation>;
            ~Allocation();

            uint32_t getHeapEntryIndex(uint32_t index) const { FALCOR_ASSERT(index < mDescCount); return index + mBaseIndex; }
            CpuHandle getCpuHandle(uint32_t index) const { return mpHeap->getCpuHandle(getHeapEntryIndex(index)); } // Index is relative to the allocation
            GpuHandle getGpuHandle(uint32_t index) const { return mpHeap->getGpuHandle(getHeapEntryIndex(index)); } // Index is relative to the allocation
            D3D12DescriptorHeap* getHeap() { return mpHeap.get(); }
        private:
            friend D3D12DescriptorHeap;
            static SharedPtr create(D3D12DescriptorHeap::SharedPtr pHeap, uint32_t baseIndex, uint32_t descCount, std::shared_ptr<Chunk> pChunk);
            Allocation(D3D12DescriptorHeap::SharedPtr pHeap, uint32_t baseIndex, uint32_t descCount, std::shared_ptr<Chunk> pChunk);
            D3D12DescriptorHeap::SharedPtr mpHeap;
            uint32_t mBaseIndex;
            uint32_t mDescCount;
            std::shared_ptr<Chunk> mpChunk;
        };

        Allocation::SharedPtr allocateDescriptors(uint32_t count);
        const ApiHandle& getApiHandle() const { return mApiHandle; }
        D3D12_DESCRIPTOR_HEAP_TYPE getType() const { return mType; }

        uint32_t getReservedChunkCount() const { return mMaxChunkCount; }
        uint32_t getDescriptorSize() const { return mDescriptorSize; }

    private:
        friend Allocation;
        D3D12DescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t chunkCount, bool shaderVisible);

        CpuHandle getCpuHandle(uint32_t index) const;
        GpuHandle getGpuHandle(uint32_t index) const; // Only valid if heap is shader visible

        CpuHandle mCpuHeapStart = {};
        GpuHandle mGpuHeapStart = {}; // Only valid if heap is shader visible
        uint32_t mDescriptorSize;
        const uint32_t mMaxChunkCount = 0;
        uint32_t mAllocatedChunks = 0;
        ApiHandle mApiHandle;
        D3D12_DESCRIPTOR_HEAP_TYPE mType;
        bool mShaderVisible;

        struct Chunk
        {
        public:
            using SharedPtr = std::shared_ptr<Chunk>;
            Chunk(uint32_t index, uint32_t count) : chunkIndex(index), chunkCount(count) {}

            void reset() { allocCount = 0; currentDesc = 0; }
            uint32_t getCurrentAbsoluteIndex() const { return chunkIndex * kDescPerChunk + currentDesc; }
            uint32_t getRemainingDescs() const { return chunkCount * kDescPerChunk - currentDesc; }

            uint32_t chunkIndex = 0;
            uint32_t chunkCount = 1; // For outstanding requests we can allocate more then a single chunk. This is the number of chunks we actually allocated
            uint32_t allocCount = 0;
            uint32_t currentDesc = 0;
        };

        // Helper to compare Chunk::SharedPtr types
        struct ChunkComparator
        {
            bool operator()(const Chunk::SharedPtr& lhs, const Chunk::SharedPtr& rhs) const
            {
                return lhs->chunkCount < rhs->chunkCount;
            }

            bool operator()(const Chunk::SharedPtr& lhs, uint32_t rhs) const
            {
                return lhs->chunkCount < rhs;
            };
        };

        bool setupCurrentChunk(uint32_t descCount);
        void releaseChunk(Chunk::SharedPtr pChunk);

        Chunk::SharedPtr mpCurrentChunk;
        std::vector<Chunk::SharedPtr> mFreeChunks; // Free list for standard sized chunks (1 chunk * kDescPerChunk)
        std::multiset<Chunk::SharedPtr, ChunkComparator> mFreeLargeChunks; // Free list for large chunks with the capacity of multiple chunks (>1 chunk * kDescPerChunk)
    };
}
