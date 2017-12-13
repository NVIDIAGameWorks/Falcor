/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include <queue>

namespace Falcor
{
    class D3D12DescriptorHeap : public std::enable_shared_from_this<D3D12DescriptorHeap>
    {
    public:
        using SharedPtr = std::shared_ptr<D3D12DescriptorHeap>;
        using SharedConstPtr = std::shared_ptr<const D3D12DescriptorHeap>;
        using ApiHandle = DescriptorHeapHandle;
        using CpuHandle = HeapCpuHandle;
        using GpuHandle = HeapGpuHandle;

        ~D3D12DescriptorHeap();
        static const uint32_t kDescPerChunk = 64;

        static SharedPtr create(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t descCount, bool shaderVisible = true);
        GpuHandle getBaseGpuHandle() const { return mGpuHeapStart; }
        CpuHandle getBaseCpuHandle() const { return mCpuHeapStart; }
    private:
        struct Chunk;

    public:
        class Allocation
        {
        public:
            using SharedPtr = std::shared_ptr<Allocation>;
            ~Allocation();

            CpuHandle getCpuHandle(uint32_t index) const { assert(index < mDescCount); return mpHeap->getCpuHandle(index + mBaseIndex); }; // Index is relative to the allocation
            GpuHandle getGpuHandle(uint32_t index) const { assert(index < mDescCount); return mpHeap->getGpuHandle(index + mBaseIndex); }; // Index is relative to the allocation
            
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
        ApiHandle getApiHandle() const { return mApiHandle; }
        D3D12_DESCRIPTOR_HEAP_TYPE getType() const { return mType; }

        uint32_t getReservedChunkCount() const { return mChunkCount; }
        uint32_t getDescriptorSize() const { return mDescriptorSize; }
    private:
        friend Allocation;
        D3D12DescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t chunkCount);
        

        CpuHandle getCpuHandle(uint32_t index) const;
        GpuHandle getGpuHandle(uint32_t index) const;

        CpuHandle mCpuHeapStart = {};
        GpuHandle mGpuHeapStart = {};
        uint32_t mDescriptorSize;
        uint32_t mChunkCount = 0;
        uint32_t mAllocatedChunks = 0;
        ApiHandle mApiHandle;
        D3D12_DESCRIPTOR_HEAP_TYPE mType;

        struct Chunk
        {
        public:
            using SharedPtr = std::shared_ptr<Chunk>;
            Chunk(uint32_t index, uint32_t count) : chunkIndex(index), chunkCount(count) {}

            void reset() { allocCount = 0; currentDesc = 0; }
            uint32_t getCurrentAbsoluteIndex() const { return chunkIndex * kDescPerChunk + currentDesc; }

            uint32_t chunkIndex = 0;
            uint32_t chunkCount = 1; // For outstanding requests we can allocate more then a single chunk. This is the number of chunks we actually allocated
            uint32_t allocCount = 0;
            uint32_t currentDesc = 0;
        };

        Chunk::SharedPtr mpCurrentChunk;
        bool setupCurrentChunk(uint32_t descCount);
        void releaseChunk(Chunk::SharedPtr pChunk);
        std::queue<Chunk::SharedPtr> mFreeChunks;
    };
}
