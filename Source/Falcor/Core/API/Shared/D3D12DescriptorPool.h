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
#include "Core/Macros.h"
#include "Core/API/ShaderResourceType.h"
#include "Core/API/GpuFence.h"
#include <memory>
#include <queue>

namespace Falcor
{
    struct DescriptorPoolApiData;
    struct DescriptorSetApiData;

    class FALCOR_API D3D12DescriptorPool
    {
    public:
        using SharedPtr = std::shared_ptr<D3D12DescriptorPool>;
        using SharedConstPtr = std::shared_ptr<const D3D12DescriptorPool>;
        using ApiHandle = D3D12DescriptorHeapHandle;
        using CpuHandle = D3D12DescriptorCpuHandle;
        using GpuHandle = D3D12DescriptorGpuHandle;
        using ApiData = DescriptorPoolApiData;
        using Type = ShaderResourceType;

        ~D3D12DescriptorPool();

        static const uint32_t kTypeCount = uint32_t(Type::Count);

        class FALCOR_API Desc
        {
        public:
            Desc& setDescCount(Type type, uint32_t count)
            {
                uint32_t t = (uint32_t)type;
                mTotalDescCount -= mDescCount[t];
                mTotalDescCount += count;
                mDescCount[t] = count;
                return *this;
            }

            Desc& setShaderVisible(bool visible) { mShaderVisible = visible; return *this; }
        private:
            friend D3D12DescriptorPool;
            uint32_t mDescCount[kTypeCount] = { 0 };
            uint32_t mTotalDescCount = 0;
            bool mShaderVisible = false;
        };

        /** Create a new descriptor pool.
            \param[in] desc Description of the descriptor type and count.
            \param[in] pFence Fence object for synchronization.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const Desc& desc, const GpuFence::SharedPtr& pFence);

        uint32_t getDescCount(Type type) const { return mDesc.mDescCount[(uint32_t)type]; }
        uint32_t getTotalDescCount() const { return mDesc.mTotalDescCount; }
        bool isShaderVisible() const { return mDesc.mShaderVisible; }
        const ApiHandle& getApiHandle(uint32_t heapIndex) const;
        const ApiData* getApiData() const { return mpApiData.get(); }
        void executeDeferredReleases();

        static uint32_t getMaxShaderVisibleSamplerHeapSize();

    private:
        friend class D3D12DescriptorSet;
        D3D12DescriptorPool(const Desc& desc, const GpuFence::SharedPtr & pFence);
        void releaseAllocation(std::shared_ptr<DescriptorSetApiData> pData);
        Desc mDesc;
        std::shared_ptr<ApiData> mpApiData;
        GpuFence::SharedPtr mpFence;

        struct DeferredRelease
        {
            std::shared_ptr<DescriptorSetApiData> pData;
            uint64_t fenceValue;
            bool operator>(const DeferredRelease& other) const { return fenceValue > other.fenceValue; }
        };

        std::priority_queue<DeferredRelease, std::vector<DeferredRelease>, std::greater<DeferredRelease>> mpDeferredReleases;
    };
}
