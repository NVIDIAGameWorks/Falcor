/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <queue>

namespace Falcor
{
    struct DescriptorPoolApiData;
    struct DescriptorSetApiData;

    class dlldecl DescriptorPool : public std::enable_shared_from_this<DescriptorPool>
    {
    public:
        using SharedPtr = std::shared_ptr<DescriptorPool>;
        using SharedConstPtr = std::shared_ptr<const DescriptorPool>;
        using ApiHandle = DescriptorHeapHandle;
        using CpuHandle = HeapCpuHandle;
        using GpuHandle = HeapGpuHandle;
        using ApiData = DescriptorPoolApiData;

        ~DescriptorPool();

        enum class Type
        {
            TextureSrv,
            TextureUav,
            RawBufferSrv,
            RawBufferUav,
            TypedBufferSrv,
            TypedBufferUav,
            Cbv,
            StructuredBufferUav,
            StructuredBufferSrv,
            Dsv,
            Rtv,
            Sampler,

            Count
        };

        static const uint32_t kTypeCount = uint32_t(Type::Count);

        class dlldecl Desc
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
            friend DescriptorPool;
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

    private:
        friend DescriptorSet;
        DescriptorPool(const Desc& desc, const GpuFence::SharedPtr & pFence);
        void apiInit();
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
