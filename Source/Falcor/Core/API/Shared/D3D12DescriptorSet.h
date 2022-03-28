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
#include "D3D12DescriptorPool.h"

namespace Falcor
{
    class ShaderResourceView;
    class UnorderedAccessView;
    class ConstantBufferView;
    class Sampler;
    class CopyContext;
    class D3D12RootSignature;

    enum class ShaderVisibility
    {
        None = 0,
        Vertex = (1 << (uint32_t)ShaderType::Vertex),
        Pixel = (1 << (uint32_t)ShaderType::Pixel),
        Hull = (1 << (uint32_t)ShaderType::Hull),
        Domain = (1 << (uint32_t)ShaderType::Domain),
        Geometry = (1 << (uint32_t)ShaderType::Geometry),
        Compute = (1 << (uint32_t)ShaderType::Compute),

        All = (1 << (uint32_t)ShaderType::Count) - 1,
    };

    FALCOR_ENUM_CLASS_OPERATORS(ShaderVisibility);

    class FALCOR_API D3D12DescriptorSet
    {
    public:
        using SharedPtr = std::shared_ptr<D3D12DescriptorSet>;
        using Type = ShaderResourceType;
        using CpuHandle = D3D12DescriptorPool::CpuHandle;
        using GpuHandle = D3D12DescriptorPool::GpuHandle;
        using ApiHandle = D3D12DescriptorSetApiHandle;
        using ApiData = DescriptorSetApiData;

        ~D3D12DescriptorSet();

        class FALCOR_API Layout
        {
        public:
            struct Range
            {
                Type type;
                uint32_t baseRegIndex;
                uint32_t descCount;
                uint32_t regSpace;
            };

            Layout(ShaderVisibility visibility = ShaderVisibility::All) : mVisibility(visibility) {}
            Layout& addRange(Type type, uint32_t baseRegIndex, uint32_t descriptorCount, uint32_t regSpace = 0);
            size_t getRangeCount() const { return mRanges.size(); }
            const Range& getRange(size_t index) const { return mRanges[index]; }
            ShaderVisibility getVisibility() const { return mVisibility; }
        private:
            std::vector<Range> mRanges;
            ShaderVisibility mVisibility;
        };

        /** Create a new descriptor set.
            \param[in] pPool The descriptor pool.
            \param[in] layout The layout.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const D3D12DescriptorPool::SharedPtr& pPool, const Layout& layout);

        size_t getRangeCount() const { return mLayout.getRangeCount(); }
        const Layout::Range& getRange(uint32_t range) const { return mLayout.getRange(range); }
        ShaderVisibility getVisibility() const { return mLayout.getVisibility(); }

        CpuHandle getCpuHandle(uint32_t rangeIndex, uint32_t descInRange = 0) const;
        GpuHandle getGpuHandle(uint32_t rangeIndex, uint32_t descInRange = 0) const;
        const ApiHandle& getApiHandle() const { return mApiHandle; }
        const ApiData* getApiData() const { return mpApiData.get(); }

        void setSrv(uint32_t rangeIndex, uint32_t descIndex, const ShaderResourceView* pSrv);
        void setUav(uint32_t rangeIndex, uint32_t descIndex, const UnorderedAccessView* pUav);
        void setSampler(uint32_t rangeIndex, uint32_t descIndex, const Sampler* pSampler);
        void setCbv(uint32_t rangeIndex, uint32_t descIndex, ConstantBufferView* pView);

        void bindForGraphics(CopyContext* pCtx, const D3D12RootSignature* pRootSig, uint32_t rootIndex);
        void bindForCompute(CopyContext* pCtx, const D3D12RootSignature* pRootSig, uint32_t rootIndex);

    private:
        D3D12DescriptorSet(D3D12DescriptorPool::SharedPtr pPool, const Layout& layout);

        Layout mLayout;
        std::shared_ptr<ApiData> mpApiData;
        D3D12DescriptorPool::SharedPtr mpPool;
        ApiHandle mApiHandle = {};
    };
}
