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
#include "LowLevel/DescriptorPool.h"
#include "ResourceViews.h"
#include "Sampler.h"

namespace Falcor
{
    struct DescriptorSetApiData;
    class CopyContext;
    class RootSignature;
    class Buffer;

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

    enum_class_operators(ShaderVisibility);

    class DescriptorSet
    {
    public:
        using SharedPtr = std::shared_ptr<DescriptorSet>;
        using Type = DescriptorPool::Type;
        using CpuHandle = DescriptorPool::CpuHandle;
        using GpuHandle = DescriptorPool::GpuHandle;
        using ApiHandle = DescriptorSetApiHandle;

        ~DescriptorSet();

        class Layout
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

        static SharedPtr create(const DescriptorPool::SharedPtr& pPool, const Layout& layout);

        size_t getRangeCount() const { return mLayout.getRangeCount(); }
        const Layout::Range& getRange(uint32_t range) const { return mLayout.getRange(range); }
        ShaderVisibility getVisibility() const { return mLayout.getVisibility(); }

        CpuHandle getCpuHandle(uint32_t rangeIndex, uint32_t descInRange = 0) const;
        GpuHandle getGpuHandle(uint32_t rangeIndex, uint32_t descInRange = 0) const;
        ApiHandle getApiHandle() const { return mApiHandle; }

        void setSrv(uint32_t rangeIndex, uint32_t descIndex, const ShaderResourceView* pSrv);
        void setUav(uint32_t rangeIndex, uint32_t descIndex, const UnorderedAccessView* pUav);
        void setSampler(uint32_t rangeIndex, uint32_t descIndex, const Sampler* pSampler);
        void setCbv(uint32_t rangeIndex, uint32_t descIndex, const ConstantBufferView::SharedPtr& pView);

        void bindForGraphics(CopyContext* pCtx, const RootSignature* pRootSig, uint32_t rootIndex);
        void bindForCompute(CopyContext* pCtx, const RootSignature* pRootSig, uint32_t rootIndex);
    private:
        using ApiData = DescriptorSetApiData;
        DescriptorSet(DescriptorPool::SharedPtr pPool, const Layout& layout) : mpPool(pPool), mLayout(layout) {}

        bool apiInit();
        Layout mLayout;
        std::shared_ptr<ApiData> mpApiData;
        DescriptorPool::SharedPtr mpPool;
        ApiHandle mApiHandle = {};
    };
}