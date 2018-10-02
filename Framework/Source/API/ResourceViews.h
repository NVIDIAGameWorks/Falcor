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
#include <vector>

namespace Falcor
{
    class Resource;
    using ResourceWeakPtr = std::weak_ptr<const Resource>;

    struct ResourceViewInfo
    {
        ResourceViewInfo() = default;
        ResourceViewInfo(uint32_t mostDetailedMip_, uint32_t mipCount_, uint32_t firstArraySlice_, uint32_t arraySize_) : mostDetailedMip(mostDetailedMip_), mipCount(mipCount_), firstArraySlice(firstArraySlice_), arraySize(arraySize_) {}
        uint32_t mostDetailedMip = 0;
        uint32_t mipCount = kMaxPossible;
        uint32_t firstArraySlice = 0;
        uint32_t arraySize = kMaxPossible;

        static const uint32_t kMaxPossible = -1;
        bool operator==(const ResourceViewInfo& other) const
        {
            return (firstArraySlice == other.firstArraySlice) && (arraySize == other.arraySize) && (mipCount == other.mipCount) && (mostDetailedMip == other.mostDetailedMip);
        }
    };

    /** Abstracts API resource views.
    */
    template<typename ApiHandleType>
    class ResourceView
    {
    public:
        using ApiHandle = ApiHandleType;
        static const uint32_t kMaxPossible = -1;
        virtual ~ResourceView();

        ResourceView(ResourceWeakPtr& pResource, ApiHandle handle, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
            : mApiHandle(handle), mpResource(pResource), mViewInfo(mostDetailedMip, mipCount, firstArraySlice, arraySize) {}

        /** Get the raw API handle.
        */
        const ApiHandle& getApiHandle() const { return mApiHandle; }

        /** Get information about the view.
        */
        const ResourceViewInfo& getViewInfo() const { return mViewInfo; }

        /** Get the resource referenced by the view.
        */
        const Resource* getResource() const { return mpResource.lock().get(); }
    protected:
        ApiHandle mApiHandle;
        ResourceViewInfo mViewInfo;
        ResourceWeakPtr mpResource;
    };

    class ShaderResourceView : public ResourceView<SrvHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<ShaderResourceView>;
        using SharedConstPtr = std::shared_ptr<const ShaderResourceView>;
        
        static SharedPtr create(ResourceWeakPtr pResource, uint32_t mostDetailedMip = 0, uint32_t mipCount = kMaxPossible, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);
        static SharedPtr getNullView();
        ShaderResourceView(ResourceWeakPtr& pResource, ApiHandle handle, uint32_t mostDetailedMip_, uint32_t mipCount_, uint32_t firstArraySlice_, uint32_t arraySize_) :
            ResourceView(pResource, handle, mostDetailedMip_, mipCount_, firstArraySlice_, arraySize_) {}
    private:
    };

    class DepthStencilView : public ResourceView<DsvHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<DepthStencilView>;
        using SharedConstPtr = std::shared_ptr<const DepthStencilView>;

        static SharedPtr create(ResourceWeakPtr pResource, uint32_t mipLevel, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);
        static SharedPtr getNullView();
    private:
        DepthStencilView(ResourceWeakPtr& pResource, ApiHandle handle, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize) :
            ResourceView(pResource, handle, mipLevel, 1, firstArraySlice, arraySize) {}
    };

    class UnorderedAccessView : public ResourceView<UavHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<UnorderedAccessView>;
        using SharedConstPtr = std::shared_ptr<const UnorderedAccessView>;
        static SharedPtr create(ResourceWeakPtr pResource, uint32_t mipLevel, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);
        static SharedPtr getNullView();
    private:
        UnorderedAccessView(ResourceWeakPtr& pResource, ApiHandle handle, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize) :
            ResourceView(pResource, handle, mipLevel, 1, firstArraySlice, arraySize) {}
    };

    class RenderTargetView : public ResourceView<RtvHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<RenderTargetView>;
        using SharedConstPtr = std::shared_ptr<const RenderTargetView>;
        static SharedPtr create(ResourceWeakPtr pResource, uint32_t mipLevel, uint32_t firstArraySlice = 0, uint32_t arraySize = kMaxPossible);
        static SharedPtr getNullView();
        ~RenderTargetView();
    private:
        RenderTargetView(ResourceWeakPtr& pResource, ApiHandle handle, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize) :
            ResourceView(pResource, handle, mipLevel, 1, firstArraySlice, arraySize) {}
    };

    class ConstantBufferView : public ResourceView<CbvHandle>
    {
    public:
        using SharedPtr = std::shared_ptr<ConstantBufferView>;
        using SharedConstPtr = std::shared_ptr<const ConstantBufferView>;
        static SharedPtr create(ResourceWeakPtr pResource);
        static SharedPtr getNullView();

    private:
        ConstantBufferView(ResourceWeakPtr& pResource, ApiHandle handle) :
            ResourceView(pResource, handle, 0, 1, 0, 1) {}        
    };

    dlldecl ShaderResourceView::SharedPtr gNullSrv;
    dlldecl ConstantBufferView::SharedPtr gNullCbv;
    dlldecl RenderTargetView::SharedPtr   gNullRtv;
    dlldecl UnorderedAccessView::SharedPtr gNullUav;
    dlldecl DepthStencilView::SharedPtr gNullDsv;
}
