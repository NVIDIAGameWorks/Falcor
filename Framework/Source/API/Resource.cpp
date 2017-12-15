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
#include "Framework.h"
#include "Resource.h"
#include "Texture.h"

namespace Falcor
{
    Resource::~Resource() = default;

    const std::string to_string(Resource::Type type)
    {
#define type_2_string(a) case Resource::Type::a: return #a;
        switch (type)
        {
            type_2_string(Buffer);
            type_2_string(Texture1D);
            type_2_string(Texture2D);
            type_2_string(Texture3D);
            type_2_string(TextureCube);
            type_2_string(Texture2DMultisample);
        default:
            should_not_get_here();
            return "";
        }
#undef type_2_string
    }

    const std::string to_string(Resource::BindFlags flags)
    {
        std::string s;
        if (flags == Resource::BindFlags::None)
        {
            return "None";
        }

#define flag_to_str(f_) if (is_set(flags, Resource::BindFlags::f_)) (s += (s.size() ? " | " : "") + std::string(#f_))

        flag_to_str(Vertex);
        flag_to_str(Index);
        flag_to_str(Constant);
        flag_to_str(StreamOutput);
        flag_to_str(ShaderResource);
        flag_to_str(UnorderedAccess);
        flag_to_str(RenderTarget);
        flag_to_str(DepthStencil);
#undef flag_to_str

        return s;
    }

    const std::string to_string(Resource::State state)
    {
        if (state == Resource::State::Common)
        {
            return "Common";
        }
        std::string s;
#define state_to_str(f_) if (state == Resource::State::f_) {return #f_; }

        state_to_str(Common);
        state_to_str(VertexBuffer);
        state_to_str(ConstantBuffer);
        state_to_str(IndexBuffer);
        state_to_str(RenderTarget);
        state_to_str(UnorderedAccess);
        state_to_str(DepthStencil);
        state_to_str(ShaderResource);
        state_to_str(StreamOut);
        state_to_str(IndirectArg);
        state_to_str(CopyDest);
        state_to_str(CopySource);
        state_to_str(ResolveDest);
        state_to_str(ResolveSource);
        state_to_str(Present);
        state_to_str(Predication);
#undef state_to_str
        return s;
    }

    template<typename ViewClass>
    using CreateFuncType = std::function<typename ViewClass::SharedPtr(const Resource* pResource, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)>;

    template<typename ViewClass, typename ViewMapType>
    typename ViewClass::SharedPtr findViewCommon(const Resource* pResource, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize, ViewMapType& viewMap, CreateFuncType<ViewClass> createFunc)
    {
        uint32_t resMipCount = 1;
        uint32_t resArraySize = 1;

        const Texture* pTexture = dynamic_cast<const Texture*>(pResource);

        if (pTexture)
        {
            resArraySize = pTexture->getArraySize();
            resMipCount = pTexture->getMipCount();
        }
        else
        {
            assert(pResource->getType() == Resource::Type::Buffer);
        }

        if (firstArraySlice >= resArraySize)
        {
            logWarning("First array slice is OOB when creating resource view. Clamping");
            firstArraySlice = resArraySize - 1;
        }

        if (mostDetailedMip >= resMipCount)
        {
            logWarning("Most detailed mip is OOB when creating resource view. Clamping");
            mostDetailedMip = resMipCount - 1;
        }

        if (mipCount == Resource::kMaxPossible)
        {
            mipCount = resMipCount - mostDetailedMip;
        }
        else if (mipCount + mostDetailedMip > resMipCount)
        {
            logWarning("Mip count is OOB when creating resource view. Clamping");
            mipCount = resMipCount - mostDetailedMip;
        }

        if (arraySize == Resource::kMaxPossible)
        {
            arraySize = resArraySize - firstArraySlice;
        }
        else if (arraySize + firstArraySlice > resArraySize)
        {
            logWarning("Array size is OOB when creating resource view. Clamping");
            arraySize = resArraySize - firstArraySlice;
        }

        ResourceViewInfo view = ResourceViewInfo(mostDetailedMip, mipCount, firstArraySlice, arraySize);

        if (viewMap.find(view) == viewMap.end())
        {
            viewMap[view] = createFunc(pResource, mostDetailedMip, mipCount, firstArraySlice, arraySize);
        }

        return viewMap[view];
    }

    DepthStencilView::SharedPtr Resource::getDSV(uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize) const
    {
        auto createFunc = [](const Resource* pResource, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
        {
            return DepthStencilView::create(pResource->shared_from_this(), mostDetailedMip, firstArraySlice, arraySize);
        };

        return findViewCommon<DepthStencilView>(this, mipLevel, 1, firstArraySlice, arraySize, mDsvs, createFunc);
    }

    UnorderedAccessView::SharedPtr Resource::getUAV(uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize) const
    {
        auto createFunc = [](const Resource* pResource, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
        {
            return UnorderedAccessView::create(pResource->shared_from_this(), mostDetailedMip, firstArraySlice, arraySize);
        };

        return findViewCommon<UnorderedAccessView>(this, mipLevel, 1, firstArraySlice, arraySize, mUavs, createFunc);
    }

    RenderTargetView::SharedPtr Resource::getRTV(uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize) const
    {
        auto createFunc = [](const Resource* pResource, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
        {
            return RenderTargetView::create(pResource->shared_from_this(), mostDetailedMip, firstArraySlice, arraySize);
        };

        return findViewCommon<RenderTargetView>(this, mipLevel, 1, firstArraySlice, arraySize, mRtvs, createFunc);
    }

    ShaderResourceView::SharedPtr Resource::getSRV(uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize) const
    {
        auto createFunc = [](const Resource* pResource, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
        {
            return ShaderResourceView::create(pResource->shared_from_this(), mostDetailedMip, mipCount, firstArraySlice, arraySize);
        };

        return findViewCommon<ShaderResourceView>(this, mostDetailedMip, mipCount, firstArraySlice, arraySize, mSrvs, createFunc);
    }

    void Resource::invalidateViews() const
    {
        logInfo("Invalidating resource views");
        mSrvs.clear();
        mUavs.clear();
        mRtvs.clear();
        mDsvs.clear();
    }
}