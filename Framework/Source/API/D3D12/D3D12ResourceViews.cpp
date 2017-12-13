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
#include "API/ResourceViews.h"
#include "API/Resource.h"
#include "API/D3D12/D3DViews.h"
#include "API/Device.h"
#include "API/DescriptorSet.h"

namespace Falcor
{
    template<typename T>
    ResourceView<T>::~ResourceView() = default;

    ResourceWeakPtr getEmptyTexture()
    {
        return ResourceWeakPtr();
    }

    ShaderResourceView::SharedPtr ShaderResourceView::create(ResourceWeakPtr pResource, uint32_t mostDetailedMip, uint32_t mipCount, uint32_t firstArraySlice, uint32_t arraySize)
    {
        Resource::SharedConstPtr pSharedPtr = pResource.lock();
        if (!pSharedPtr && sNullView)
        {
            return sNullView;
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        if(pSharedPtr)
        {
            initializeSrvDesc(pSharedPtr.get(), firstArraySlice, arraySize, mostDetailedMip, mipCount, desc);
            resHandle = pSharedPtr->getApiHandle();
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        }
        SharedPtr pNewObj;
        SharedPtr& pObj = pSharedPtr ? pNewObj : sNullView;

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::TextureSrv, 0, 1);
        ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateShaderResourceView(pSharedPtr ? pSharedPtr->getApiHandle() : nullptr, &desc, handle->getCpuHandle(0));

        pObj = SharedPtr(new ShaderResourceView(pResource, handle, mostDetailedMip, mipCount, firstArraySlice, arraySize));
        return pObj;
    }

    DepthStencilView::SharedPtr DepthStencilView::create(ResourceWeakPtr pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        Resource::SharedConstPtr pSharedPtr = pResource.lock();
        if (!pSharedPtr && sNullView)
        {
            return sNullView;
        }

        D3D12_DEPTH_STENCIL_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        if(pSharedPtr)
        {
            initializeDsvDesc(pSharedPtr.get(), mipLevel, firstArraySlice, arraySize, desc);
            resHandle = pSharedPtr->getApiHandle();
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_D16_UNORM;
            desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        }
        SharedPtr pNewObj;
        SharedPtr& pObj = pSharedPtr ? pNewObj : sNullView;

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::Dsv, 0, 1);
        ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateDepthStencilView(resHandle, &desc, handle->getCpuHandle(0));

        pObj = SharedPtr(new DepthStencilView(pResource, handle, mipLevel, firstArraySlice, arraySize));
        return pObj;
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::create(ResourceWeakPtr pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        Resource::SharedConstPtr pSharedPtr = pResource.lock();

        if (!pSharedPtr && sNullView)
        {
            return sNullView;
        }

        D3D12_UNORDERED_ACCESS_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        Resource::ApiHandle counterHandle = nullptr;

        if(pSharedPtr != nullptr)
        {
            initializeUavDesc(pSharedPtr.get(), mipLevel, firstArraySlice, arraySize, desc);
            resHandle = pSharedPtr->getApiHandle();

            StructuredBuffer::SharedConstPtr pStructuredBuffer = std::dynamic_pointer_cast<const StructuredBuffer>(pSharedPtr);
            if (pStructuredBuffer != nullptr && pStructuredBuffer->hasUAVCounter())
            {
                counterHandle = pStructuredBuffer->getUAVCounter()->getApiHandle();
            }
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_R32_UINT;
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        }

        SharedPtr pNewObj;
        SharedPtr& pObj = pSharedPtr ? pNewObj : sNullView;

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::TextureUav, 0, 1);
        ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateUnorderedAccessView(resHandle, counterHandle, &desc, handle->getCpuHandle(0));

        pObj = SharedPtr(new UnorderedAccessView(pResource, handle, mipLevel, firstArraySlice, arraySize));

        return pObj;
    }

    RenderTargetView::~RenderTargetView() = default;

    RenderTargetView::SharedPtr RenderTargetView::create(ResourceWeakPtr pResource, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        Resource::SharedConstPtr pSharedPtr = pResource.lock();

        if (!pSharedPtr && sNullView)
        {
            return sNullView;
        }

        D3D12_RENDER_TARGET_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        if(pSharedPtr)
        {
            initializeRtvDesc(pSharedPtr.get(), mipLevel, firstArraySlice, arraySize, desc);
            resHandle = pSharedPtr->getApiHandle();
        }
        else
        {
            desc = {};
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;;
            desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        }

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::Rtv, 0, 1);
        ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateRenderTargetView(resHandle, &desc, handle->getCpuHandle(0));

        SharedPtr pNewObj;
        SharedPtr& pObj = pSharedPtr ? pNewObj : sNullView;

        pObj = SharedPtr(new RenderTargetView(pResource, handle, mipLevel, firstArraySlice, arraySize));
        return pObj;
    }

    ConstantBufferView::SharedPtr ConstantBufferView::create(ResourceWeakPtr pResource)
    {
        Resource::SharedConstPtr pSharedPtr = pResource.lock();

        if (!pSharedPtr && sNullView)
        {
            return sNullView;
        }

        D3D12_CONSTANT_BUFFER_VIEW_DESC desc;
        Resource::ApiHandle resHandle = nullptr;
        if (pSharedPtr)
        {
            ConstantBuffer* pBuffer = dynamic_cast<ConstantBuffer*>(const_cast<Resource*>(pSharedPtr.get()));
            desc.BufferLocation = pBuffer->getGpuAddress();
            desc.SizeInBytes = (uint32_t)pBuffer->getSize();
            resHandle = pSharedPtr->getApiHandle();
        }
        else
        {
            desc = {};
        }

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::Cbv, 0, 1);
        ApiHandle handle = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        gpDevice->getApiHandle()->CreateConstantBufferView(&desc, handle->getCpuHandle(0));

        SharedPtr pNewObj;
        SharedPtr& pObj = pSharedPtr ? pNewObj : sNullView;

        pObj = SharedPtr(new ConstantBufferView(pResource, handle));
        return pObj;
    }
}

