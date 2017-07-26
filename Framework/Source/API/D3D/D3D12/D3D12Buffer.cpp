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
#include "API/Buffer.h"
#include "API/Device.h"
#include "Api/LowLevel/ResourceAllocator.h"
#include "D3D12Resource.h"

namespace Falcor
{

    ID3D12ResourcePtr createBuffer(Buffer::State initState, size_t size, const D3D12_HEAP_PROPERTIES& heapProps, Buffer::BindFlags bindFlags)
    {
        ID3D12Device* pDevice = gpDevice->getApiHandle();

        // Create the buffer
        D3D12_RESOURCE_DESC bufDesc = {};
        bufDesc.Alignment = 0;
        bufDesc.DepthOrArraySize = 1;
        bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufDesc.Flags = getD3D12ResourceFlags(bindFlags);
        bufDesc.Format = DXGI_FORMAT_UNKNOWN;
        bufDesc.Height = 1;
        bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        bufDesc.MipLevels = 1;
        bufDesc.SampleDesc.Count = 1;
        bufDesc.SampleDesc.Quality = 0;
        bufDesc.Width = size;

        D3D12_RESOURCE_STATES d3dState = getD3D12ResourceState(initState);
        ID3D12ResourcePtr pApiHandle;
        d3d_call(pDevice->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufDesc, d3dState, nullptr, IID_PPV_ARGS(&pApiHandle)));

        // Map and upload data if needed
        return pApiHandle;
    }

    size_t getBufferDataAlignment(const Buffer* pBuffer)
    {
		// This in order of the alignment size
		const auto& bindFlags = pBuffer->getBindFlags();
		if (is_set(bindFlags, Buffer::BindFlags::Constant)) return D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
		if (is_set(bindFlags, Buffer::BindFlags::Index)) return sizeof(uint32_t); // This actually depends on the size of the index, but we can handle losing 2 bytes

		return D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
	}

    void* mapBufferApi(const Buffer::ApiHandle& apiHandle, size_t size)
    {
        D3D12_RANGE r{ 0, size };
        void* pData;
        d3d_call(apiHandle->Map(0, &r, &pData));
        return pData;
    }

    bool Buffer::apiInit(bool hasInitData)
    {
        if (mBindFlags == BindFlags::Constant)
        {
            mSize = align_to(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, mSize);
        }

        if (mCpuAccess == CpuAccess::Write)
        {
            mState = Resource::State::GenericRead;
            if(hasInitData == false) // Else the allocation will happen when updating the data
            {
                mDynamicData = gpDevice->getResourceAllocator()->allocate(mSize, getBufferDataAlignment(this));
                mApiHandle = mDynamicData.pResourceHandle;
            }
        }
        else if (mCpuAccess == CpuAccess::Read && mBindFlags == BindFlags::None)
        {
            mState = Resource::State::CopyDest;
            mApiHandle = createBuffer(mState, mSize, kReadbackHeapProps, mBindFlags);
        }
        else
        {
            mState = Resource::State::Common;
            mApiHandle = createBuffer(mState, mSize, kDefaultHeapProps, mBindFlags);
        }

        return true;
    }

    uint64_t Buffer::getGpuAddress() const
    {
        return mDynamicData.offset + mApiHandle->GetGPUVirtualAddress();
    }

    void Buffer::unmap()
    {
        // Only unmap read buffers
        D3D12_RANGE r{};
        if (mpStagingResource)
        {
            mpStagingResource->mApiHandle->Unmap(0, &r);
        }
        else if (mCpuAccess == CpuAccess::Read)
        {
            mApiHandle->Unmap(0, &r);
        }
    }

    uint64_t Buffer::makeResident(Buffer::GpuAccessFlags flags) const
    {
        UNSUPPORTED_IN_D3D12("Buffer::makeResident()");
        return 0;
    }

    void Buffer::evict() const
    {
        UNSUPPORTED_IN_D3D12("Buffer::evict()");
    }

    template<bool forClear>
    UavHandle getUavCommon(UavHandle& handle, size_t bufSize, Buffer::ApiHandle apiHandle)
    {
        if (handle == nullptr)
        {

            DescriptorHeap* pHeap = forClear ? gpDevice->getCpuUavDescriptorHeap().get() : gpDevice->getUavDescriptorHeap().get();
            handle = pHeap->allocateEntry();
            gpDevice->getApiHandle()->CreateUnorderedAccessView(apiHandle, nullptr, &desc, handle->getCpuHandle());
        }

        return handle;
    }
}
