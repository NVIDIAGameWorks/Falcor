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
#include "stdafx.h"
#include "Core/API/Buffer.h"
#include "Core/API/Device.h"
#include "D3D12Resource.h"

namespace Falcor
{
    ID3D12ResourcePtr createBuffer(Buffer::State initState, size_t size, const D3D12_HEAP_PROPERTIES& heapProps, Buffer::BindFlags bindFlags)
    {
        FALCOR_ASSERT(gpDevice);
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
        FALCOR_ASSERT(bufDesc.Width > 0);

        D3D12_RESOURCE_STATES d3dState = getD3D12ResourceState(initState);
        ID3D12ResourcePtr pApiHandle;
        D3D12_HEAP_FLAGS heapFlags = is_set(bindFlags, ResourceBindFlags::Shared) ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE;
        FALCOR_D3D_CALL(pDevice->CreateCommittedResource(&heapProps, heapFlags, &bufDesc, d3dState, nullptr, IID_PPV_ARGS(&pApiHandle)));
        FALCOR_ASSERT(pApiHandle);

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
        FALCOR_D3D_CALL(apiHandle->Map(0, &r, &pData));
        return pData;
    }

    void Buffer::apiInit(bool hasInitData)
    {
        if (mCpuAccess != CpuAccess::None && is_set(mBindFlags, BindFlags::Shared))
        {
            throw ArgumentError("Can't create shared resource with CPU access other than 'None'.");
        }

        if (mBindFlags == BindFlags::Constant)
        {
            mSize = align_to((size_t)D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, mSize);
        }

        if (mCpuAccess == CpuAccess::Write)
        {
            mState.global = Resource::State::GenericRead;
            if(hasInitData == false) // Else the allocation will happen when updating the data
            {
                FALCOR_ASSERT(gpDevice);
                mDynamicData = gpDevice->getUploadHeap()->allocate(mSize, getBufferDataAlignment(this));
                mApiHandle = mDynamicData.pResourceHandle;
                mGpuVaOffset = mDynamicData.offset;
            }
        }
        else if (mCpuAccess == CpuAccess::Read && mBindFlags == BindFlags::None)
        {
            mState.global = Resource::State::CopyDest;
            mApiHandle = createBuffer(mState.global, mSize, kReadbackHeapProps, mBindFlags);
        }
        else
        {
            mState.global = Resource::State::Common;
            if (is_set(mBindFlags, BindFlags::AccelerationStructure)) mState.global = Resource::State::AccelerationStructure;
            mApiHandle = createBuffer(mState.global, mSize, kDefaultHeapProps, mBindFlags);
        }
    }

    uint64_t Buffer::getGpuAddress() const
    {
        return mGpuVaOffset + mApiHandle->GetGPUVirtualAddress();
    }

    void Buffer::unmap()
    {
        // Only unmap read buffers, write buffers are persistently mapped
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

    Buffer::SharedPtr Buffer::createFromD3D12Handle(D3D12ResourceHandle handle, size_t size, Resource::BindFlags bindFlags, CpuAccess cpuAccess)
    {
        return Buffer::createFromApiHandle(handle, size, bindFlags, cpuAccess);
    }
}
