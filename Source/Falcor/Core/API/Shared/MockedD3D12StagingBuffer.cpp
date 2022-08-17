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
#if FALCOR_HAS_D3D12

#include "MockedD3D12StagingBuffer.h"
#include "Core/API/Buffer.h"

namespace Falcor
{
    void MockedD3D12StagingBuffer::resize(size_t size)
    {
        mData.resize(size);
        mpGpuBuffer = Buffer::create(size, Resource::BindFlags::Constant | Resource::BindFlags::ShaderResource, Falcor::Buffer::CpuAccess::Write);
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::QueryInterface(REFIID riid, void** ppvObject)
    {
        return mpGpuBuffer->getD3D12Handle()->QueryInterface(riid, ppvObject);
    }

    ULONG __stdcall MockedD3D12StagingBuffer::AddRef(void)
    {
        return mpGpuBuffer->getD3D12Handle()->AddRef();
    }

    ULONG __stdcall MockedD3D12StagingBuffer::Release(void)
    {
        return mpGpuBuffer->getD3D12Handle()->Release();
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::GetPrivateData(REFGUID guid, UINT* pDataSize, void* pData)
    {
        return mpGpuBuffer->getD3D12Handle()->GetPrivateData(guid, pDataSize, pData);
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::SetPrivateData(REFGUID guid, UINT DataSize, const void* pData)
    {
        return mpGpuBuffer->getD3D12Handle()->SetPrivateData(guid, DataSize, pData);
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::SetPrivateDataInterface(REFGUID guid, const IUnknown* pData)
    {
        return mpGpuBuffer->getD3D12Handle()->SetPrivateDataInterface(guid, pData);
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::SetName(LPCWSTR Name)
    {
        return mpGpuBuffer->getD3D12Handle()->SetName(Name);
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::GetDevice(REFIID riid, void** ppvDevice)
    {
        return mpGpuBuffer->getD3D12Handle()->GetDevice(riid, ppvDevice);
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::Map(UINT Subresource, const D3D12_RANGE* pReadRange, void** ppData)
    {
        *ppData = mData.data();
        return 0;
    }

    void __stdcall MockedD3D12StagingBuffer::Unmap(UINT Subresource, const D3D12_RANGE* pWrittenRange)
    {
        // Write CPU data into GPU buffer.
        mpGpuBuffer->setBlob(mData.data(), 0, mData.size());
    }

    D3D12_RESOURCE_DESC __stdcall MockedD3D12StagingBuffer::GetDesc(void)
    {
        return mpGpuBuffer->getD3D12Handle()->GetDesc();
    }

    D3D12_GPU_VIRTUAL_ADDRESS __stdcall MockedD3D12StagingBuffer::GetGPUVirtualAddress(void)
    {
        return mpGpuBuffer->getGpuAddress();
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::WriteToSubresource(UINT DstSubresource, const D3D12_BOX* pDstBox, const void* pSrcData, UINT SrcRowPitch, UINT SrcDepthPitch)
    {
        return mpGpuBuffer->getD3D12Handle()->WriteToSubresource(DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch);
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::ReadFromSubresource(void* pDstData, UINT DstRowPitch, UINT DstDepthPitch, UINT SrcSubresource, const D3D12_BOX* pSrcBox)
    {
        return mpGpuBuffer->getD3D12Handle()->ReadFromSubresource(pDstData, DstRowPitch, DstDepthPitch, SrcSubresource, pSrcBox);
    }

    HRESULT __stdcall MockedD3D12StagingBuffer::GetHeapProperties(D3D12_HEAP_PROPERTIES* pHeapProperties, D3D12_HEAP_FLAGS* pHeapFlags)
    {
        return mpGpuBuffer->getD3D12Handle()->GetHeapProperties(pHeapProperties, pHeapFlags);
    }
}

#endif // FALCOR_HAS_D3D12
