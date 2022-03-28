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

#include "stdafx.h"

namespace Falcor
{
#ifdef FALCOR_D3D12_AVAILABLE
    // A mocked `ID3D12Resource` that supports mapped write and allow reading contents directly
    // from CPU memory.
    // This is currently used to call `DDGIVolume::Update` to receive the contents to write into
    // a ParameterBlock later without reading back from GPU memory.
    // Since DDGIVolume will use the buffer passed into `DDGIVolume::Update` to run its internal
    // passes, this buffer implementation also provides an actual GPU resource for those passes.
    // The only methods that matter here are `Map` `Unmap` and `GetGPUVirtualAddress`.
    // In `Map`, we just return a CPU memory allocation so the SDK can write update-to-date data
    // into it.
    // In `Unmap`, we update our internal GPU buffer with the contents that the SDK just wrote into.
    // In `GetGPUVirtualAddress`, we return the address of the GPU buffer, so the SDK can use it to
    // run its internal passes.
    // 
    // With this class, we have a temporary solution that avoids the hackery around
    // `ParameterBlock::getUnderlyingConstantBuffer`.
    // When `DDGIVolume` provides a better interface to allow us to get the constant buffer data without
    // GPU readback in the future, this class can be removed.
    //
    class FALCOR_API MockedD3D12StagingBuffer : public ID3D12Resource
    {
    public:
        void resize(size_t size);

        size_t getSize() const { return mData.size(); }
        const void* getData() const { return mData.data(); }

        // Inherited via ID3D12Resource
        virtual HRESULT __stdcall QueryInterface(REFIID riid, void** ppvObject) override;
        virtual ULONG __stdcall AddRef(void) override;
        virtual ULONG __stdcall Release(void) override;
        virtual HRESULT __stdcall GetPrivateData(REFGUID guid, UINT* pDataSize, void* pData) override;
        virtual HRESULT __stdcall SetPrivateData(REFGUID guid, UINT DataSize, const void* pData) override;
        virtual HRESULT __stdcall SetPrivateDataInterface(REFGUID guid, const IUnknown* pData) override;
        virtual HRESULT __stdcall SetName(LPCWSTR Name) override;
        virtual HRESULT __stdcall GetDevice(REFIID riid, void** ppvDevice) override;
        virtual HRESULT __stdcall Map(UINT Subresource, const D3D12_RANGE* pReadRange, void** ppData) override;
        virtual void __stdcall Unmap(UINT Subresource, const D3D12_RANGE* pWrittenRange) override;
        virtual D3D12_RESOURCE_DESC __stdcall GetDesc(void) override;
        virtual D3D12_GPU_VIRTUAL_ADDRESS __stdcall GetGPUVirtualAddress(void) override;
        virtual HRESULT __stdcall WriteToSubresource(UINT DstSubresource, const D3D12_BOX* pDstBox, const void* pSrcData, UINT SrcRowPitch, UINT SrcDepthPitch) override;
        virtual HRESULT __stdcall ReadFromSubresource(void* pDstData, UINT DstRowPitch, UINT DstDepthPitch, UINT SrcSubresource, const D3D12_BOX* pSrcBox) override;
        virtual HRESULT __stdcall GetHeapProperties(D3D12_HEAP_PROPERTIES* pHeapProperties, D3D12_HEAP_FLAGS* pHeapFlags) override;

    private:
        std::vector<uint8_t> mData; // CPU Buffer.
        Buffer::SharedPtr mpGpuBuffer; // GPU Buffer.
    };

#endif // FALCOR_D3D12_AVAILABLE
}
