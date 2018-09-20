/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "API/D3D12/FalcorD3D12.h"

namespace Falcor
{
    struct SlangBlob : ID3DBlob
    {
        std::vector<uint8_t> data;
        size_t refCount;

        SlangBlob(const void* buffer, size_t bufferSize)
            : data((uint8_t*)buffer, ((uint8_t*)buffer) + bufferSize)
            , refCount(1)
        {}

        // IUnknown

        virtual HRESULT STDMETHODCALLTYPE QueryInterface(
            /* [in] */ REFIID riid,
            /* [iid_is][out] */ _COM_Outptr_ void __RPC_FAR *__RPC_FAR *ppvObject) override
        {
            *ppvObject = this;
            return S_OK;
        }

        virtual ULONG STDMETHODCALLTYPE AddRef(void) override
        {
            ++refCount;
            return (ULONG)refCount;
        }

        virtual ULONG STDMETHODCALLTYPE Release(void) override
        {
            --refCount;
            if (refCount == 0)
            {
                delete this;
            }
            return (ULONG)refCount;
        }

        // ID3DBlob

        virtual LPVOID STDMETHODCALLTYPE GetBufferPointer() override
        {
            return data.data();
        }

        virtual SIZE_T STDMETHODCALLTYPE GetBufferSize() override
        {
            return data.size();
        }
    };
}