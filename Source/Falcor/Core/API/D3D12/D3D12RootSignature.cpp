/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/API/RootSignature.h"
#include "D3D12State.h"
#include "Core/API/Device.h"

namespace Falcor
{
    void RootSignature::apiInit()
    {
        // Get vector of root parameters
        RootSignatureParams params;
        initD3D12RootParams(mDesc, params);

        // Create the root signature
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC versionedDesc = {};
        versionedDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;

        D3D12_ROOT_SIGNATURE_DESC1& desc = versionedDesc.Desc_1_1;
        desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        if (mDesc.mIsLocal) desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_LOCAL_ROOT_SIGNATURE;
        mSizeInBytes = params.signatureSizeInBytes; 
        mElementByteOffset = params.elementByteOffset;

        desc.pParameters = params.rootParams.data();
        desc.NumParameters = (uint32_t)params.rootParams.size();
        desc.pStaticSamplers = nullptr;
        desc.NumStaticSamplers = 0;

        // Create versioned root signature
        ID3DBlobPtr pSigBlob;
        ID3DBlobPtr pErrorBlob;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&versionedDesc, &pSigBlob, &pErrorBlob);
        if (FAILED(hr))
        {
            std::string msg = convertBlobToString(pErrorBlob.GetInterfacePtr());
            throw std::exception(("Failed to create root signature:\n" + msg).c_str());
        }

        if (mSizeInBytes > sizeof(uint32_t) * D3D12_MAX_ROOT_COST)
        {
            throw std::exception(("Root signature cost is too high. D3D12 root signatures are limited to 64 DWORDs, trying to create a signature with " + std::to_string(mSizeInBytes / sizeof(uint32_t)) + " DWORDs.").c_str());
        }

        createApiHandle(pSigBlob);
    }

    void RootSignature::createApiHandle(ID3DBlobPtr pSigBlob)
    {
        Device::ApiHandle pDevice = gpDevice->getApiHandle();
        d3d_call(pDevice->CreateRootSignature(0, pSigBlob->GetBufferPointer(), pSigBlob->GetBufferSize(), IID_PPV_ARGS(&mApiHandle)));
    }

    template<bool forGraphics>
    static void bindRootSigCommon(CopyContext* pCtx, const RootSignature::ApiHandle& rootSig)
    {
        if (forGraphics)
        {
            pCtx->getLowLevelData()->getCommandList()->SetGraphicsRootSignature(rootSig);
        }
        else
        {
            pCtx->getLowLevelData()->getCommandList()->SetComputeRootSignature(rootSig);
        }
    }

    void RootSignature::bindForCompute(CopyContext* pCtx)
    {
        bindRootSigCommon<false>(pCtx, mApiHandle);
    }

    void RootSignature::bindForGraphics(CopyContext* pCtx)
    {
        bindRootSigCommon<true>(pCtx, mApiHandle);
    }
}
