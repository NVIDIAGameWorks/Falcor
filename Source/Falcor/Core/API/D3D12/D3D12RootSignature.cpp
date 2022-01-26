/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "D3D12RootSignature.h"
#include "D3D12State.h"
#include "Core/API/Device.h"
#include "Core/Program/ProgramReflection.h"

namespace Falcor
{
    D3D12RootSignature::SharedPtr D3D12RootSignature::spEmptySig;
    uint64_t D3D12RootSignature::sObjCount = 0;

    D3D12RootSignature::Desc& D3D12RootSignature::Desc::addDescriptorSet(const DescriptorSetLayout& setLayout)
    {
        FALCOR_ASSERT(mRootConstants.empty()); // For now we disallow both root-constants and descriptor-sets
        FALCOR_ASSERT(setLayout.getRangeCount());
        mSets.push_back(setLayout);
        return *this;
    }

    D3D12RootSignature::Desc& D3D12RootSignature::Desc::addRootDescriptor(DescType type, uint32_t regIndex, uint32_t spaceIndex, ShaderVisibility visibility)
    {
        RootDescriptorDesc desc;
        desc.type = type;
        desc.regIndex = regIndex;
        desc.spaceIndex = spaceIndex;
        desc.visibility = visibility;
        mRootDescriptors.push_back(desc);
        return *this;
    }

    D3D12RootSignature::Desc& D3D12RootSignature::Desc::addRootConstants(uint32_t regIndex, uint32_t spaceIndex, uint32_t count)
    {
        FALCOR_ASSERT(mSets.empty()); // For now we disallow both root-constants and descriptor-sets
        RootConstantsDesc desc;
        desc.count = count;
        desc.regIndex = regIndex;
        desc.spaceIndex = spaceIndex;
        mRootConstants.push_back(desc);
        return *this;
    }

    D3D12RootSignature::D3D12RootSignature(const Desc& desc)
        : mDesc(desc)
    {
        sObjCount++;

        // Get vector of root parameters
        RootSignatureParams params;
        initD3D12RootParams(mDesc, params);

        // Create the root signature
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC versionedDesc = {};
        versionedDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;

        D3D12_ROOT_SIGNATURE_DESC1& d3dDesc = versionedDesc.Desc_1_1;
        d3dDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        mSizeInBytes = params.signatureSizeInBytes;
        mElementByteOffset = params.elementByteOffset;

        d3dDesc.pParameters = params.rootParams.data();
        d3dDesc.NumParameters = (uint32_t)params.rootParams.size();
        d3dDesc.pStaticSamplers = nullptr;
        d3dDesc.NumStaticSamplers = 0;

        // Create versioned root signature
        ID3DBlobPtr pSigBlob;
        ID3DBlobPtr pErrorBlob;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&versionedDesc, &pSigBlob, &pErrorBlob);
        if (FAILED(hr))
        {
            std::string msg = convertBlobToString(pErrorBlob.GetInterfacePtr());
            throw RuntimeError("Failed to create root signature: " + msg);
        }

        if (mSizeInBytes > sizeof(uint32_t) * D3D12_MAX_ROOT_COST)
        {
            throw RuntimeError("Root signature cost is too high. D3D12 root signatures are limited to 64 DWORDs, trying to create a signature with {} DWORDs.", mSizeInBytes / sizeof(uint32_t));
        }

        createApiHandle(pSigBlob);
    }

    D3D12RootSignature::~D3D12RootSignature()
    {
        sObjCount--;
        if (spEmptySig && sObjCount == 1) // That's right, 1. It means spEmptySig is the only object
        {
            spEmptySig = nullptr;
        }
    }

    D3D12RootSignature::SharedPtr D3D12RootSignature::getEmpty()
    {
        if (spEmptySig) return spEmptySig;
        return create(Desc());
    }

    D3D12RootSignature::SharedPtr D3D12RootSignature::create(const Desc& desc)
    {
        bool empty = desc.mSets.empty() && desc.mRootDescriptors.empty() && desc.mRootConstants.empty();
        if (empty && spEmptySig) return spEmptySig;

        SharedPtr pSig = SharedPtr(new D3D12RootSignature(desc));
        if (empty) spEmptySig = pSig;

        return pSig;
    }

    ReflectionResourceType::ShaderAccess getRequiredShaderAccess(D3D12RootSignature::DescType type)
    {
        switch (type)
        {
        case D3D12RootSignature::DescType::TextureSrv:
        case D3D12RootSignature::DescType::RawBufferSrv:
        case D3D12RootSignature::DescType::TypedBufferSrv:
        case D3D12RootSignature::DescType::StructuredBufferSrv:
        case D3D12RootSignature::DescType::AccelerationStructureSrv:
        case D3D12RootSignature::DescType::Cbv:
        case D3D12RootSignature::DescType::Sampler:
            return ReflectionResourceType::ShaderAccess::Read;
        case D3D12RootSignature::DescType::TextureUav:
        case D3D12RootSignature::DescType::RawBufferUav:
        case D3D12RootSignature::DescType::TypedBufferUav:
        case D3D12RootSignature::DescType::StructuredBufferUav:
            return ReflectionResourceType::ShaderAccess::ReadWrite;
        default:
            FALCOR_UNREACHABLE();
            return ReflectionResourceType::ShaderAccess(-1);
        }
    }

    // Add the descriptor set layouts from `pBlock` to the list
    // of descriptor set layouts being built for a root signature.
    //
    static void addParamBlockSets(
        const ParameterBlockReflection* pBlock,
        D3D12RootSignature::Desc& ioDesc)
    {
        auto defaultConstantBufferInfo = pBlock->getDefaultConstantBufferBindingInfo();
        if (defaultConstantBufferInfo.useRootConstants)
        {
            uint32_t count = uint32_t(pBlock->getElementType()->getByteSize() / sizeof(uint32_t));
            ioDesc.addRootConstants(defaultConstantBufferInfo.regIndex, defaultConstantBufferInfo.regSpace, count);
        }

        uint32_t setCount = pBlock->getD3D12DescriptorSetCount();
        for (uint32_t s = 0; s < setCount; ++s)
        {
            auto& setLayout = pBlock->getD3D12DescriptorSetLayout(s);
            ioDesc.addDescriptorSet(setLayout);
        }

        uint32_t parameterBlockRangeCount = pBlock->getParameterBlockSubObjectRangeCount();
        for (uint32_t i = 0; i < parameterBlockRangeCount; ++i)
        {
            uint32_t rangeIndex = pBlock->getParameterBlockSubObjectRangeIndex(i);

            auto& resourceRange = pBlock->getResourceRange(rangeIndex);
            auto& bindingInfo = pBlock->getResourceRangeBindingInfo(rangeIndex);
            FALCOR_ASSERT(bindingInfo.flavor == ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ParameterBlock);
            FALCOR_ASSERT(resourceRange.count == 1); // TODO: The code here does not handle arrays of sub-objects

            addParamBlockSets(bindingInfo.pSubObjectReflector.get(), ioDesc);
        }
    }

    // Add the root descriptors from `pBlock` to the root signature being built.
    static void addRootDescriptors(
        const ParameterBlockReflection* pBlock,
        D3D12RootSignature::Desc& ioDesc)
    {
        uint32_t rootDescriptorRangeCount = pBlock->getRootDescriptorRangeCount();
        for (uint32_t i = 0; i < rootDescriptorRangeCount; ++i)
        {
            uint32_t rangeIndex = pBlock->getRootDescriptorRangeIndex(i);

            auto& resourceRange = pBlock->getResourceRange(rangeIndex);
            auto& bindingInfo = pBlock->getResourceRangeBindingInfo(rangeIndex);
            FALCOR_ASSERT(bindingInfo.isRootDescriptor());
            FALCOR_ASSERT(resourceRange.count == 1); // Root descriptors cannot be arrays

            ioDesc.addRootDescriptor(resourceRange.descriptorType, bindingInfo.regIndex, bindingInfo.regSpace); // TODO: Using shader visibility *all* for now, get this info from reflection if we have it.
        }

        // Iterate over constant buffers and parameter blocks to recursively add their root descriptors.
        uint32_t resourceRangeCount = pBlock->getResourceRangeCount();
        for (uint32_t rangeIndex = 0; rangeIndex < resourceRangeCount; ++rangeIndex)
        {
            auto& bindingInfo = pBlock->getResourceRangeBindingInfo(rangeIndex);

            if (bindingInfo.flavor != ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ConstantBuffer &&
                bindingInfo.flavor != ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ParameterBlock)
                continue;

            auto& resourceRange = pBlock->getResourceRange(rangeIndex);
            FALCOR_ASSERT(resourceRange.count == 1); // TODO: The code here does not handle arrays of sub-objects

            addRootDescriptors(bindingInfo.pSubObjectReflector.get(), ioDesc);
        }
    }

    D3D12RootSignature::SharedPtr D3D12RootSignature::create(const ProgramReflection* pReflector)
    {
        FALCOR_ASSERT(pReflector);
        D3D12RootSignature::Desc d;
        addParamBlockSets(pReflector->getDefaultParameterBlock().get(), d);
        addRootDescriptors(pReflector->getDefaultParameterBlock().get(), d);
        return D3D12RootSignature::create(d);
    }

    void D3D12RootSignature::createApiHandle(ID3DBlobPtr pSigBlob)
    {
        Device::ApiHandle pDevice = gpDevice->getApiHandle();
        FALCOR_D3D_CALL(pDevice->CreateRootSignature(0, pSigBlob->GetBufferPointer(), pSigBlob->GetBufferSize(), IID_PPV_ARGS(&mApiHandle)));
    }

    template<bool forGraphics>
    static void bindRootSigCommon(CopyContext* pCtx, const D3D12RootSignature::ApiHandle& rootSig)
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

    void D3D12RootSignature::bindForCompute(CopyContext* pCtx)
    {
        bindRootSigCommon<false>(pCtx, mApiHandle);
    }

    void D3D12RootSignature::bindForGraphics(CopyContext* pCtx)
    {
        bindRootSigCommon<true>(pCtx, mApiHandle);
    }
}
