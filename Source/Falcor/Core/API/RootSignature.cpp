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
#include "RootSignature.h"
#include "Core/Program/ProgramReflection.h"

namespace Falcor
{
    RootSignature::SharedPtr RootSignature::spEmptySig;
    uint64_t RootSignature::sObjCount = 0;

    RootSignature::Desc& RootSignature::Desc::addDescriptorSet(const DescriptorSetLayout& setLayout)
    {
        assert(mRootConstants.empty()); // For now we disallow both root-constants and descriptor-sets
        assert(setLayout.getRangeCount());
        mSets.push_back(setLayout);
        return *this; 
    }

    RootSignature::Desc& RootSignature::Desc::addRootDescriptor(DescType type, uint32_t regIndex, uint32_t spaceIndex, ShaderVisibility visibility)
    {
        RootDescriptorDesc desc;
        desc.type = type;
        desc.regIndex = regIndex;
        desc.spaceIndex = spaceIndex;
        desc.visibility = visibility;
        mRootDescriptors.push_back(desc);
        return *this;
    }

    RootSignature::Desc& RootSignature::Desc::addRootConstants(uint32_t regIndex, uint32_t spaceIndex, uint32_t count)
    {
        assert(mSets.empty()); // For now we disallow both root-constants and descriptor-sets
        RootConstantsDesc desc;
        desc.count = count;
        desc.regIndex = regIndex;
        desc.spaceIndex = spaceIndex;
        mRootConstants.push_back(desc);
        return *this;
    }

    RootSignature::RootSignature(const Desc& desc)
        : mDesc(desc)
    {
        sObjCount++;
        apiInit();
    }

    RootSignature::~RootSignature()
    {
        sObjCount--;
        if (spEmptySig && sObjCount == 1) // That's right, 1. It means spEmptySig is the only object
        {
            spEmptySig = nullptr;
        }
    }

    RootSignature::SharedPtr RootSignature::getEmpty()
    {
        if (spEmptySig) return spEmptySig;
        return create(Desc());
    }

    RootSignature::SharedPtr RootSignature::create(const Desc& desc)
    {
        bool empty = desc.mSets.empty() && desc.mRootDescriptors.empty() && desc.mRootConstants.empty();
        if (empty && spEmptySig) return spEmptySig;

        SharedPtr pSig = SharedPtr(new RootSignature(desc));
        if (empty) spEmptySig = pSig;

        return pSig;
    }

    ReflectionResourceType::ShaderAccess getRequiredShaderAccess(RootSignature::DescType type)
    {
        switch (type)
        {
        case RootSignature::DescType::TextureSrv:
        case RootSignature::DescType::RawBufferSrv:
        case RootSignature::DescType::TypedBufferSrv:
        case RootSignature::DescType::StructuredBufferSrv:
        case RootSignature::DescType::Cbv:
        case RootSignature::DescType::Sampler:
            return ReflectionResourceType::ShaderAccess::Read;
        case RootSignature::DescType::TextureUav:
        case RootSignature::DescType::RawBufferUav:
        case RootSignature::DescType::TypedBufferUav:
        case RootSignature::DescType::StructuredBufferUav:
            return ReflectionResourceType::ShaderAccess::ReadWrite;
        default:
            should_not_get_here();
            return ReflectionResourceType::ShaderAccess(-1);
        }
    }

    // Add the descriptor set layouts from `pBlock` to the list
    // of descriptor set layouts being built for a root signature.
    //
    static void addParamBlockSets(
        const ParameterBlockReflection*     pBlock,
        RootSignature::Desc&                ioDesc)
    {
        auto defaultConstantBufferInfo = pBlock->getDefaultConstantBufferBindingInfo();
        if( defaultConstantBufferInfo.useRootConstants )
        {
            uint32_t count = uint32_t(pBlock->getElementType()->getByteSize() / sizeof(uint32_t));
            ioDesc.addRootConstants(defaultConstantBufferInfo.regIndex, defaultConstantBufferInfo.regSpace, count);
        }

        uint32_t setCount = pBlock->getDescriptorSetCount();
        for (uint32_t s = 0; s < setCount; ++s)
        {
            auto& setLayout = pBlock->getDescriptorSetLayout(s);
            ioDesc.addDescriptorSet(setLayout);
        }

        uint32_t parameterBlockRangeCount = pBlock->getParameterBlockSubObjectRangeCount();
        for (uint32_t i = 0; i < parameterBlockRangeCount; ++i )
        {
            uint32_t rangeIndex = pBlock->getParameterBlockSubObjectRangeIndex(i);

            auto& resourceRange = pBlock->getResourceRange(rangeIndex);
            auto& bindingInfo = pBlock->getResourceRangeBindingInfo(rangeIndex);
            assert(bindingInfo.flavor == ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ParameterBlock);
            assert(resourceRange.count == 1); // TODO: The code here does not handle arrays of sub-objects

            addParamBlockSets(bindingInfo.pSubObjectReflector.get(), ioDesc);
        }
    }

    // Add the root descriptors from `pBlock` to the root signature being built.
    static void addRootDescriptors(
        const ParameterBlockReflection*     pBlock,
        RootSignature::Desc&                ioDesc)
    {
        uint32_t rootDescriptorRangeCount = pBlock->getRootDescriptorRangeCount();
        for (uint32_t i = 0; i < rootDescriptorRangeCount; ++i)
        {
            uint32_t rangeIndex = pBlock->getRootDescriptorRangeIndex(i);

            auto& resourceRange = pBlock->getResourceRange(rangeIndex);
            auto& bindingInfo = pBlock->getResourceRangeBindingInfo(rangeIndex);
            assert(bindingInfo.isRootDescriptor());
            assert(resourceRange.count == 1); // Root descriptors cannot be arrays

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
            assert(resourceRange.count == 1); // TODO: The code here does not handle arrays of sub-objects

            addRootDescriptors(bindingInfo.pSubObjectReflector.get(), ioDesc);
        }
    }

    RootSignature::SharedPtr RootSignature::create(const ProgramReflection* pReflector)
    {
        assert(pReflector);
        RootSignature::Desc d;
        addParamBlockSets(pReflector->getDefaultParameterBlock().get(), d);
        addRootDescriptors(pReflector->getDefaultParameterBlock().get(), d);
        return RootSignature::create(d);
    }

    RootSignature::SharedPtr RootSignature::createLocal(const EntryPointBaseReflection* pReflector)
    {
        assert(pReflector);
        RootSignature::Desc d;
        addParamBlockSets(pReflector, d);
        addRootDescriptors(pReflector, d);

#ifdef FALCOR_D3D12
        d.setLocal(true);
#else
        logWarning("Local root-signatures are only supported in D3D12 for use with DXR. Make sure you are using the correct build configuration.");
#endif

        return RootSignature::create(d);
    }
}
