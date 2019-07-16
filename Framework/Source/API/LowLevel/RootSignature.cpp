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
#include "API/LowLevel/RootSignature.h"
#include "Graphics/Program/ProgramReflection.h"

namespace Falcor
{
    RootSignature::SharedPtr RootSignature::spEmptySig;
    uint64_t RootSignature::sObjCount = 0;

    RootSignature::Desc& RootSignature::Desc::addDescriptorSet(const DescriptorSetLayout& setLayout)
    {
        assert(setLayout.getRangeCount());
        mSets.push_back(setLayout);
        return *this; 
    }

    RootSignature::RootSignature(const Desc& desc) : mDesc(desc)
    {
        sObjCount++;
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
        bool empty = desc.mSets.size() == 0;
        if (empty && spEmptySig) return spEmptySig;

        SharedPtr pSig = SharedPtr(new RootSignature(desc));
        if (pSig->apiInit() == false)
        {
            pSig = nullptr;
        }

        if (empty) spEmptySig = pSig;

        return pSig;
    }

    ReflectionResourceType::ShaderAccess getRequiredShaderAccess(RootSignature::DescType type)
    {
        switch (type)
        {
        case RootSignature::DescType::TextureSrv:
        case RootSignature::DescType::TypedBufferSrv:
        case RootSignature::DescType::StructuredBufferSrv:
        case RootSignature::DescType::Cbv:
        case RootSignature::DescType::Sampler:
            return ReflectionResourceType::ShaderAccess::Read;
        case RootSignature::DescType::TextureUav:
        case RootSignature::DescType::StructuredBufferUav:
        case RootSignature::DescType::TypedBufferUav:
            return ReflectionResourceType::ShaderAccess::ReadWrite;
        default:
            should_not_get_here();
            return ReflectionResourceType::ShaderAccess(-1);
        }
    }

    static void addParamBlockSets(const ParameterBlockReflection* pBlock, RootSignature::Desc& d)
    {
        const auto& setLayouts = pBlock->getDescriptorSetLayouts();
        for (const auto& s : setLayouts)
        {
            d.addDescriptorSet(s);
        }
    }

    RootSignature::Desc getRootDescFromReflector(const ProgramReflection* pReflector, bool isLocal)
    {
        RootSignature::Desc d;
        for (uint32_t i = 0; i < pReflector->getParameterBlockCount(); i++)
        {
            addParamBlockSets(pReflector->getParameterBlock(i).get(), d);
        }
        d.setLocal(isLocal);
#ifdef FALCOR_VK
        // Validate no more than one shader record and store its size in root signature for building SBT
        if (isLocal)
        {
            const auto& pBlock = pReflector->getDefaultParameterBlock();
            bool shaderRecordFound = false;
            for (const auto& r : pBlock->getResourceVec())
            {
                // The only thing allowed in Vulkan "local" root signature is a cbuffer with ShaderRecord qualifier
                if (is_set(pBlock->getResource(r.name)->getModifier(), ReflectionVar::Modifier::ShaderRecord))
                {
                    assert(r.pType->getType() == ReflectionResourceType::Type::ConstantBuffer);
                    assert(!shaderRecordFound); // only 1 shader record allowed

                    // The cbuffer is to be embedded in the shader table, hence the size here is size of the cbuffer
                    // struct, unlike in D3D12 it's the size taken by all the CBV/SRV handles
                    const auto& structType = r.pType->getStructType()->asStructType();
                    uint32_t structSize = static_cast<uint32_t>(structType->getSize());
                    d.setSize(structSize);

                    shaderRecordFound = true;
                }
            }
        }
#endif
        return d;
    }

    RootSignature::SharedPtr RootSignature::create(const ProgramReflection* pReflector, bool isLocal)
    {
        RootSignature::Desc d = getRootDescFromReflector(pReflector, isLocal);
        return RootSignature::create(d);
    }
}
