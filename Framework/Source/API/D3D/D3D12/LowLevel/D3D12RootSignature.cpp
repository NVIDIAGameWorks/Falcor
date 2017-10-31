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
#include "API/D3D/D3DState.h"
#include "API/Device.h"

namespace Falcor
{
    using RootParameterVec = std::vector<D3D12_ROOT_PARAMETER>;

    D3D12_SHADER_VISIBILITY getShaderVisibility(ShaderVisibility visibility)
    {
        // D3D12 doesn't support a combination of flags, it's either ALL or a single stage
        if (isPowerOf2(visibility) == false)
        {
            return D3D12_SHADER_VISIBILITY_ALL;
        }
        else if ((visibility & ShaderVisibility::Vertex) != ShaderVisibility::None)
        {
            return D3D12_SHADER_VISIBILITY_VERTEX;
        }
        else if ((visibility & ShaderVisibility::Pixel) != ShaderVisibility::None)
        {
            return D3D12_SHADER_VISIBILITY_PIXEL;
        }
        else if ((visibility & ShaderVisibility::Geometry) != ShaderVisibility::None)
        {
            return D3D12_SHADER_VISIBILITY_GEOMETRY;
        }
        else if ((visibility & ShaderVisibility::Domain) != ShaderVisibility::None)
        {
            return D3D12_SHADER_VISIBILITY_DOMAIN;
        }
        else if ((visibility & ShaderVisibility::Hull) != ShaderVisibility::None)
        {
            return D3D12_SHADER_VISIBILITY_HULL;
        }
        // If it was compute, it can't be anything else and so the first `if` would have handled it
        should_not_get_here();
        return (D3D12_SHADER_VISIBILITY)-1;
    }

    D3D12_DESCRIPTOR_RANGE_TYPE getDescRangeType(RootSignature::DescType type)
    {
        switch (type)
        {
        case RootSignature::DescType::TextureSrv:
        case RootSignature::DescType::TypedBufferSrv:
        case RootSignature::DescType::StructuredBufferSrv:
            return D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        case RootSignature::DescType::TextureUav:
        case RootSignature::DescType::TypedBufferUav:
        case RootSignature::DescType::StructuredBufferUav:
            return D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        case RootSignature::DescType::Cbv:
            return D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
        case RootSignature::DescType::Sampler:
            return D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
        default:
            should_not_get_here();
            return (D3D12_DESCRIPTOR_RANGE_TYPE)-1;
        }
    }

    void convertCbvSet(const RootSignature::DescriptorSetLayout& set, D3D12_ROOT_PARAMETER& desc)
    {
        assert(set.getRangeCount() == 1);
        const auto& range = set.getRange(0);
        assert(range.type == RootSignature::DescType::Cbv && range.descCount == 1);

        desc.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        desc.Descriptor.RegisterSpace = range.regSpace;
        desc.Descriptor.ShaderRegister = range.baseRegIndex;
        desc.ShaderVisibility = getShaderVisibility(set.getVisibility());
    }

    void convertDescTable(const RootSignature::DescriptorSetLayout& falcorSet, D3D12_ROOT_PARAMETER& desc, std::vector<D3D12_DESCRIPTOR_RANGE>& d3dRange)
    {
        desc.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        desc.ShaderVisibility = getShaderVisibility(falcorSet.getVisibility());
        d3dRange.resize(falcorSet.getRangeCount());
        desc.DescriptorTable.NumDescriptorRanges = (uint32_t)falcorSet.getRangeCount();
        desc.DescriptorTable.pDescriptorRanges = d3dRange.data();

        for (size_t i = 0; i < falcorSet.getRangeCount(); i++)
        {
            const auto& falcorRange = falcorSet.getRange(i);
            d3dRange[i].BaseShaderRegister = falcorRange.baseRegIndex;
            d3dRange[i].NumDescriptors = falcorRange.descCount;
            d3dRange[i].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
            d3dRange[i].RangeType = getDescRangeType(falcorRange.type);
            d3dRange[i].RegisterSpace = falcorRange.regSpace;
        }
    }

    bool RootSignature::apiInit()
    {
        mSizeInBytes = 0;
        RootParameterVec rootParams(mDesc.mSets.size());
        mElementByteOffset.resize(mDesc.mSets.size());

        // Descriptor sets. Need to allocate some space for the D3D12 tables
        std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> d3dRanges(mDesc.mSets.size());
        for (size_t i = 0 ; i < mDesc.mSets.size() ; i++)
        {
            const auto& set = mDesc.mSets[i];
            assert(set.getRangeCount() == 1);
            uint32_t byteOffset;
            if (set.getRangeCount() == 1 && set.getRange(0).type == DescType::Cbv)
            {
                convertCbvSet(set, rootParams[i]);
                byteOffset = 8;
            }
            else
            {
                convertDescTable(mDesc.mSets[i], rootParams[i], d3dRanges[i]);
                byteOffset = 4;
            }
            mElementByteOffset[i] = mSizeInBytes;
            mSizeInBytes += byteOffset;
        }

        // Create the root signature
        D3D12_ROOT_SIGNATURE_DESC desc;
        desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        desc.pParameters = rootParams.data();
        desc.NumParameters = (uint32_t)rootParams.size();
        desc.pStaticSamplers = nullptr;
        desc.NumStaticSamplers = 0;

        ID3DBlobPtr pSigBlob;
        ID3DBlobPtr pErrorBlob;
        HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &pSigBlob, &pErrorBlob);
        if (FAILED(hr))
        {
            std::string msg = convertBlobToString(pErrorBlob.GetInterfacePtr());
            logError(msg);
            return false;
        }

        if (mSizeInBytes > sizeof(uint32_t) * D3D12_MAX_ROOT_COST)
        {
            logError("Root-signature cost is too high. D3D12 root-signatures are limited to 64 DWORDs, trying to create a signature with " + std::to_string(mSizeInBytes / sizeof(uint32_t)) + " DWORDs");
            return false;
        }

        createApiHandle(pSigBlob);       
        return true;
    }

    void RootSignature::createApiHandle(ID3DBlobPtr pSigBlob)
    {
        Device::ApiHandle pDevice = gpDevice->getApiHandle();
        d3d_call(pDevice->CreateRootSignature(0, pSigBlob->GetBufferPointer(), pSigBlob->GetBufferSize(), IID_PPV_ARGS(&mApiHandle)));
    }

    ProgramReflection::ShaderAccess getRequiredShaderAccess(RootSignature::DescType type);

    static uint32_t initializeBufferDescriptors(const ProgramReflection* pReflector, RootSignature::Desc& desc, ProgramReflection::BufferReflection::Type bufferType, RootSignature::DescType descType)
    {
        uint32_t cost = 0;
        const auto& bufMap = pReflector->getBufferMap(bufferType);
        for (const auto& buf : bufMap)
        {
            const ProgramReflection::BufferReflection* pBuffer = buf.second.get();
            if (pBuffer->getShaderAccess() == getRequiredShaderAccess(descType))
            {
                RootSignature::DescriptorSetLayout descTable;
                uint32_t count = buf.second->getArraySize() ? buf.second->getArraySize() : 1;
                descTable.addRange(descType, pBuffer->getRegisterIndex(), count, pBuffer->getRegisterSpace());
                cost += 1;
                desc.addDescriptorSet(descTable);
            }
        }
        return cost;
    }

    uint32_t getRootDescFromReflector(const ProgramReflection* pReflector, RootSignature::Desc& d)
    {
        uint32_t cost = 0;
        d = RootSignature::Desc();

        cost += initializeBufferDescriptors(pReflector, d, ProgramReflection::BufferReflection::Type::Constant, RootSignature::DescType::Cbv);
        cost += initializeBufferDescriptors(pReflector, d, ProgramReflection::BufferReflection::Type::Structured, RootSignature::DescType::StructuredBufferSrv);
        cost += initializeBufferDescriptors(pReflector, d, ProgramReflection::BufferReflection::Type::Structured, RootSignature::DescType::StructuredBufferUav);

        const ProgramReflection::ResourceMap& resMap = pReflector->getResourceMap();
        for (auto& resIt : resMap)
        {
            const ProgramReflection::Resource& resource = resIt.second;
            assert(resource.descOffset == 0);
            RootSignature::DescType descType;
            if (resource.type == ProgramReflection::Resource::ResourceType::Sampler)
            {
                descType = RootSignature::DescType::Sampler;
            }
            else
            {
                switch (resource.type)
                {
                case ProgramReflection::Resource::ResourceType::RawBuffer:
                case ProgramReflection::Resource::ResourceType::Texture:
                    descType = (resource.shaderAccess == ProgramReflection::ShaderAccess::ReadWrite) ? RootSignature::DescType::TextureUav : RootSignature::DescType::TextureSrv;
                    break;
                case ProgramReflection::Resource::ResourceType::StructuredBuffer:
                    descType = (resource.shaderAccess == ProgramReflection::ShaderAccess::ReadWrite) ? RootSignature::DescType::StructuredBufferUav : RootSignature::DescType::StructuredBufferSrv;
                    break;
                case ProgramReflection::Resource::ResourceType::TypedBuffer:
                    descType = (resource.shaderAccess == ProgramReflection::ShaderAccess::ReadWrite) ? RootSignature::DescType::TypedBufferUav : RootSignature::DescType::TypedBufferSrv;
                    break;;
                default:
                    should_not_get_here();
                }
            }

            uint32_t count = resource.arraySize ? resource.arraySize : 1;
            RootSignature::DescriptorSetLayout descTable;
            descTable.addRange(descType, resource.regIndex, count, resource.regSpace);
            d.addDescriptorSet(descTable);
            cost += 1;
        }

        return cost;
    }

    RootSignature::SharedPtr RootSignature::create(const ProgramReflection* pReflector)
    {
        RootSignature::Desc d;
        uint32_t cost = getRootDescFromReflector(pReflector, d);

        if (cost > 64)
        {
            logError("RootSignature::create(): The required storage cost is " + std::to_string(cost) + " DWORDS, which is larger then the max allowed cost of 64 DWORDS");
            return nullptr;
        }
        return (cost != 0) ? RootSignature::create(d) : RootSignature::getEmpty();
    }

    template<bool forGraphics>
    static void bindRootSigCommon(CopyContext* pCtx, RootSignature::ApiHandle rootSig)
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
