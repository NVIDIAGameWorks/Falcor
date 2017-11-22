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
#include "ProgramVars.h"
#include "API/Buffer.h"
#include "API/CopyContext.h"
#include "API/RenderContext.h"
#include "API/DescriptorSet.h"
#include "API/Device.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
    template<bool forGraphics>
    void bindConstantBuffers(CopyContext* pContext, const ProgramVars::ResourceMap<ConstantBuffer>& cbMap, const ProgramVars::RootSetVec& rootSets, bool forceBind);
    ReflectionResourceType::ShaderAccess getRequiredShaderAccess(RootSignature::DescType type);

    bool verifyResourceVar(const ReflectionVar* pVar, ReflectionResourceType::Type type, ReflectionResourceType::ShaderAccess access, bool expectBuffer, const std::string& varName, const std::string& funcName)
    {
        if (pVar == nullptr)
        {
            logWarning(to_string(type) + " \"" + varName + "\" was not found. Ignoring " + funcName + " call.");
            return false;
        }
        const ReflectionResourceType* pType = pVar->getType()->unwrapArray()->asResourceType();
        if (!pType)
        {
            logWarning(varName + " is not a resource. Ignoring " + funcName + " call.");
            return false;
        }
#if _LOG_ENABLED
        if (pType->getType() != type)
        {
            logWarning("ProgramVars::" + funcName + " was called, but variable \"" + varName + "\" has different resource type. Expecting + " + to_string(pType->getType()) + " but provided resource is " + to_string(type) + ". Ignoring call");
            return false;
        }

        if (expectBuffer && pType->getDimensions() != ReflectionResourceType::Dimensions::Buffer)
        {
            logWarning("ProgramVars::" + funcName + " was called expecting a buffer variable, but the variable \"" + varName + "\" is not a buffer. Ignoring call");
            return false;
        }

        if (access != ReflectionResourceType::ShaderAccess::Undefined && pType->getShaderAccess() != access)
        {
            logWarning("ProgramVars::" + funcName + " was called, but variable \"" + varName + "\" has different shader access type. Expecting + " + to_string(pType->getShaderAccess()) + " but provided resource is " + to_string(access) + ". Ignoring call");
            return false;
        }
#endif
        return true;
    }

    ProgramVars::RootData findRootData(const RootSignature* pRootSig, uint32_t regIndex, uint32_t regSpace, RootSignature::DescType descType)
    {
        // Search the descriptor-tables
        for (size_t i = 0; i < pRootSig->getDescriptorSetCount(); i++)
        {
            const RootSignature::DescriptorSetLayout& set = pRootSig->getDescriptorSet(i);
            for(uint32_t r = 0 ; r < set.getRangeCount() ; r++)
            {
                const RootSignature::DescriptorSetLayout::Range& range = set.getRange(r);
                if (range.type == descType && range.regSpace == regSpace)
                {
                    if (range.baseRegIndex == regIndex)
                    {
                        return ProgramVars::RootData((uint32_t)i, r);
                    }
                }
            }
        }
        should_not_get_here();
        return ProgramVars::RootData();
    }

    template<typename BufferType, typename ViewType, RootSignature::DescType descType, typename ViewInitFunc>
    void addBuffers(const ParameterBlockReflection::ResourceDesc& desc, const ParameterBlockReflection* pBlockReflection, ProgramVars::ResourceMap<ViewType>& bufferMap, bool createBuffers, const ViewInitFunc& viewInitFunc, const RootSignature* pRootSig)
    {
        uint32_t regIndex = desc.regIndex;
        uint32_t regSpace = desc.regSpace;
        uint32_t arraySize = desc.descCount;
        ProgramVars::ResourceData<ViewType> data(findRootData(pRootSig, regIndex, regSpace, descType));
        const ReflectionResourceType::SharedConstPtr pType = pBlockReflection->getResource(desc.name)->getType()->unwrapArray()->asResourceType()->inherit_shared_from_this::shared_from_this();

        if (data.rootData.rootIndex == -1)
        {
            logError("Can't find a root-signature information matching buffer '" + desc.name + " when creating ProgramVars");
            return;
        }

        for (uint32_t a = 0; a < arraySize; a++)
        {
            // Only create the buffer if needed
            if (createBuffers)
            {
                data.pResource = BufferType::create(desc.name, pType);
                data.pView = viewInitFunc(data.pResource);
            }

            bufferMap[ProgramVars::BindLocation(regSpace, regIndex)].push_back(data);
        }
    }

    static RootSignature::DescType getRootDescTypeFromResourceType(ReflectionResourceType::Type type, ReflectionResourceType::ShaderAccess access)
    {
        switch (type)
        {
        case ReflectionResourceType::Type::Sampler:
            return RootSignature::DescType::Sampler;
        case ReflectionResourceType::Type::Texture:
        case ReflectionResourceType::Type::RawBuffer:  // Vulkan doesn't have raw-buffer and DX doesn't care
            return (access == ReflectionResourceType::ShaderAccess::Read) ? RootSignature::DescType::TextureSrv : RootSignature::DescType::TextureUav;
        case ReflectionResourceType::Type::StructuredBuffer:
            return (access == ReflectionResourceType::ShaderAccess::Read) ? RootSignature::DescType::StructuredBufferSrv : RootSignature::DescType::StructuredBufferUav;
        case ReflectionResourceType::Type::TypedBuffer:
            return (access == ReflectionResourceType::ShaderAccess::Read) ? RootSignature::DescType::TypedBufferSrv : RootSignature::DescType::TypedBufferUav;
        case ReflectionResourceType::Type::ConstantBuffer:
            return RootSignature::DescType::Cbv;
        default:
            should_not_get_here();
            return RootSignature::DescType(-1);
        }
    }

    ProgramVars::ProgramVars(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig) : mpReflector(pReflector)
    {
        // Initialize the CB and StructuredBuffer maps. We always do it, to mark which slots are used in the shader.
        mpRootSignature = pRootSig ? pRootSig : RootSignature::create(pReflector.get());

        auto getNullPtrFunc = [](const Resource::SharedPtr& pResource) { return nullptr; };
        auto getSrvFunc = [](const Resource::SharedPtr& pResource) { return pResource->getSRV(0, 1, 0, 1); };
        auto getUavFunc = [](const Resource::SharedPtr& pResource) { return pResource->getUAV(0, 0, 1); };

        ParameterBlockReflection::SharedConstPtr pGlobalBlock = pReflector->getParameterBlock("");

        // Initialize the textures and samplers map
        for (const auto& res : pGlobalBlock->getResources())
        {
            uint32_t count = res.descCount;
            for (uint32_t index = 0; index < (0 + count); ++index) // #PARAMBLOCK the 0 used to be descOffset
            {
                uint32_t regIndex = res.regIndex;
                uint32_t regSpace = res.regSpace;
                BindLocation loc(regSpace, regIndex);
                ProgramVars::RootData rootData = findRootData(mpRootSignature.get(), regIndex, regSpace, res.type);

                switch (res.type)
                {
                case ParameterBlockReflection::ResourceDesc::Type::Sampler:
                    mAssignedSamplers[loc].push_back(rootData);
                    break;
                case ParameterBlockReflection::ResourceDesc::Type::TextureSrv:
                case ParameterBlockReflection::ResourceDesc::Type::TypedBufferSrv:
                    assert(mAssignedSrvs.find(loc) == mAssignedSrvs.end() || mAssignedSrvs[loc].size() == index);
                    mAssignedSrvs[loc].push_back(rootData);
                    break;
                case ParameterBlockReflection::ResourceDesc::Type::TextureUav:
                case ParameterBlockReflection::ResourceDesc::Type::TypedBufferUav:
                    assert(mAssignedUavs.find(loc) == mAssignedUavs.end() || mAssignedUavs[loc].size() == index);
                    mAssignedUavs[loc].push_back(rootData);
                    break;
                case ParameterBlockReflection::ResourceDesc::Type::Cbv:
                    addBuffers<ConstantBuffer, ConstantBuffer, RootSignature::DescType::Cbv>(res, pGlobalBlock.get(), mAssignedCbs, createBuffers, getNullPtrFunc, mpRootSignature.get());
                    break;
                case ParameterBlockReflection::ResourceDesc::Type::StructuredBufferSrv:
                    addBuffers<StructuredBuffer, ShaderResourceView, RootSignature::DescType::StructuredBufferSrv>(res, pGlobalBlock.get(), mAssignedSrvs, createBuffers, getSrvFunc, mpRootSignature.get());
                    break;
                case ParameterBlockReflection::ResourceDesc::Type::StructuredBufferUav:
                    addBuffers<StructuredBuffer, UnorderedAccessView, RootSignature::DescType::StructuredBufferUav>(res, pGlobalBlock.get(), mAssignedUavs, createBuffers, getUavFunc, mpRootSignature.get());
                    break;
                default:
                    should_not_get_here();
                }
            }
        }

        mRootSets = RootSetVec(mpRootSignature->getDescriptorSetCount());

        // Mark the active descs (not empty, not CBs)
        for (size_t i = 0; i < mpRootSignature->getDescriptorSetCount(); i++)
        {
            const auto& set = mpRootSignature->getDescriptorSet(i);
#ifdef FALCOR_D3D12
            mRootSets[i].active = (set.getRangeCount() >= 1 && set.getRange(0).type != RootSignature::DescType::Cbv);
#else
            mRootSets[i].active = true;
#endif
        }
    }


    GraphicsVars::SharedPtr GraphicsVars::create(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig)
    {
        return SharedPtr(new GraphicsVars(pReflector, createBuffers, pRootSig));
    }

    ComputeVars::SharedPtr ComputeVars::create(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig)
    {
        return SharedPtr(new ComputeVars(pReflector, createBuffers, pRootSig));
    }

    static const ProgramReflection::ResourceBinding getBufferBindLocation(const ProgramReflection* pReflector, const std::string& name, uint32_t& arrayIndex, ReflectionResourceType::Type bufferType)
    {
        // #PARAMBLOCK handle non-global blocks
        const ReflectionVar* pVar = nullptr;
        pVar = pReflector->getParameterBlock("")->getResource(name).get();

        if (pVar == nullptr)
        {
            logWarning("Couldn't find a " + to_string(bufferType) + " named " + name);
            return ProgramReflection::ResourceBinding();
        }
        else if (pVar->getType()->unwrapArray()->asResourceType()->getType() != bufferType)
        {
            logWarning("Found a variable named '" + name + "' but it is not a " + to_string(bufferType));
            return ProgramReflection::ResourceBinding();
        }

        // #PARAMBLOCK Handle arrays
        arrayIndex = 0;
        return pReflector->getResourceBinding(name);
    }

    ConstantBuffer::SharedPtr ProgramVars::getConstantBuffer(const std::string& name) const
    {
        uint32_t arrayIndex;
        const auto& binding = getBufferBindLocation(mpReflector.get(), name, arrayIndex, ReflectionResourceType::Type::ConstantBuffer);
        if (binding.regSpace == ProgramReflection::kInvalidLocation)
        {
            logWarning("Constant buffer \"" + name + "\" was not found. Ignoring getConstantBuffer() call.");
            return nullptr;
        }
        return getConstantBuffer(binding.regSpace, binding.regIndex, arrayIndex);
    }

    ConstantBuffer::SharedPtr ProgramVars::getConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        auto& it = mAssignedCbs.find({ regSpace, baseRegIndex });
        if (it == mAssignedCbs.end())
        {
            logWarning("Can't find constant buffer at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace) + ". Ignoring getConstantBuffer() call.");
            return nullptr;
        }
        return std::static_pointer_cast<ConstantBuffer>(it->second[arrayIndex].pResource);
    }

    bool ProgramVars::setConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ConstantBuffer::SharedPtr& pCB)
    {
        BindLocation loc(regSpace, baseRegIndex);

        // Check that the index is valid
        if (mAssignedCbs.find(loc) == mAssignedCbs.end())
        {
            logWarning("No constant buffer was found at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace) + ". Ignoring setConstantBuffer() call.");
            return false;
        }

        // Just need to make sure the buffer is large enough
        // #PARAMBLOCK optimize this, maybe add a helper function to parameter block. Handle arrays
        const auto& cbs = mpReflector->getParameterBlock("")->getResources();
        for (const auto& desc : cbs)
        {
            if (desc.type != ParameterBlockReflection::ResourceDesc::Type::Cbv) continue;
            if (desc.regIndex == baseRegIndex && desc.regSpace == regSpace)
            {
                const auto& pVar = mpReflector->getParameterBlock("")->getResource(desc.name);
                if (pVar->getType()->asResourceType()->getSize() > pCB->getSize())
                {
                    logError("Can't attach the constant buffer. Size mismatch.");
                    return false;
                }
                break;
            }
        }

        mAssignedCbs[loc][arrayIndex].pResource = pCB;
        return true;
    }

    bool ProgramVars::setConstantBuffer(const std::string& name, const ConstantBuffer::SharedPtr& pCB)
    {
        // Find the buffer
       uint32_t arrayIndex;
        const auto loc = getBufferBindLocation(mpReflector.get(), name, arrayIndex, ReflectionResourceType::Type::ConstantBuffer);

        if (loc.regSpace == ProgramReflection::kInvalidLocation)
        {
            logWarning("Constant buffer \"" + name + "\" was not found. Ignoring setConstantBuffer() call.");
            return false;
        }
        return setConstantBuffer(loc.regSpace, loc.regIndex, arrayIndex, pCB);
        return false;
    }

    void setResourceSrvUavCommon(const ProgramVars::BindLocation& bindLoc,
        uint32_t descOffset, 
        ReflectionResourceType::ShaderAccess shaderAccess,
        const Resource::SharedPtr& resource, 
        ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, 
        ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs,
        std::vector<ProgramVars::RootSet>& rootSets)
    {
        switch (shaderAccess)
        {
        case ReflectionResourceType::ShaderAccess::ReadWrite:
        {
            auto uavIt = assignedUavs.find(bindLoc);
            assert(uavIt != assignedUavs.end());
            auto resUav = resource ? resource->getUAV() : nullptr;
            auto& data = uavIt->second[descOffset];
            if (data.pView != resUav)
            {
                rootSets[data.rootData.rootIndex].pDescSet = nullptr;
                data.pResource = resource;
                data.pView = resUav;
            }
            break;
        }

        case ReflectionResourceType::ShaderAccess::Read:
        {
            auto srvIt = assignedSrvs.find(bindLoc);
            assert(srvIt != assignedSrvs.end());

            auto resSrv = resource ? resource->getSRV() : nullptr;
            auto& data = srvIt->second[descOffset];

            if (data.pView != resSrv)
            {
                rootSets[data.rootData.rootIndex].pDescSet = nullptr;
                data.pResource = resource;
                data.pView = resSrv;
            }
            break;
        }

        default:
            should_not_get_here();
        }
    }

    void setResourceSrvUavCommon(const ReflectionVar* pVar, const Resource::SharedPtr& resource, ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs, std::vector<ProgramVars::RootSet>& rootSets)
    {
        auto shaderAccess = pVar->getType()->unwrapArray()->asResourceType()->getShaderAccess();
        setResourceSrvUavCommon({ pVar->getRegisterSpace(), pVar->getRegisterIndex() }, pVar->getDescOffset(), shaderAccess, resource, assignedSrvs, assignedUavs, rootSets);
    }

//     void setResourceSrvUavCommon(const ProgramReflection::BufferReflection *pDesc, uint32_t arrayIndex, const Resource::SharedPtr& resource, ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs, std::vector<ProgramVars::RootSet>& rootSets)
//     {
// //         setResourceSrvUavCommon({ pDesc->getRegisterSpace(), pDesc->getRegisterIndex() }, arrayIndex, pDesc->getShaderAccess(), resource, assignedSrvs, assignedUavs, rootSets);
//     }

    bool ProgramVars::setRawBuffer(const std::string& name, Buffer::SharedPtr pBuf)
    {
        // Find the buffer
        const ParameterBlockReflection* pGlobalBlock = mpReflector->getParameterBlock("").get();
        const ReflectionVar::SharedConstPtr pVar = pGlobalBlock->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::RawBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "setRawBuffer()") == false)
        {
            return false;
        }

        setResourceSrvUavCommon(pVar.get(), pBuf, mAssignedSrvs, mAssignedUavs, mRootSets);

        return true;
    }

    bool ProgramVars::setTypedBuffer(const std::string& name, TypedBufferBase::SharedPtr pBuf)
    {
        // Find the buffer
        const ParameterBlockReflection* pGlobalBlock = mpReflector->getParameterBlock("").get();
        const ReflectionVar::SharedConstPtr pVar = pGlobalBlock->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::TypedBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "setTypedBuffer()") == false)
        {
            return false;
        }

        setResourceSrvUavCommon(pVar.get(), pBuf, mAssignedSrvs, mAssignedUavs, mRootSets);
 
        return true;
    }

    static const ReflectionVar::SharedConstPtr getStructuredBufferReflection(const ProgramReflection* pReflector, const std::string& name, uint32_t& arrayIndex, const std::string& callStr)
    {
        arrayIndex = 0;
        const ParameterBlockReflection* pGlobalBlock = pReflector->getParameterBlock("").get();
        ReflectionVar::SharedConstPtr pVar;
        
        std::string noArray;
        if (parseArrayIndex(name, noArray, arrayIndex))
        {
            // #PARAMBLOCK nope, handling of the array index should happen elsewhere
            pVar = pGlobalBlock->getResource(noArray);
        }
        else
        {
            pVar = pGlobalBlock->getResource(name);
        }

        if (pVar == nullptr)
        {
            logWarning("Structured buffer \"" + name + "\" was not found. Ignoring " + callStr + "StructuredBuffer() call.");
            return false;
        }

        if (pVar->getType()->unwrapArray()->asResourceType()->getType() != ReflectionResourceType::Type::StructuredBuffer)
        {
            logWarning("Variable '" + name + "' is not a structured-buffer. Ignoring " + callStr + "StructuredBuffer() call.");
            return false;
        }

        return pVar;
    }

    bool ProgramVars::setStructuredBuffer(const std::string& name, StructuredBuffer::SharedPtr pBuf)
    {
        uint32_t arrayIndex;
        const ReflectionVar::SharedConstPtr pVar = getStructuredBufferReflection(mpReflector.get(), name, arrayIndex, "set");
        if (!pVar) return false;
        setResourceSrvUavCommon(pVar.get(), pBuf, mAssignedSrvs, mAssignedUavs, mRootSets);
        return true;
    }

    template<typename ResourceType>
    typename ResourceType::SharedPtr getResourceFromSrvUavCommon(uint32_t regSpace, uint32_t regIndex, uint32_t arrayIndex, ReflectionResourceType::ShaderAccess shaderAccess, const ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, const ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs, const std::string& varName, const std::string& funcName)
    {
        ProgramVars::BindLocation bindLoc(regSpace, regIndex);
        switch (shaderAccess)
        {
        case ReflectionResourceType::ShaderAccess::ReadWrite:
            if (assignedUavs.find(bindLoc) == assignedUavs.end())
            {
                logWarning("ProgramVars::" + funcName + " - variable \"" + varName + "\' was not found in UAVs. Shader Access = " + to_string(shaderAccess));
                return nullptr;
            }
            return std::dynamic_pointer_cast<ResourceType>(assignedUavs.at(bindLoc)[arrayIndex].pResource);

        case ReflectionResourceType::ShaderAccess::Read:
            if (assignedSrvs.find(bindLoc) == assignedSrvs.end())
            {
                logWarning("ProgramVars::" + funcName + " - variable \"" + varName + "\' was not found in SRVs. Shader Access = " + to_string(shaderAccess));
                return nullptr;
            }
            return std::dynamic_pointer_cast<ResourceType>(assignedSrvs.at(bindLoc)[arrayIndex].pResource);

        default:
            should_not_get_here();
        }

        return nullptr;
    }

    template<typename ResourceType>
    typename ResourceType::SharedPtr getResourceFromSrvUavCommon(const ReflectionVar* pVar, const ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, const ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs, const std::string& varName, const std::string& funcName)
    {
        return getResourceFromSrvUavCommon<ResourceType>(pVar->getRegisterSpace(), pVar->getRegisterIndex(), pVar->getDescOffset(), pVar->getType()->asResourceType()->getShaderAccess(), assignedSrvs, assignedUavs, varName, funcName);
    }
    
    Buffer::SharedPtr ProgramVars::getRawBuffer(const std::string& name) const
    {
        // Find the buffer
        const ParameterBlockReflection* pGlobalBlock = mpReflector->getParameterBlock("").get();
        const ReflectionVar::SharedConstPtr pVar = pGlobalBlock->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::RawBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "getRawBuffer()") == false)
        {
            return false;
        }

        return getResourceFromSrvUavCommon<Buffer>(pVar.get(), mAssignedSrvs, mAssignedUavs, name, "getRawBuffer()");
    }

    TypedBufferBase::SharedPtr ProgramVars::getTypedBuffer(const std::string& name) const
    {
        // Find the buffer
        const ParameterBlockReflection* pGlobalBlock = mpReflector->getParameterBlock("").get();
        const ReflectionVar::SharedConstPtr pVar = pGlobalBlock->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::TypedBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "getTypedBuffer()") == false)

        {
            return false;
        }

        return getResourceFromSrvUavCommon<TypedBufferBase>(pVar.get(), mAssignedSrvs, mAssignedUavs, name, "getTypedBuffer()");
        return nullptr;
    }

    StructuredBuffer::SharedPtr ProgramVars::getStructuredBuffer(const std::string& name) const
    {
        uint32_t arrayIndex;
        const ReflectionVar::SharedConstPtr pVar = getStructuredBufferReflection(mpReflector.get(), name, arrayIndex, "get");

        if (pVar == nullptr) return nullptr;
        return getResourceFromSrvUavCommon<StructuredBuffer>(pVar.get(), mAssignedSrvs, mAssignedUavs, name, "getStructuredBuffer()");
        return nullptr;
    }

    bool ProgramVars::setSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const Sampler::SharedPtr& pSampler)
    {
        auto& it = mAssignedSamplers.at({regSpace, baseRegIndex})[arrayIndex];
        if (it.pSampler != pSampler)
        {
            it.pSampler = pSampler;
            mRootSets[it.rootData.rootIndex].pDescSet = nullptr;
        }
        return true;
    }

    bool ProgramVars::setSampler(const std::string& name, const Sampler::SharedPtr& pSampler)
    {
        const ParameterBlockReflection* pGlobalBlock = mpReflector->getParameterBlock("").get();
        const ReflectionVar::SharedConstPtr pVar = pGlobalBlock->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Sampler, ReflectionResourceType::ShaderAccess::Read, false, name, "setSampler()") == false)
        {
            return false;
        }

        return setSampler(pVar->getRegisterSpace(), pVar->getRegisterIndex(), pVar->getDescOffset(), pSampler);
    }

    Sampler::SharedPtr ProgramVars::getSampler(const std::string& name) const
    {
        const ParameterBlockReflection* pGlobalBlock = mpReflector->getParameterBlock("").get();
        const ReflectionVar::SharedConstPtr pVar = pGlobalBlock->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Sampler, ReflectionResourceType::ShaderAccess::Read, false, name, "getSampler()") == false)
        {
            return nullptr;
        }
        // #PARAMBLOCK handle descOffset
        return getSampler(pVar->getRegisterSpace(), pVar->getRegisterIndex(), pVar->getDescOffset());
    }

    Sampler::SharedPtr ProgramVars::getSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        auto it = mAssignedSamplers.find({regSpace, baseRegIndex});
        if (it == mAssignedSamplers.end())
        {
            logWarning("ProgramVars::getSampler() - Cannot find sampler at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace));
            return nullptr;
        }

        return it->second[arrayIndex].pSampler;
        return nullptr;
    }

    ShaderResourceView::SharedPtr ProgramVars::getSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        auto it = mAssignedSrvs.find({ regSpace, baseRegIndex});
        if (it == mAssignedSrvs.end())
        {
            logWarning("ProgramVars::getSrv() - Cannot find SRV at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace));
            return nullptr;
        }

        return it->second[arrayIndex].pView;
    }

    UnorderedAccessView::SharedPtr ProgramVars::getUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        auto it = mAssignedUavs.find({ regSpace, baseRegIndex});
        if (it == mAssignedUavs.end())
        {
            logWarning("ProgramVars::getUav() - Cannot find UAV at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace));
            return nullptr;
        }

        return it->second[arrayIndex].pView;
    }

    bool ProgramVars::setTexture(const std::string& name, const Texture::SharedPtr& pTexture)
    {
       const ParameterBlockReflection* pGlobalBlock = mpReflector->getParameterBlock("").get();
       const ReflectionVar::SharedConstPtr pVar = pGlobalBlock->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Texture, ReflectionResourceType::ShaderAccess::Undefined, false,  name, "setTexture()") == false)
        {
            return false;
        }

        setResourceSrvUavCommon(pVar.get(), pTexture, mAssignedSrvs, mAssignedUavs, mRootSets);

        return true;
    }

    Texture::SharedPtr ProgramVars::getTexture(const std::string& name) const
    {
       const ParameterBlockReflection* pGlobalBlock = mpReflector->getParameterBlock("").get();
       const ReflectionVar::SharedConstPtr pVar = pGlobalBlock->getResource(name);

       if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Texture, ReflectionResourceType::ShaderAccess::Undefined, false, name, "getTexture()") == false)
        {
            return nullptr;
        }

        return getResourceFromSrvUavCommon<Texture>(pVar.get(), mAssignedSrvs, mAssignedUavs, name, "getTexture()");
    }

    template<typename ViewType>
    Resource::SharedPtr getResourceFromView(const ViewType* pView)
    {
        if (pView)
        {
            return const_cast<Resource*>(pView->getResource())->shared_from_this();
        }
        else
        {
            return nullptr;
        }
    }

    bool ProgramVars::setSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ShaderResourceView::SharedPtr& pSrv)
    {
        auto it = mAssignedSrvs.find({ regSpace, baseRegIndex });
        if (it != mAssignedSrvs.end())
        {
            auto& data = it->second[arrayIndex];
            if (data.pView != pSrv)
            {
                mRootSets[data.rootData.rootIndex].pDescSet = nullptr;
                data.pView = pSrv;
                data.pResource = getResourceFromView(pSrv.get()); // TODO: Fix resource/view const-ness so we don't need to do this
            }
        }
        else
        {
            logWarning("Can't find SRV with index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace) + ". Ignoring call to ProgramVars::setSrv()");
            return false;
        }

        return true;
    }

    bool ProgramVars::setUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const UnorderedAccessView::SharedPtr& pUav)
    {
        auto it = mAssignedUavs.find({ regSpace, baseRegIndex });
        if (it != mAssignedUavs.end())
        {
            auto& data = it->second[arrayIndex];
            if (data.pView != pUav)
            {
                mRootSets[data.rootData.rootIndex].pDescSet = nullptr;
                data.pView = pUav;
                data.pResource = getResourceFromView(pUav.get()); // TODO: Fix resource/view const-ness so we don't need to do this
            }
        }
        else
        {
            logWarning("Can't find UAV with index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace) + ". Ignoring call to ProgramVars::setUav()");
            return false;
        }

        return true;
    }

    void bindSamplers(const ProgramVars::ResourceMap<Sampler>& samplers, const ProgramVars::RootSetVec& rootSets)
    {
        // Bind the samplers
        for (auto& samplerIt : samplers)
        {
            const auto& samplerVec = samplerIt.second;
            const auto& rootData = samplerVec[0].rootData;
            if (rootSets[rootData.rootIndex].dirty)
            {
                for(uint32_t i = 0 ; i < samplerVec.size() ; i++)
                {
                    const Sampler* pSampler = samplerVec[i].pSampler.get();
                    if (pSampler == nullptr)
                    {
                        pSampler = Sampler::getDefault().get();
                    }
                    // Allocate a GPU descriptor
                    const auto& pDescSet = rootSets[rootData.rootIndex].pDescSet;
                    assert(pDescSet);
                    pDescSet->setSampler(rootData.rangeIndex, i, pSampler);
                }
            }
        }
    }

    template<typename ViewType, bool isUav, bool forGraphics>
    void bindUavSrvCommon(CopyContext* pContext, const ProgramVars::ResourceMap<ViewType>& resMap, const ProgramVars::RootSetVec& rootSets)
    {
        for (auto& resIt : resMap)
        {
            const auto& resVec = resIt.second;
            auto& rootData = resVec[0].rootData;

            if (rootSets[rootData.rootIndex].dirty)
            {
                for (uint32_t i = 0; i < resVec.size(); i++)
                {
                    auto& resDesc = resVec[i];
                    Resource* pResource = resDesc.pResource.get();
                    ViewType::SharedPtr view = pResource ? resDesc.pView : ViewType::getNullView();

                    // Get the set and copy the GPU handle
                    const auto& pDescSet = rootSets[rootData.rootIndex].pDescSet;
                    if (isUav)
                    {
                        pDescSet->setUav(rootData.rangeIndex, i, (UnorderedAccessView*)view.get());
                    }
                    else
                    {
                        pDescSet->setSrv(rootData.rangeIndex, i, (ShaderResourceView*)view.get());
                    }
                }
            }
        }
    }

    void uploadConstantBuffers(const ProgramVars::ResourceMap<ConstantBuffer>& cbMap, ProgramVars::RootSetVec& rootSets)
    {
        for (auto& bufIt : cbMap)
        {
            assert(bufIt.second.size() == 1);
            const auto& rootData = bufIt.second[0].rootData;
            ConstantBuffer* pCB = dynamic_cast<ConstantBuffer*>(bufIt.second[0].pResource.get());

            if (pCB && pCB->uploadToGPU())
            {
                rootSets[rootData.rootIndex].pDescSet = nullptr;
            }
        }
    }

    template<typename ViewType, bool isUav>
    void uploadUavSrvCommon(CopyContext* pContext, const ProgramVars::ResourceMap<ViewType>& resMap, ProgramVars::RootSetVec& rootSets)
    {
        for (auto& resIt : resMap)
        {
            const auto& resVec = resIt.second;
            auto& rootData = resVec[0].rootData;

            for(const auto& resDesc : resVec)
            {
                Resource* pResource = resDesc.pResource.get();
                if (pResource)
                {
                    bool invalidate = false;
                    // If it's a typed buffer, upload it to the GPU
                    TypedBufferBase* pTypedBuffer = dynamic_cast<TypedBufferBase*>(pResource);
                    if (pTypedBuffer)
                    {
                        invalidate = pTypedBuffer->uploadToGPU();
                    }
                    StructuredBuffer* pStructured = dynamic_cast<StructuredBuffer*>(pResource);
                    if (pStructured)
                    {
                        invalidate = pStructured->uploadToGPU();

                        if (isUav && pStructured->hasUAVCounter())
                        {
                            pContext->resourceBarrier(pStructured->getUAVCounter().get(), Resource::State::UnorderedAccess);
                        }
                    }

                    pContext->resourceBarrier(resDesc.pResource.get(), isUav ? Resource::State::UnorderedAccess : Resource::State::ShaderResource);
                    if (isUav)
                    {
                        if (pTypedBuffer) pTypedBuffer->setGpuCopyDirty();
                        if (pStructured)  pStructured->setGpuCopyDirty();
                    }
                    if (invalidate) rootSets[rootData.rootIndex].pDescSet = nullptr;
                }
            }
        }
    }

    template<bool forGraphics>
    bool applyProgramVarsCommon(const ProgramVars* pVars, ProgramVars::RootSetVec& rootSets, CopyContext* pContext, bool bindRootSig)
    {
        if (bindRootSig)
        {
            if (forGraphics)
            {
                pVars->getRootSignature()->bindForGraphics(pContext);
            }
            else
            {
                pVars->getRootSignature()->bindForCompute(pContext);
            }
        }

        // Upload the resources. This will also invalidate descriptor-sets that contain dynamic resources
        uploadConstantBuffers(pVars->getAssignedCbs(), rootSets);
        uploadUavSrvCommon<ShaderResourceView, false>(pContext, pVars->getAssignedSrvs(), rootSets);
        uploadUavSrvCommon<UnorderedAccessView, true>(pContext, pVars->getAssignedUavs(), rootSets);

        // Allocate and mark the dirty sets
        for (uint32_t i = 0; i < rootSets.size(); i++)
        {
            rootSets[i].dirty = (rootSets[i].pDescSet == nullptr);
            if (rootSets[i].active)
            {
                if (rootSets[i].pDescSet == nullptr)
                {
                    DescriptorSet::Layout layout;
                    const auto& set = pVars->getRootSignature()->getDescriptorSet(i);
                    rootSets[i].pDescSet = DescriptorSet::create(gpDevice->getGpuDescriptorPool(), set);
                    if (rootSets[i].pDescSet == nullptr)
                    {
                        return false;
                    }
                }
            }
        }
        
        bindUavSrvCommon<ShaderResourceView, false, forGraphics>(pContext, pVars->getAssignedSrvs(), rootSets);
        bindUavSrvCommon<UnorderedAccessView, true, forGraphics>(pContext, pVars->getAssignedUavs(), rootSets);
        bindSamplers(pVars->getAssignedSamplers(), rootSets);
        bindConstantBuffers<forGraphics>(pContext, pVars->getAssignedCbs(), rootSets, bindRootSig);

        // Bind the sets
        for (uint32_t i = 0; i < rootSets.size(); i++)
        {
            if (rootSets[i].active == false) continue;

            if (rootSets[i].dirty || bindRootSig)
            {
                rootSets[i].dirty = false;
                if (forGraphics)
                {
                    rootSets[i].pDescSet->bindForGraphics(pContext, pVars->getRootSignature().get(), i);
                }
                else
                {
                    rootSets[i].pDescSet->bindForCompute(pContext, pVars->getRootSignature().get(), i);
                }
            }
        }
        return true;
    }

    bool ComputeVars::apply(ComputeContext* pContext, bool bindRootSig)
    {
        return applyProgramVarsCommon<false>(this, mRootSets, pContext, bindRootSig);
    }

    bool GraphicsVars::apply(RenderContext* pContext, bool bindRootSig)
    {
        return applyProgramVarsCommon<true>(this, mRootSets, pContext, bindRootSig);
    }

    ProgramVars::BindLocation getResourceBindLocation(const ProgramReflection* pReflector, const std::string& name)
    {
        ProgramVars::BindLocation loc;
        const auto& desc = pReflector->getResourceBinding(name);
        loc.baseRegIndex = desc.regIndex;
        loc.regSpace = desc.regSpace;
        return loc;
    }

    ProgramVars::BindLocation getBufferBindLocation(const ProgramReflection* pReflector, const std::string& name)
    {
        ProgramVars::BindLocation loc;
        const auto& desc = pReflector->getResourceBinding(name);
        loc.baseRegIndex = desc.regIndex;
        loc.regSpace = desc.regSpace;
        return loc;
    }
}