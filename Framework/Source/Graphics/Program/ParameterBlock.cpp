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
#include "ParameterBlock.h"
#include "API/LowLevel/RootSignature.h"
#include "API/Device.h"

namespace Falcor
{
    void bindConstantBuffers(const ParameterBlock::ResourceMap<ConstantBuffer>& cbMap, const ParameterBlock::RootSetVec& rootSets);
    ReflectionResourceType::ShaderAccess getRequiredShaderAccess(RootSignature::DescType type);

    ParameterBlock::RootData findRootData(const RootSignature* pRootSig, uint32_t regIndex, uint32_t regSpace, RootSignature::DescType descType)
    {
        // Search the descriptor-tables
        for (size_t i = 0; i < pRootSig->getDescriptorSetCount(); i++)
        {
            const RootSignature::DescriptorSetLayout& set = pRootSig->getDescriptorSet(i);
            for (uint32_t r = 0; r < set.getRangeCount(); r++)
            {
                const RootSignature::DescriptorSetLayout::Range& range = set.getRange(r);
                if (range.type == descType && range.regSpace == regSpace)
                {
                    if (range.baseRegIndex == regIndex)
                    {
                        return ParameterBlock::RootData((uint32_t)i, r);
                    }
                }
            }
        }
        should_not_get_here();
        return ParameterBlock::RootData();
    }

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
            logWarning("ParameterBlock::" + funcName + " was called, but variable \"" + varName + "\" has different resource type. Expecting + " + to_string(pType->getType()) + " but provided resource is " + to_string(type) + ". Ignoring call");
            return false;
        }

        if (expectBuffer && pType->getDimensions() != ReflectionResourceType::Dimensions::Buffer)
        {
            logWarning("ParameterBlock::" + funcName + " was called expecting a buffer variable, but the variable \"" + varName + "\" is not a buffer. Ignoring call");
            return false;
        }

        if (access != ReflectionResourceType::ShaderAccess::Undefined && pType->getShaderAccess() != access)
        {
            logWarning("ParameterBlock::" + funcName + " was called, but variable \"" + varName + "\" has different shader access type. Expecting + " + to_string(pType->getShaderAccess()) + " but provided resource is " + to_string(access) + ". Ignoring call");
            return false;
        }
#endif
        return true;
    }

    template<typename BufferType, typename ViewType, RootSignature::DescType descType, typename ViewInitFunc>
    void addBuffer(const ParameterBlockReflection::ResourceDesc& desc, const ParameterBlockReflection* pBlockReflection, ParameterBlock::ResourceMap<ViewType>& bufferMap, bool createBuffers, const ViewInitFunc& viewInitFunc, const RootSignature* pRootSig)
    {
        uint32_t regIndex = desc.regIndex;
        uint32_t regSpace = desc.regSpace;
        ParameterBlock::ResourceData<ViewType> data(findRootData(pRootSig, regIndex, regSpace, descType));
        const ReflectionResourceType::SharedConstPtr pType = pBlockReflection->getResource(desc.name)->getType()->unwrapArray()->asResourceType()->inherit_shared_from_this::shared_from_this();

        if (data.rootData.rootIndex == -1)
        {
            logError("Can't find a root-signature information matching buffer '" + desc.name + " when creating ParameterBlock");
            return;
        }

        // Only create the buffer if needed
        if (createBuffers)
        {
            data.pResource = BufferType::create(desc.name, pType);
            data.pView = viewInitFunc(data.pResource);
        }

        bufferMap[ParameterBlock::BindLocation(regSpace, regIndex)].push_back(data);
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

    ParameterBlock::~ParameterBlock() = default;

    ParameterBlock::SharedPtr ParameterBlock::create(const ParameterBlockReflection::SharedConstPtr& pReflection, const RootSignature* pRootSig, bool createBuffers)
    {
        return SharedPtr(new ParameterBlock(pReflection, pRootSig, createBuffers));
    }

    ParameterBlock::ParameterBlock(const ParameterBlockReflection::SharedConstPtr& pReflection, const RootSignature* pRootSig, bool createBuffers) : mpReflector(pReflection)
    {
        auto getNullPtrFunc = [](const Resource::SharedPtr& pResource) { return nullptr; };
        auto getSrvFunc = [](const Resource::SharedPtr& pResource) { return pResource->getSRV(0, 1, 0, 1); };
        auto getUavFunc = [](const Resource::SharedPtr& pResource) { return pResource->getUAV(0, 0, 1); };

        // Initialize the textures and samplers map
        for (const auto& res : mpReflector->getResources())
        {
            for (uint32_t index = 0; index < res.descCount; ++index)
            {
                uint32_t regIndex = res.regIndex;
                uint32_t regSpace = res.regSpace;
                BindLocation loc(regSpace, regIndex);
                ParameterBlock::RootData rootData = findRootData(pRootSig, regIndex, regSpace, res.type);

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
                    addBuffer<ConstantBuffer, ConstantBuffer, RootSignature::DescType::Cbv>(res, mpReflector.get(), mAssignedCbs, createBuffers, getNullPtrFunc, pRootSig);
                    break;
                case ParameterBlockReflection::ResourceDesc::Type::StructuredBufferSrv:
                    addBuffer<StructuredBuffer, ShaderResourceView, RootSignature::DescType::StructuredBufferSrv>(res, mpReflector.get(), mAssignedSrvs, createBuffers, getSrvFunc, pRootSig);
                    break;
                case ParameterBlockReflection::ResourceDesc::Type::StructuredBufferUav:
                    addBuffer<StructuredBuffer, UnorderedAccessView, RootSignature::DescType::StructuredBufferUav>(res, mpReflector.get(), mAssignedUavs, createBuffers, getUavFunc, pRootSig);
                    break;
                default:
                    should_not_get_here();
                }
            }
        }

        mRootSets = RootSetVec(pRootSig->getDescriptorSetCount());
    }

    static const ProgramReflection::ResourceBinding getBufferBindLocation(const ParameterBlockReflection* pReflector, const std::string& name, uint32_t& arrayIndex, ReflectionResourceType::Type bufferType)
    {
        // #PARAMBLOCK handle non-global blocks
        const ReflectionVar* pVar = nullptr;
        pVar = pReflector->getResource(name).get();

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

    ConstantBuffer::SharedPtr ParameterBlock::getConstantBuffer(const std::string& name) const
    {
        uint32_t arrayIndex;
        const auto& binding = getBufferBindLocation(mpReflector.get(), name, arrayIndex, ReflectionResourceType::Type::ConstantBuffer);
        if (binding.regSpace == ParameterBlockReflection::kInvalidLocation)
        {
            logWarning("Constant buffer \"" + name + "\" was not found. Ignoring getConstantBuffer() call.");
            return nullptr;
        }
        return getConstantBuffer(binding.regSpace, binding.regIndex, arrayIndex);
    }

    ConstantBuffer::SharedPtr ParameterBlock::getConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        auto& it = mAssignedCbs.find({ regSpace, baseRegIndex });
        if (it == mAssignedCbs.end())
        {
            logWarning("Can't find constant buffer at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace) + ". Ignoring getConstantBuffer() call.");
            return nullptr;
        }
        return std::static_pointer_cast<ConstantBuffer>(it->second[arrayIndex].pResource);
    }

    bool ParameterBlock::setConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ConstantBuffer::SharedPtr& pCB)
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
        const auto& cbs = mpReflector->getResources();
        for (const auto& desc : cbs)
        {
            if (desc.type != ParameterBlockReflection::ResourceDesc::Type::Cbv) continue;
            if (desc.regIndex == baseRegIndex && desc.regSpace == regSpace)
            {
                const auto& pVar = mpReflector->getResource(desc.name);
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

    bool ParameterBlock::setConstantBuffer(const std::string& name, const ConstantBuffer::SharedPtr& pCB)
    {
        // Find the buffer
        uint32_t arrayIndex;
        const auto loc = getBufferBindLocation(mpReflector.get(), name, arrayIndex, ReflectionResourceType::Type::ConstantBuffer);

        if (loc.regSpace == ParameterBlockReflection::kInvalidLocation)
        {
            logWarning("Constant buffer \"" + name + "\" was not found. Ignoring setConstantBuffer() call.");
            return false;
        }
        return setConstantBuffer(loc.regSpace, loc.regIndex, arrayIndex, pCB);
    }

    void setResourceSrvUavCommon(const ParameterBlock::BindLocation& bindLoc,
        uint32_t descOffset,
        ReflectionResourceType::ShaderAccess shaderAccess,
        const Resource::SharedPtr& resource,
        ParameterBlock::ResourceMap<ShaderResourceView>& assignedSrvs,
        ParameterBlock::ResourceMap<UnorderedAccessView>& assignedUavs,
        std::vector<ParameterBlock::RootSet>& rootSets)
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

    void setResourceSrvUavCommon(const ReflectionVar* pVar, const Resource::SharedPtr& resource, ParameterBlock::ResourceMap<ShaderResourceView>& assignedSrvs, ParameterBlock::ResourceMap<UnorderedAccessView>& assignedUavs, std::vector<ParameterBlock::RootSet>& rootSets)
    {
        auto shaderAccess = pVar->getType()->unwrapArray()->asResourceType()->getShaderAccess();
        setResourceSrvUavCommon({ pVar->getRegisterSpace(), pVar->getRegisterIndex() }, pVar->getDescOffset(), shaderAccess, resource, assignedSrvs, assignedUavs, rootSets);
    }

    bool ParameterBlock::setRawBuffer(const std::string& name, Buffer::SharedPtr pBuf)
    {
        // Find the buffer
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::RawBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "setRawBuffer()") == false)
        {
            return false;
        }

        setResourceSrvUavCommon(pVar.get(), pBuf, mAssignedSrvs, mAssignedUavs, mRootSets);
        return true;
    }

    bool ParameterBlock::setTypedBuffer(const std::string& name, TypedBufferBase::SharedPtr pBuf)
    {
        // Find the buffer
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::TypedBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "setTypedBuffer()") == false)
        {
            return false;
        }

        setResourceSrvUavCommon(pVar.get(), pBuf, mAssignedSrvs, mAssignedUavs, mRootSets);
        return true;
    }

    bool ParameterBlock::setStructuredBuffer(const std::string& name, StructuredBuffer::SharedPtr pBuf)
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::StructuredBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "setStructuredBuffer()") == false)
        {
            return false;
        }

        setResourceSrvUavCommon(pVar.get(), pBuf, mAssignedSrvs, mAssignedUavs, mRootSets);
        return true;
    }

    template<typename ResourceType>
    typename ResourceType::SharedPtr getResourceFromSrvUavCommon(uint32_t regSpace, uint32_t regIndex, uint32_t arrayIndex, ReflectionResourceType::ShaderAccess shaderAccess, const ParameterBlock::ResourceMap<ShaderResourceView>& assignedSrvs, const ParameterBlock::ResourceMap<UnorderedAccessView>& assignedUavs, const std::string& varName, const std::string& funcName)
    {
        ParameterBlock::BindLocation bindLoc(regSpace, regIndex);
        switch (shaderAccess)
        {
        case ReflectionResourceType::ShaderAccess::ReadWrite:
            if (assignedUavs.find(bindLoc) == assignedUavs.end())
            {
                logWarning("ParameterBlock::" + funcName + " - variable \"" + varName + "\' was not found in UAVs. Shader Access = " + to_string(shaderAccess));
                return nullptr;
            }
            return std::dynamic_pointer_cast<ResourceType>(assignedUavs.at(bindLoc)[arrayIndex].pResource);

        case ReflectionResourceType::ShaderAccess::Read:
            if (assignedSrvs.find(bindLoc) == assignedSrvs.end())
            {
                logWarning("ParameterBlock::" + funcName + " - variable \"" + varName + "\' was not found in SRVs. Shader Access = " + to_string(shaderAccess));
                return nullptr;
            }
            return std::dynamic_pointer_cast<ResourceType>(assignedSrvs.at(bindLoc)[arrayIndex].pResource);

        default:
            should_not_get_here();
        }

        return nullptr;
    }

    template<typename ResourceType>
    typename ResourceType::SharedPtr getResourceFromSrvUavCommon(const ReflectionVar* pVar, const ParameterBlock::ResourceMap<ShaderResourceView>& assignedSrvs, const ParameterBlock::ResourceMap<UnorderedAccessView>& assignedUavs, const std::string& varName, const std::string& funcName)
    {
        return getResourceFromSrvUavCommon<ResourceType>(pVar->getRegisterSpace(), pVar->getRegisterIndex(), pVar->getDescOffset(), pVar->getType()->asResourceType()->getShaderAccess(), assignedSrvs, assignedUavs, varName, funcName);
    }

    Buffer::SharedPtr ParameterBlock::getRawBuffer(const std::string& name) const
    {
        // Find the buffer
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::RawBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "getRawBuffer()") == false)
        {
            return false;
        }

        return getResourceFromSrvUavCommon<Buffer>(pVar.get(), mAssignedSrvs, mAssignedUavs, name, "getRawBuffer()");
    }

    TypedBufferBase::SharedPtr ParameterBlock::getTypedBuffer(const std::string& name) const
    {
        // Find the buffer
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::TypedBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "getTypedBuffer()") == false)

        {
            return false;
        }

        return getResourceFromSrvUavCommon<TypedBufferBase>(pVar.get(), mAssignedSrvs, mAssignedUavs, name, "getTypedBuffer()");
    }

    StructuredBuffer::SharedPtr ParameterBlock::getStructuredBuffer(const std::string& name) const
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

        if (pVar == nullptr) return nullptr;
        return getResourceFromSrvUavCommon<StructuredBuffer>(pVar.get(), mAssignedSrvs, mAssignedUavs, name, "getStructuredBuffer()");
    }

    bool ParameterBlock::setSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const Sampler::SharedPtr& pSampler)
    {
        auto& it = mAssignedSamplers.at({ regSpace, baseRegIndex })[arrayIndex];
        if (it.pSampler != pSampler)
        {
            it.pSampler = pSampler;
            mRootSets[it.rootData.rootIndex].pDescSet = nullptr;
        }
        return true;
    }

    bool ParameterBlock::setSampler(const std::string& name, const Sampler::SharedPtr& pSampler)
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Sampler, ReflectionResourceType::ShaderAccess::Read, false, name, "setSampler()") == false)
        {
            return false;
        }

        return setSampler(pVar->getRegisterSpace(), pVar->getRegisterIndex(), pVar->getDescOffset(), pSampler);
    }

    Sampler::SharedPtr ParameterBlock::getSampler(const std::string& name) const
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Sampler, ReflectionResourceType::ShaderAccess::Read, false, name, "getSampler()") == false)
        {
            return nullptr;
        }
        // #PARAMBLOCK handle descOffset
        return getSampler(pVar->getRegisterSpace(), pVar->getRegisterIndex(), pVar->getDescOffset());
    }

    Sampler::SharedPtr ParameterBlock::getSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        auto it = mAssignedSamplers.find({ regSpace, baseRegIndex });
        if (it == mAssignedSamplers.end())
        {
            logWarning("ParameterBlock::getSampler() - Cannot find sampler at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace));
            return nullptr;
        }

        return it->second[arrayIndex].pSampler;
    }

    ShaderResourceView::SharedPtr ParameterBlock::getSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        auto it = mAssignedSrvs.find({ regSpace, baseRegIndex });
        if (it == mAssignedSrvs.end())
        {
            logWarning("ParameterBlock::getSrv() - Cannot find SRV at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace));
            return nullptr;
        }

        return it->second[arrayIndex].pView;
    }


    UnorderedAccessView::SharedPtr ParameterBlock::getUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
        auto it = mAssignedUavs.find({ regSpace, baseRegIndex });
        if (it == mAssignedUavs.end())
        {
            logWarning("ParameterBlock::getUav() - Cannot find UAV at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace));
            return nullptr;
        }

        return it->second[arrayIndex].pView;
    }

    bool ParameterBlock::setTexture(const std::string& name, const Texture::SharedPtr& pTexture)
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Texture, ReflectionResourceType::ShaderAccess::Undefined, false, name, "setTexture()") == false)
        {
            return false;
        }

        setResourceSrvUavCommon(pVar.get(), pTexture, mAssignedSrvs, mAssignedUavs, mRootSets);

        return true;
    }

    Texture::SharedPtr ParameterBlock::getTexture(const std::string& name) const
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

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

    bool ParameterBlock::setSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ShaderResourceView::SharedPtr& pSrv)
    {
        auto it = mAssignedSrvs.find({ regSpace, baseRegIndex });
        if (it != mAssignedSrvs.end())
        {
            auto& data = it->second[arrayIndex];
            if (data.pView != pSrv)
            {
                mRootSets[data.rootData.rootIndex].pDescSet = nullptr;
                data.pView = pSrv;
                data.pResource = getResourceFromView(pSrv.get());
            }
        }
        else
        {
            logWarning("Can't find SRV with index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace) + ". Ignoring call to ParameterBlock::setSrv()");
            return false;
        }

        return true;
    }

    bool ParameterBlock::setUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const UnorderedAccessView::SharedPtr& pUav)
    {
        auto it = mAssignedUavs.find({ regSpace, baseRegIndex });
        if (it != mAssignedUavs.end())
        {
            auto& data = it->second[arrayIndex];
            if (data.pView != pUav)
            {
                mRootSets[data.rootData.rootIndex].pDescSet = nullptr;
                data.pView = pUav;
                data.pResource = getResourceFromView(pUav.get());
            }
        }
        else
        {
            logWarning("Can't find UAV with index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace) + ". Ignoring call to ParameterBlock::setUav()");
            return false;
        }

        return true;
    }

    void bindSamplers(const ParameterBlock::ResourceMap<Sampler>& samplers, const ParameterBlock::RootSetVec& rootSets)
    {
        // Bind the samplers
        for (auto& samplerIt : samplers)
        {
            const auto& samplerVec = samplerIt.second;
            const auto& rootData = samplerVec[0].rootData;
            if (rootSets[rootData.rootIndex].dirty)
            {
                for (uint32_t i = 0; i < samplerVec.size(); i++)
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

    template<typename ViewType, bool isUav>
    void bindUavSrvCommon(const ParameterBlock::ResourceMap<ViewType>& resMap, const ParameterBlock::RootSetVec& rootSets)
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

    void uploadConstantBuffers(const ParameterBlock::ResourceMap<ConstantBuffer>& cbMap, ParameterBlock::RootSetVec& rootSets)
    {
        for (auto& bufIt : cbMap)
        {
            const auto& cbVec = bufIt.second;
            for (size_t i = 0; i < cbVec.size(); i++)
            {
                const auto& rootData = cbVec[i].rootData;
                ConstantBuffer* pCB = dynamic_cast<ConstantBuffer*>(cbVec[i].pResource.get());

                if (pCB && pCB->uploadToGPU())
                {
                    rootSets[rootData.rootIndex].pDescSet = nullptr;
                }
            }
        }
    }

    template<typename ViewType, bool isUav>
    void uploadUavSrvCommon(CopyContext* pContext, const ParameterBlock::ResourceMap<ViewType>& resMap, ParameterBlock::RootSetVec& rootSets)
    {
        for (auto& resIt : resMap)
        {
            const auto& resVec = resIt.second;
            auto& rootData = resVec[0].rootData;

            for (const auto& resDesc : resVec)
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

    void bindConstantBuffers(const ParameterBlock::ResourceMap<ConstantBuffer>& cbMap, const ParameterBlock::RootSetVec& rootSets)
    {
        for (auto& bufIt : cbMap)
        {
            const auto& cbVec = bufIt.second;
            auto& rootData = cbVec[0].rootData;

            if (rootSets[rootData.rootIndex].dirty)
            {
                for (uint32_t i = 0; i < cbVec.size(); i++)
                {
                    auto& cbDesc = cbVec[i];
                    ConstantBuffer* pCB = dynamic_cast<ConstantBuffer*>(cbDesc.pResource.get());
                    ConstantBufferView::SharedPtr pView = pCB ? pCB->getCbv() : ConstantBufferView::getNullView();

                    // Get the set and copy the GPU handle
                    const auto& pDescSet = rootSets[rootData.rootIndex].pDescSet;
                    pDescSet->setCbv(rootData.rangeIndex, i, pView);
                }
            }
        }
    }

    bool ParameterBlock::prepareForDraw(CopyContext* pContext, const RootSignature* pRootSig)
    {
        // Upload the resources. This will also invalidate descriptor-sets that contain dynamic resources
        uploadConstantBuffers(mAssignedCbs, mRootSets);
        uploadUavSrvCommon<ShaderResourceView, false>(pContext, mAssignedSrvs, mRootSets);
        uploadUavSrvCommon<UnorderedAccessView, true>(pContext, mAssignedUavs, mRootSets);

        // Allocate and mark the dirty sets
        for (uint32_t i = 0; i < mRootSets.size(); i++)
        {
            mRootSets[i].dirty = (mRootSets[i].pDescSet == nullptr);
            if (mRootSets[i].pDescSet == nullptr)
            {
                DescriptorSet::Layout layout;
                const auto& set = pRootSig->getDescriptorSet(i);
                mRootSets[i].pDescSet = DescriptorSet::create(gpDevice->getGpuDescriptorPool(), set);
                if (mRootSets[i].pDescSet == nullptr)
                {
                    return false;
                }
            }
        }

        bindUavSrvCommon<ShaderResourceView, false>(mAssignedSrvs, mRootSets);
        bindUavSrvCommon<UnorderedAccessView, true>(mAssignedUavs, mRootSets);
        bindSamplers(mAssignedSamplers, mRootSets);
        bindConstantBuffers(mAssignedCbs, mRootSets);
        return true;
    }
}