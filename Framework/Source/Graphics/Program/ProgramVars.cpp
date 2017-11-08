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
    ReflectionType::ShaderAccess getRequiredShaderAccess(RootSignature::DescType type);

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
    bool initializeBuffersMap(ProgramVars::ResourceMap<ViewType>& bufferMap, bool createBuffers, const ViewInitFunc& viewInitFunc, const ParameterBlockReflection::ResourceMap& resMap, const RootSignature* pRootSig)
    {
        ReflectionType::ShaderAccess shaderAccess = getRequiredShaderAccess(descType);

        for (auto& buf : resMap)
        {
            const ReflectionVar::SharedConstPtr& pVar = buf.second;
            const ReflectionType::SharedConstPtr& pType = pVar->getType();
            if (pType->getShaderAccess() == shaderAccess)
            {
                uint32_t regIndex = pVar->getRegisterIndex();
                uint32_t regSpace = pVar->getRegisterSpace();
                uint32_t arraySize = max(1u, pType->getArraySize());
                ProgramVars::ResourceData<ViewType> data(findRootData(pRootSig, regIndex, regSpace, descType));
                if (data.rootData.rootIndex == -1)
                {
                    logError("Can't find a root-signature information matching buffer '" + pVar->getName() + " when creating ProgramVars");
                    return false;
                }

                for(uint32_t a = 0 ; a < arraySize ; a++)
                {
                    // Only create the buffer if needed
                    if (createBuffers)
                    {
                        data.pResource = BufferType::create(pType);
                        data.pView = viewInitFunc(data.pResource);
                    }

                    bufferMap[ProgramVars::BindLocation(regSpace, regIndex)].push_back(data);
                }
            }
        }
        return true;
    }

    static RootSignature::DescType getRootDescTypeFromResourceType(ReflectionType::Type type, ReflectionType::ShaderAccess access)
    {
        switch (type)
        {
        case ReflectionType::Type::Texture:
        case ReflectionType::Type::RawBuffer:  // Vulkan doesn't have raw-buffer and DX doesn't care
            return (access == ReflectionType::ShaderAccess::Read) ? RootSignature::DescType::TextureSrv : RootSignature::DescType::TextureUav;
        case ReflectionType::Type::StructuredBuffer:
            return (access == ReflectionType::ShaderAccess::Read) ? RootSignature::DescType::StructuredBufferSrv : RootSignature::DescType::StructuredBufferUav;
        case ReflectionType::Type::TypedBuffer:
            return (access == ReflectionType::ShaderAccess::Read) ? RootSignature::DescType::TypedBufferSrv : RootSignature::DescType::TypedBufferUav;
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
        initializeBuffersMap<ConstantBuffer, ConstantBuffer, RootSignature::DescType::Cbv>(mAssignedCbs, createBuffers, getNullPtrFunc, pGlobalBlock->getConstantBuffers(), mpRootSignature.get());
        initializeBuffersMap<StructuredBuffer, ShaderResourceView, RootSignature::DescType::StructuredBufferSrv>(mAssignedSrvs, createBuffers, getSrvFunc, pGlobalBlock->getStructuredBuffers(), mpRootSignature.get());
        initializeBuffersMap<StructuredBuffer, UnorderedAccessView, RootSignature::DescType::StructuredBufferUav>(mAssignedUavs, createBuffers, getUavFunc, pGlobalBlock->getStructuredBuffers(), mpRootSignature.get());

        // Initialize the textures and samplers map
        for (const auto& res : pGlobalBlock->getResources())
        {
            const ReflectionVar::SharedConstPtr& pVar = res.second;
            const ReflectionType::SharedConstPtr& pType = pVar->getType();
            uint32_t count = pType->getArraySize() ? pType->getArraySize() : 1;
            for (uint32_t index = 0; index < (0 + count); ++index) // PARAMBLOCK the 0 used to be descOffset
            {
                uint32_t regIndex = pVar->getRegisterIndex();
                uint32_t regSpace = pVar->getRegisterSpace();
                BindLocation loc(regSpace, regIndex);
                ReflectionType::ShaderAccess shaderAccess = pType->getShaderAccess();
                ReflectionType::Type type = pType->getType();
                switch (type)
                {
                case ReflectionType::Type::Sampler:
                    mAssignedSamplers[loc].push_back(findRootData(mpRootSignature.get(), regIndex, regSpace, RootSignature::DescType::Sampler));
                    break;
                case ReflectionType::Type::Texture:
                case ReflectionType::Type::RawBuffer:
                case ReflectionType::Type::TypedBuffer:
                    if (shaderAccess == ReflectionType::ShaderAccess::Read)
                    {
                        assert(mAssignedSrvs.find(loc) == mAssignedSrvs.end() || mAssignedSrvs[loc].size() == index);
                        mAssignedSrvs[loc].push_back(findRootData(mpRootSignature.get(), regIndex, regSpace, getRootDescTypeFromResourceType(type, shaderAccess)));
                    }
                    else
                    {
                        assert(mAssignedUavs.find(loc) == mAssignedUavs.end() || mAssignedUavs[loc].size() == index);
                        assert(shaderAccess == ReflectionType::ShaderAccess::ReadWrite);
                        mAssignedUavs[loc].push_back(findRootData(mpRootSignature.get(), regIndex, regSpace, getRootDescTypeFromResourceType(type, shaderAccess)));
                    }
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

//     static const ProgramReflection::BindLocation getBufferBindLocation(const ProgramReflection* pReflector, const std::string& name, uint32_t& arrayIndex, ProgramReflection::BufferReflection::Type bufferType)
//     {
//         auto binding = pReflector->getBufferBinding(name);
//         arrayIndex = 0;
//         if (binding.regSpace == ProgramReflection::kInvalidLocation)
//         {
//             std::string nameNoIndex;
//             if (parseArrayIndex(name, nameNoIndex, arrayIndex) == false) return binding;
//             binding = pReflector->getBufferBinding(name);
//             if (binding.regSpace == ProgramReflection::kInvalidLocation)
//             {
//                 logWarning("Constant buffer \"" + name + "\" was not found. Ignoring getConstantBuffer() call.");
//             }
//         }
// 
//         auto& pDesc = pReflector->getBufferDesc(name, bufferType);
//         if (pDesc->getType() != bufferType)
//         {
//             logWarning("Buffer \"" + name + "\" is not a " + to_string(bufferType) + ". Type = " + to_string(pDesc->getType()));
//             return ProgramReflection::BindLocation();
//         }
//         return binding;
//     }

    ConstantBuffer::SharedPtr ProgramVars::getConstantBuffer(const std::string& name) const
    {
//         uint32_t arrayIndex;
//         const auto& binding = getBufferBindLocation(mpReflector.get(), name, arrayIndex, ProgramReflection::BufferReflection::Type::Constant);
//         if (binding.regSpace == ProgramReflection::kInvalidLocation)
//         {
//             logWarning("Constant buffer \"" + name + "\" was not found. Ignoring getConstantBuffer() call.");
//             return nullptr;
//         }
        return getConstantBuffer(0, 0, 0);
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
//         const auto& desc = mpReflector->getBufferDesc(regSpace, baseRegIndex, ProgramReflection::ShaderAccess::Read, ProgramReflection::BufferReflection::Type::Constant);
//         if (desc->getRequiredSize() > pCB->getSize())
//         {
//             logError("Can't attach the constant buffer. Size mismatch.");
//             return false;
//         }
//        mAssignedCbs[loc][arrayIndex].pResource = pCB;
        return true;
    }

    bool ProgramVars::setConstantBuffer(const std::string& name, const ConstantBuffer::SharedPtr& pCB)
    {
        // Find the buffer
//        uint32_t arrayIndex;
//         const auto loc = getBufferBindLocation(mpReflector.get(), name, arrayIndex, ProgramReflection::BufferReflection::Type::Constant);
// 
//         if (loc.regSpace == ProgramReflection::kInvalidLocation)
//         {
//             logWarning("Constant buffer \"" + name + "\" was not found. Ignoring setConstantBuffer() call.");
//             return false;
//         }
//         return setConstantBuffer(loc.regSpace, loc.baseRegIndex, arrayIndex, pCB);
        return false;
    }

//     void setResourceSrvUavCommon(const ProgramVars::BindLocation& bindLoc, uint32_t arrayIndex, ProgramReflection::ShaderAccess shaderAccess, const Resource::SharedPtr& resource, ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs,
//         std::vector<ProgramVars::RootSet>& rootSets)
//     {
//         switch (shaderAccess)
//         {
//         case ProgramReflection::ShaderAccess::ReadWrite:
//         {
//             auto uavIt = assignedUavs.find(bindLoc);
//             assert(uavIt != assignedUavs.end());
//             auto resUav = resource ? resource->getUAV() : nullptr;
//             auto& data = uavIt->second[arrayIndex];
//             if (data.pView != resUav)
//             {
//                 rootSets[data.rootData.rootIndex].pDescSet = nullptr;
//                 data.pResource = resource;
//                 data.pView = resUav;
//             }
//             break;
//         }
// 
//         case ProgramReflection::ShaderAccess::Read:
//         {
//             auto srvIt = assignedSrvs.find(bindLoc);
//             assert(srvIt != assignedSrvs.end());
// 
//             auto resSrv = resource ? resource->getSRV() : nullptr;
//             auto& data = srvIt->second[arrayIndex];
// 
//             if (data.pView != resSrv)
//             {
//                 rootSets[data.rootData.rootIndex].pDescSet = nullptr;
//                 data.pResource = resource;
//                 data.pView = resSrv;
//             }
//             break;
//         }
// 
//         default:
//             should_not_get_here();
//         }
//     }

//     void setResourceSrvUavCommon(const ProgramReflection::Resource* pDesc, uint32_t arrayIndex, const Resource::SharedPtr& resource, ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs, std::vector<ProgramVars::RootSet>& rootSets)
//     {
// 		arrayIndex += pDesc->descOffset;
// //        setResourceSrvUavCommon({ pDesc->regSpace, pDesc->regIndex }, arrayIndex, pDesc->shaderAccess, resource, assignedSrvs, assignedUavs, rootSets);
//     }

//     void setResourceSrvUavCommon(const ProgramReflection::BufferReflection *pDesc, uint32_t arrayIndex, const Resource::SharedPtr& resource, ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs, std::vector<ProgramVars::RootSet>& rootSets)
//     {
// //         setResourceSrvUavCommon({ pDesc->getRegisterSpace(), pDesc->getRegisterIndex() }, arrayIndex, pDesc->getShaderAccess(), resource, assignedSrvs, assignedUavs, rootSets);
//     }

//     bool verifyBufferResourceDesc(const ProgramReflection::Resource* pDesc, uint32_t arrayIndex, const std::string& name, ProgramReflection::Resource::ResourceType expectedType, ProgramReflection::Resource::Dimensions expectedDims, const std::string& funcName)
//     {
//         if (pDesc == nullptr)
//         {
//             logWarning("ProgramVars::" + funcName + " - resource \"" + name + "\" was not found. Ignoring " + funcName + " call.");
//             return false;
//         }
// 
//         if (pDesc->type != expectedType || pDesc->dims != expectedDims)
//         {
//             logWarning("ProgramVars::" + funcName + " - variable '" + name + "' is the incorrect type. VarType = " + to_string(pDesc->type) + ", VarDims = " + to_string(pDesc->dims) + ". Ignoring call");
//             return false;
//         }
// 
//         if (pDesc->arraySize && arrayIndex >= pDesc->arraySize)
//         {
//             logWarning("ProgramVars::" + funcName + " was called, but array index is out-of-bound. Ignoring call");
//             return false;
//         }
//         return true;
//     }

//     static const ProgramReflection::Resource* getResourceDescAndArrayIndex(const ProgramReflection* pReflector, const std::string& name, uint32_t& arrayIndex)
//     {
//         const ProgramReflection::Resource* pDesc = pReflector->getResourceDesc(name);
//         arrayIndex = 0;
//         if (!pDesc)
//         {
//             std::string nameNoIndex;
//             if (parseArrayIndex(name, nameNoIndex, arrayIndex) == false) return nullptr;
//             pDesc = pReflector->getResourceDesc(nameNoIndex);
//             if (pDesc->arraySize == 0) return nullptr;
//         }
//         return pDesc;
//     }

    bool ProgramVars::setRawBuffer(const std::string& name, Buffer::SharedPtr pBuf)
    {
        // Find the buffer
//         uint32_t arrayIndex;
//         const ProgramReflection::Resource* pDesc = getResourceDescAndArrayIndex(mpReflector.get(), name, arrayIndex);
// 
//         if (verifyBufferResourceDesc(pDesc, arrayIndex, name, ProgramReflection::Resource::ResourceType::RawBuffer, ProgramReflection::Resource::Dimensions::Buffer, "setRawBuffer()") == false)
//         {
//             return false;
//         }
// 
//         setResourceSrvUavCommon(pDesc, arrayIndex, pBuf, mAssignedSrvs, mAssignedUavs, mRootSets);
// 
        return true;
    }

    bool ProgramVars::setTypedBuffer(const std::string& name, TypedBufferBase::SharedPtr pBuf)
    {
        // Find the buffer
//         uint32_t arrayIndex;
//         const ProgramReflection::Resource* pDesc = getResourceDescAndArrayIndex(mpReflector.get(), name, arrayIndex);
// 
//         if (verifyBufferResourceDesc(pDesc, arrayIndex, name, ProgramReflection::Resource::ResourceType::TypedBuffer, ProgramReflection::Resource::Dimensions::Buffer, "setTypedBuffer()") == false)
//         {
//             return false;
//         }
// 
//         setResourceSrvUavCommon(pDesc, arrayIndex, pBuf, mAssignedSrvs, mAssignedUavs, mRootSets);
// 
        return true;
    }

//     static const ProgramReflection::BufferReflection* getStructuredBufferReflection(const ProgramReflection* pReflector, const std::string& name, uint32_t& arrayIndex, const std::string& callStr)
//     {
//         arrayIndex = 0;
//         const ProgramReflection::BufferReflection* pBufDesc = pReflector->getBufferDesc(name, ProgramReflection::BufferReflection::Type::Structured).get();
//         if (pBufDesc == nullptr)
//         {
//             std::string noArray;
//             if (parseArrayIndex(name, noArray, arrayIndex))
//             {
//                 pBufDesc = pReflector->getBufferDesc(noArray, ProgramReflection::BufferReflection::Type::Structured).get();
//             }
//         }
// 
//         if (pBufDesc == nullptr)
//         {
//             logWarning("Structured buffer \"" + name + "\" was not found. Ignoring " + callStr + "StructuredBuffer() call.");
//             return false;
//         }
//         return pBufDesc;
//     }

    bool ProgramVars::setStructuredBuffer(const std::string& name, StructuredBuffer::SharedPtr pBuf)
    {
//         uint32_t arrayIndex;
//         const ProgramReflection::BufferReflection* pBufDesc = getStructuredBufferReflection(mpReflector.get(), name, arrayIndex, "set");
//         if (!pBufDesc) return false;
//         setResourceSrvUavCommon(pBufDesc, arrayIndex, pBuf, mAssignedSrvs, mAssignedUavs, mRootSets);
        return true;
    }

//     template<typename ResourceType>
//     typename ResourceType::SharedPtr getResourceFromSrvUavCommon(uint32_t regSpace, uint32_t regIndex, uint32_t arrayIndex, ProgramReflection::ShaderAccess shaderAccess, const ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, const ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs, const std::string& varName, const std::string& funcName)
//     {
//         ProgramVars::BindLocation bindLoc(regSpace, regIndex);
//         switch (shaderAccess)
//         {
//         case ProgramReflection::ShaderAccess::ReadWrite:
//             if (assignedUavs.find(bindLoc) == assignedUavs.end())
//             {
//                 logWarning("ProgramVars::" + funcName + " - variable \"" + varName + "\' was not found in UAVs. Shader Access = " + to_string(shaderAccess));
//                 return nullptr;
//             }
//             return std::dynamic_pointer_cast<ResourceType>(assignedUavs.at(bindLoc)[arrayIndex].pResource);
// 
//         case ProgramReflection::ShaderAccess::Read:
//             if (assignedSrvs.find(bindLoc) == assignedSrvs.end())
//             {
//                 logWarning("ProgramVars::" + funcName + " - variable \"" + varName + "\' was not found in SRVs. Shader Access = " + to_string(shaderAccess));
//                 return nullptr;
//             }
//             return std::dynamic_pointer_cast<ResourceType>(assignedSrvs.at(bindLoc)[arrayIndex].pResource);
// 
//         default:
//             should_not_get_here();
//         }
// 
//         return nullptr;
//     }

//     template<typename ResourceType>
//     typename ResourceType::SharedPtr getResourceFromSrvUavCommon(const ProgramReflection::Resource *pDesc, uint32_t arrayIndex, const ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, const ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs, const std::string& varName, const std::string& funcName)
//     {
//         return getResourceFromSrvUavCommon<ResourceType>(pDesc->regSpace, pDesc->regIndex, arrayIndex + pDesc->descOffset, pDesc->shaderAccess, assignedSrvs, assignedUavs, varName, funcName);
//     }
// 
//     template<typename ResourceType>
//     typename ResourceType::SharedPtr getResourceFromSrvUavCommon(const ProgramReflection::BufferReflection *pBufDesc, uint32_t arrayIndex, const ProgramVars::ResourceMap<ShaderResourceView>& assignedSrvs, const ProgramVars::ResourceMap<UnorderedAccessView>& assignedUavs, const std::string& varName, const std::string& funcName)
//     {
//         return getResourceFromSrvUavCommon<ResourceType>(pBufDesc->getRegisterSpace(), pBufDesc->getRegisterIndex(), arrayIndex, pBufDesc->getShaderAccess(), assignedSrvs, assignedUavs, varName, funcName);
//     }

    Buffer::SharedPtr ProgramVars::getRawBuffer(const std::string& name) const
    {
        // Find the buffer
//         uint32_t arrayIndex;
//         const ProgramReflection::Resource* pDesc = getResourceDescAndArrayIndex(mpReflector.get(), name, arrayIndex);
// 
//         if (verifyBufferResourceDesc(pDesc,arrayIndex,  name, ProgramReflection::Resource::ResourceType::RawBuffer, ProgramReflection::Resource::Dimensions::Buffer, "getRawBuffer()") == false)
//         {
//             return false;
//         }
// 
//         return getResourceFromSrvUavCommon<Buffer>(pDesc, arrayIndex, mAssignedSrvs, mAssignedUavs, name, "getRawBuffer()");
        return nullptr;
    }

    TypedBufferBase::SharedPtr ProgramVars::getTypedBuffer(const std::string& name) const
    {
//         // Find the buffer
//         uint32_t arrayIndex;
//         const ProgramReflection::Resource* pDesc = getResourceDescAndArrayIndex(mpReflector.get(), name, arrayIndex);
// 
//         if (verifyBufferResourceDesc(pDesc, arrayIndex, name, ProgramReflection::Resource::ResourceType::TypedBuffer, ProgramReflection::Resource::Dimensions::Buffer, "getTypedBuffer()") == false)
//         {
//             return false;
//         }
// 
//         return getResourceFromSrvUavCommon<TypedBufferBase>(pDesc, arrayIndex, mAssignedSrvs, mAssignedUavs, name, "getTypedBuffer()");
        return nullptr;
    }

    StructuredBuffer::SharedPtr ProgramVars::getStructuredBuffer(const std::string& name) const
    {
//         uint32_t arrayIndex;
//         const ProgramReflection::BufferReflection* pBufDesc = getStructuredBufferReflection(mpReflector.get(), name, arrayIndex, "get");
//         if (pBufDesc == nullptr) return nullptr;
//         return getResourceFromSrvUavCommon<StructuredBuffer>(pBufDesc, arrayIndex, mAssignedSrvs, mAssignedUavs, name, "getStructuredBuffer()");
        return nullptr;
    }

//     bool verifyResourceDesc(const ProgramReflection::Resource* pDesc, uint32_t arrayIndex, ProgramReflection::Resource::ResourceType type, ProgramReflection::ShaderAccess access, const std::string& varName, const std::string& funcName)
//     {
//         if (pDesc == nullptr)
//         {
//             logWarning(to_string(type) + " \"" + varName + "\" was not found. Ignoring " + funcName + " call.");
//             return false;
//         }
// #if _LOG_ENABLED
//         if (pDesc->type != type)
//         {
//             logWarning("ProgramVars::" + funcName + " was called, but variable \"" + varName + "\" has different resource type. Expecting + " + to_string(pDesc->type) + " but provided resource is " + to_string(type) + ". Ignoring call");
//             return false;
//         }
// 
//         if (access != ProgramReflection::ShaderAccess::Undefined && pDesc->shaderAccess != access)
//         {
//             logWarning("ProgramVars::" + funcName + " was called, but variable \"" + varName + "\" has different shader access type. Expecting + " + to_string(pDesc->shaderAccess) + " but provided resource is " + to_string(access) + ". Ignoring call");
//             return false;
//         }
// 
//         if (pDesc->arraySize && arrayIndex >= pDesc->arraySize)
//         {
//             logWarning("ProgramVars::" + funcName + " was called, but array index is out-of-bound. Ignoring call");
//             return false;
//         }
// #endif
//         return true;
//     }

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
//         uint32_t arrayIndex;
//         const ProgramReflection::Resource* pDesc = getResourceDescAndArrayIndex(mpReflector.get(), name, arrayIndex);
//         if (verifyResourceDesc(pDesc, arrayIndex, ProgramReflection::Resource::ResourceType::Sampler, ProgramReflection::ShaderAccess::Read, name, "setSampler()") == false)
//         {
//             return false;
//         }
// 
//         return setSampler(pDesc->regSpace, pDesc->regIndex, arrayIndex + pDesc->descOffset, pSampler);
        return false;
    }

    Sampler::SharedPtr ProgramVars::getSampler(const std::string& name) const
    {
//         uint32_t arrayIndex;
//         const ProgramReflection::Resource* pDesc = getResourceDescAndArrayIndex(mpReflector.get(), name, arrayIndex);
//         if (verifyResourceDesc(pDesc, 0, ProgramReflection::Resource::ResourceType::Sampler, ProgramReflection::ShaderAccess::Read, name, "getSampler()") == false)
//         {
//             return nullptr;
//         }
// 
//         return getSampler(pDesc->regSpace, pDesc->regIndex, arrayIndex + pDesc->descOffset);
        return nullptr;
    }

    Sampler::SharedPtr ProgramVars::getSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const
    {
//         auto it = mAssignedSamplers.find({regSpace, baseRegIndex});
//         if (it == mAssignedSamplers.end())
//         {
//             logWarning("ProgramVars::getSampler() - Cannot find sampler at index " + std::to_string(baseRegIndex) + ", space " + std::to_string(regSpace));
//             return nullptr;
//         }
// 
//         return it->second[arrayIndex].pSampler;
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
//        uint32_t arrayIndex;
//         const ProgramReflection::Resource* pDesc = getResourceDescAndArrayIndex(mpReflector.get(), name, arrayIndex);
// 
//         if (verifyResourceDesc(pDesc, arrayIndex, ProgramReflection::Resource::ResourceType::Texture, ProgramReflection::ShaderAccess::Undefined,  name, "setTexture()") == false)
//         {
//             return false;
//         }
// 
//         setResourceSrvUavCommon(pDesc, arrayIndex, pTexture, mAssignedSrvs, mAssignedUavs, mRootSets);

        return true;
    }

    Texture::SharedPtr ProgramVars::getTexture(const std::string& name) const
    {
//        uint32_t arrayIndex;
//         const ProgramReflection::Resource* pDesc = getResourceDescAndArrayIndex(mpReflector.get(), name, arrayIndex);
// 
//         if (verifyResourceDesc(pDesc, arrayIndex, ProgramReflection::Resource::ResourceType::Texture, ProgramReflection::ShaderAccess::Undefined, name, "getTexture()") == false)
//         {
//             return nullptr;
//         }
// 
//         return getResourceFromSrvUavCommon<Texture>(pDesc, arrayIndex, mAssignedSrvs, mAssignedUavs, name, "getTexture()");
        return nullptr;
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
//         const auto& pDesc = pReflector->getResourceDesc(name);
//         if (!pDesc) return loc;
//         loc.baseRegIndex = pDesc->regIndex;
//         loc.regSpace = pDesc->regSpace;
        return loc;
    }

    ProgramVars::BindLocation getBufferBindLocation(const ProgramReflection* pReflector, const std::string& name)
    {
        ProgramVars::BindLocation loc;
//         const auto& desc = pReflector->getBufferBinding(name);
//         loc.baseRegIndex = desc.baseRegIndex;
//         loc.regSpace = desc.regSpace;
        return loc;
    }
}