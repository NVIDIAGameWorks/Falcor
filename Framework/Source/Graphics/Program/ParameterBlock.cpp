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
#include "API/Device.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
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

    ParameterBlock::~ParameterBlock() = default;

    ParameterBlock::AssignedResource::AssignedResource() : pResource(nullptr), type(DescriptorSet::Type::Count), pCB(nullptr)
    {
    }

    ParameterBlock::AssignedResource::AssignedResource(const AssignedResource& other) : pResource(other.pResource), type(other.type), pCB(nullptr)
    {
        switch (type)
        {
        case DescriptorSet::Type::Cbv:
            pCB = other.pCB;
            break;
        case DescriptorSet::Type::Sampler:
            pSampler = other.pSampler;
            break;
        case DescriptorSet::Type::StructuredBufferSrv:
        case DescriptorSet::Type::TypedBufferSrv:
        case DescriptorSet::Type::TextureSrv:
            pSRV = other.pSRV;
            break;
        case DescriptorSet::Type::StructuredBufferUav:
        case DescriptorSet::Type::TypedBufferUav:
        case DescriptorSet::Type::TextureUav:
            pUAV = other.pUAV;
            break;
        case DescriptorPool::Type::Count:
            break;
        default:
            should_not_get_here();
            break;
        }
    }

    ParameterBlock::AssignedResource::~AssignedResource()
    {
        pResource = nullptr;
        switch (type)
        {
        case DescriptorSet::Type::Cbv:
            pCB = nullptr;
            break;
        case DescriptorSet::Type::Sampler:
            pSampler = nullptr;
            break;
        case DescriptorSet::Type::StructuredBufferSrv:
        case DescriptorSet::Type::TypedBufferSrv:
        case DescriptorSet::Type::TextureSrv:
            pSRV = nullptr;
            break;
        case DescriptorSet::Type::StructuredBufferUav:
        case DescriptorSet::Type::TypedBufferUav:
        case DescriptorSet::Type::TextureUav:
            pUAV = nullptr;
            break;
        case DescriptorPool::Type::Count:
            break;
        default:
            should_not_get_here();
            break;
        }
    }

    ParameterBlock::SharedPtr ParameterBlock::create(const ParameterBlockReflection::SharedConstPtr& pReflection, bool createBuffers)
    {
        return SharedPtr(new ParameterBlock(pReflection, createBuffers));
    }

    ParameterBlock::ParameterBlock(const ParameterBlockReflection::SharedConstPtr& pReflection, bool createBuffers) : mpReflector(pReflection)
    {
        // Initialize the resource vectors
        const auto& setLayouts = pReflection->getDescriptorSetLayouts();
        mAssignedResources.resize(setLayouts.size());
        mRootSets.resize(setLayouts.size());

        for (size_t s = 0; s < setLayouts.size(); s++)
        {
            const auto& set = setLayouts[s];
            size_t rangeCount = set.getRangeCount();
            mAssignedResources[s].resize(rangeCount);

            for (size_t r = 0; r < rangeCount; r++)
            {
                const auto& range = set.getRange(r);
                mAssignedResources[s][r].resize(range.descCount);
                for (auto& d : mAssignedResources[s][r])
                {
                    d.pResource = nullptr;
                    d.type = range.type;
                    switch (d.type)
                    {
                    case DescriptorSet::Type::TextureSrv:
                    case DescriptorSet::Type::TypedBufferSrv:
                    case DescriptorSet::Type::StructuredBufferSrv:
                        d.pSRV = ShaderResourceView::getNullView();
                        break;
                    case DescriptorSet::Type::TextureUav:
                    case DescriptorSet::Type::TypedBufferUav:
                    case DescriptorSet::Type::StructuredBufferUav:
                        d.pUAV = UnorderedAccessView::getNullView();
                        break;
                    case DescriptorSet::Type::Cbv:
                        break;
                    case DescriptorSet::Type::Sampler:
                        d.pSampler = Sampler::getDefault();
                        break;
                    default:
                        should_not_get_here();
                    }
                }
            }
        }

        // Loop over the resources and create structured and constant buffers
        for (const auto& resource : pReflection->getResourceVec())
        {
            ParameterBlockReflection::BindLocation bindLoc = pReflection->getResourceBinding(resource.name);
            auto& range = mAssignedResources[bindLoc.setIndex][bindLoc.rangeIndex];
            for (size_t r = 0; r < range.size(); r++)
            {
                range[r].requiredSize = resource.pType->getSize();

                if(createBuffers)
                {
                    if (resource.setType == DescriptorSet::Type::StructuredBufferSrv || resource.setType == DescriptorSet::Type::StructuredBufferUav)
                    {
                        StructuredBuffer::SharedPtr pBuffer = StructuredBuffer::create(resource.name, resource.pType);
                        setStructuredBuffer(resource.name, pBuffer);
                    }
                    else if (resource.setType == DescriptorSet::Type::Cbv)
                    {
                        ConstantBuffer::SharedPtr pCB = ConstantBuffer::create(resource.name, resource.pType);
                        setConstantBuffer(resource.name, pCB);
                    }
                }
            }
        }
    }

    static const ProgramReflection::BindLocation getBufferBindLocation(const ParameterBlockReflection* pReflector, const std::string& name, uint32_t& arrayIndex, ReflectionResourceType::Type bufferType)
    {
        // #PARAMBLOCK handle non-global blocks
        const ReflectionVar* pVar = nullptr;
        pVar = pReflector->getResource(name).get();

        if (pVar == nullptr)
        {
            logWarning("Couldn't find a " + to_string(bufferType) + " named " + name);
            return ProgramReflection::BindLocation();
        }
        else if (pVar->getType()->unwrapArray()->asResourceType()->getType() != bufferType)
        {
            logWarning("Found a variable named '" + name + "' but it is not a " + to_string(bufferType));
            return ProgramReflection::BindLocation();
        }

        // #PARAMBLOCK Handle arrays
        arrayIndex = 0;
        return pReflector->getResourceBinding(name);
    }

    ConstantBuffer::SharedPtr ParameterBlock::getConstantBuffer(const std::string& name) const
    {
        uint32_t arrayIndex;
        const auto& binding = getBufferBindLocation(mpReflector.get(), name, arrayIndex, ReflectionResourceType::Type::ConstantBuffer);
        if (binding.setIndex == ParameterBlockReflection::BindLocation::kInvalidLocation)
        {
            logWarning("Constant buffer \"" + name + "\" was not found. Ignoring getConstantBuffer() call.");
            return nullptr;
        }
        return getConstantBuffer(binding, arrayIndex);
    }

    bool ParameterBlock::checkResourceIndices(const BindLocation& bindLocation, uint32_t arrayIndex, DescriptorSet::Type type, const std::string& funcName) const
    {
        bool OK = true;
#if _LOG_ENABLED
        OK = OK && bindLocation.setIndex < mAssignedResources.size();
        OK = OK && bindLocation.rangeIndex < mAssignedResources[bindLocation.setIndex].size();
        OK = OK && arrayIndex < mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex].size();
        OK = OK && ((mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex][arrayIndex].type == type) || (type == DescriptorSet::Type::Count));
        if (!OK)
        {
            logWarning("Can't find resource at set index " + std::to_string(bindLocation.setIndex) + ", range index " + std::to_string(bindLocation.rangeIndex) + ", array index " + std::to_string(arrayIndex) + ". Ignoring " + funcName + " call");
        }
#endif
        return OK;
    }

    ConstantBuffer::SharedPtr ParameterBlock::getConstantBuffer(const BindLocation& bindLocation, uint32_t arrayIndex) const
    {
        if (checkResourceIndices(bindLocation, arrayIndex, DescriptorSet::Type::Cbv, "getConstantBuffer") == false) return nullptr;
        return std::static_pointer_cast<ConstantBuffer>(mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex][arrayIndex].pResource);
    }

    bool ParameterBlock::setConstantBuffer(const BindLocation& bindLocation, uint32_t arrayIndex, const ConstantBuffer::SharedPtr& pCB)
    {
        if (checkResourceIndices(bindLocation, arrayIndex, DescriptorSet::Type::Cbv, "setConstantBuffer") == false) return false;

        auto& res = mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex][arrayIndex];
#if _LOG_ENABLED
        if (res.requiredSize > pCB->getSize())
        {
            logError("Can't attach the constant buffer. Size mismatch.");
            return false;
        }
#endif
        res.pResource = pCB;
        mRootSets[bindLocation.setIndex].pSet = nullptr;
        return true;
    }

    bool ParameterBlock::setConstantBuffer(const std::string& name, const ConstantBuffer::SharedPtr& pCB)
    {
        // Find the buffer
        uint32_t arrayIndex;
        const auto loc = getBufferBindLocation(mpReflector.get(), name, arrayIndex, ReflectionResourceType::Type::ConstantBuffer);

        if (loc.setIndex == ParameterBlockReflection::BindLocation::kInvalidLocation)
        {
            logWarning("Constant buffer \"" + name + "\" was not found. Ignoring setConstantBuffer() call.");
            return false;
        }
        return setConstantBuffer(loc, arrayIndex, pCB);
    }

    static DescriptorSet::Type getSetTypeFromVar(const ReflectionVar::SharedConstPtr& pVar, DescriptorSet::Type srvType, DescriptorSet::Type uavType)
    {
        switch (pVar->getType()->unwrapArray()->asResourceType()->getShaderAccess())
        {
        case ReflectionResourceType::ShaderAccess::Read:
            return srvType;
        case ReflectionResourceType::ShaderAccess::ReadWrite:
            return uavType;
        default:
            should_not_get_here();
            return DescriptorSet::Type::Count;
        }
    }

    void ParameterBlock::setResourceSrvUavCommon(std::string name, uint32_t descOffset, DescriptorSet::Type type, const Resource::SharedPtr& pResource, const std::string& funcName)
    {
        uint32_t index;
        while (parseArrayIndex(name, name, index)) {};

        ParameterBlockReflection::BindLocation bindLoc = mpReflector->getResourceBinding(name);
        if (checkResourceIndices(bindLoc, descOffset, type, funcName) == false) return;
        auto& desc = mAssignedResources[bindLoc.setIndex][bindLoc.rangeIndex][descOffset];
        desc.pResource = pResource;

        switch (type)
        {
        case DescriptorSet::Type::TextureSrv:
        case DescriptorSet::Type::TypedBufferSrv:
        case DescriptorSet::Type::StructuredBufferSrv:
            desc.pSRV = pResource ? pResource->getSRV() : ShaderResourceView::getNullView();
            break;
        case DescriptorSet::Type::TextureUav:
        case DescriptorSet::Type::TypedBufferUav:
        case DescriptorSet::Type::StructuredBufferUav:
            desc.pUAV = pResource ? pResource->getUAV() : UnorderedAccessView::getNullView();
            break;
        default:
            should_not_get_here();
        }
        mRootSets[bindLoc.setIndex].pSet = nullptr;
    }

    template<typename ResourceType>
    typename ResourceType::SharedPtr ParameterBlock::getResourceSrvUavCommon(const std::string& name, uint32_t descOffset, DescriptorSet::Type type, const std::string& funcName) const
    {
        ParameterBlockReflection::BindLocation bindLoc = mpReflector->getResourceBinding(name);
        if (checkResourceIndices(bindLoc, descOffset, type, funcName) == false) return nullptr;
        auto& desc = mAssignedResources[bindLoc.setIndex][bindLoc.rangeIndex][descOffset];
        return std::dynamic_pointer_cast<ResourceType>(desc.pResource);
    }

    bool ParameterBlock::setRawBuffer(const std::string& name, Buffer::SharedPtr pBuf)
    {
        // Find the buffer
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);
#if _LOG_ENABLED

        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::RawBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "setRawBuffer()") == false)
        {
            return false;
        }
#endif
        DescriptorSet::Type type = getSetTypeFromVar(pVar, DescriptorSet::Type::TextureSrv, DescriptorSet::Type::TextureUav);
        setResourceSrvUavCommon(name, pVar->getDescOffset(), type, pBuf, "setRawBuffer()");
        return true;
    }

    bool ParameterBlock::setTypedBuffer(const std::string& name, TypedBufferBase::SharedPtr pBuf)
    {
        // Find the buffer
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);
#if _LOG_ENABLED
        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::TypedBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "setTypedBuffer()") == false)
        {
            return false;
        }
#endif
        DescriptorSet::Type type = getSetTypeFromVar(pVar, DescriptorSet::Type::TypedBufferSrv, DescriptorSet::Type::TypedBufferUav);
        setResourceSrvUavCommon(name, pVar->getDescOffset(), type, pBuf, "setTypedBuffer()");
        return true;
    }

    bool ParameterBlock::setStructuredBuffer(const std::string& name, StructuredBuffer::SharedPtr pBuf)
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);
#if _LOG_ENABLED
        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::StructuredBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "setStructuredBuffer()") == false)
        {
            return false;
        }
#endif
        DescriptorSet::Type type = getSetTypeFromVar(pVar, DescriptorSet::Type::StructuredBufferSrv, DescriptorSet::Type::StructuredBufferUav);
        setResourceSrvUavCommon(name, pVar->getDescOffset(), type, pBuf, "setStructuredBuffer()");
        return true;
    }

    Buffer::SharedPtr ParameterBlock::getRawBuffer(const std::string& name) const
    {
        // Find the buffer
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);
#if _LOG_ENABLED
        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::RawBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "getRawBuffer()") == false)
        {
            return nullptr;
        }
#endif
        DescriptorSet::Type type = getSetTypeFromVar(pVar, DescriptorSet::Type::TextureSrv, DescriptorSet::Type::TextureUav);
        return getResourceSrvUavCommon<Buffer>(name, pVar->getDescOffset(), type, "getRawBuffer()");
    }

    TypedBufferBase::SharedPtr ParameterBlock::getTypedBuffer(const std::string& name) const
    {
        // Find the buffer
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);
#if _LOG_ENABLED
        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::TypedBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "getTypedBuffer()") == false)
        {
            return nullptr;
        }
#endif
        DescriptorSet::Type type = getSetTypeFromVar(pVar, DescriptorSet::Type::TypedBufferSrv, DescriptorSet::Type::TypedBufferUav);
        return getResourceSrvUavCommon<TypedBufferBase>(name, pVar->getDescOffset(), type, "getTypedBuffer()");
    }

    StructuredBuffer::SharedPtr ParameterBlock::getStructuredBuffer(const std::string& name) const
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

#if _LOG_ENABLED
        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::StructuredBuffer, ReflectionResourceType::ShaderAccess::Undefined, true, name, "getStructuredBuffer()") == false)
        {
            return nullptr;
        }
#endif   
        DescriptorSet::Type type = getSetTypeFromVar(pVar, DescriptorSet::Type::StructuredBufferSrv, DescriptorSet::Type::StructuredBufferUav);
        return getResourceSrvUavCommon<StructuredBuffer>(name, pVar->getDescOffset(), type, "getTypedBuffer()");
    }

    bool ParameterBlock::setSampler(const BindLocation& bindLocation, uint32_t arrayIndex, const Sampler::SharedPtr& pSampler)
    {
        if (checkResourceIndices(bindLocation, arrayIndex, DescriptorSet::Type::Sampler, "setSampler()") == false) return false;
        mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex][arrayIndex].pSampler = pSampler ? pSampler : Sampler::getDefault();
        mRootSets[bindLocation.setIndex].pSet = nullptr;
        return true;
    }

    bool ParameterBlock::setSampler(const std::string& name, const Sampler::SharedPtr& pSampler)
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);
#if _LOG_ENABLED
        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Sampler, ReflectionResourceType::ShaderAccess::Read, false, name, "setSampler()") == false)
        {
            return false;
        }
#endif
        ParameterBlockReflection::BindLocation bind = mpReflector->getResourceBinding(name);
        return setSampler(bind, pVar->getDescOffset(), pSampler);
    }

    Sampler::SharedPtr ParameterBlock::getSampler(const std::string& name) const
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);
#if _LOG_ENABLED
        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Sampler, ReflectionResourceType::ShaderAccess::Read, false, name, "getSampler()") == false)
        {
            return nullptr;
        }
#endif
        ParameterBlockReflection::BindLocation bind = mpReflector->getResourceBinding(name);
        return getSampler(bind, pVar->getDescOffset());
    }

    Sampler::SharedPtr ParameterBlock::getSampler(const BindLocation& bindLocation, uint32_t arrayIndex) const
    {
        if (checkResourceIndices(bindLocation, arrayIndex, DescriptorSet::Type::Sampler, "getSampler()") == false) return nullptr;
        return mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex][arrayIndex].pSampler;
    }

    ShaderResourceView::SharedPtr ParameterBlock::getSrv(const BindLocation& bindLocation, uint32_t arrayIndex) const
    {
        if (checkResourceIndices(bindLocation, arrayIndex, DescriptorSet::Type::Sampler, "getSrv()") == false) return nullptr;
        return mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex][arrayIndex].pSRV;
    }

    UnorderedAccessView::SharedPtr ParameterBlock::getUav(const BindLocation& bindLocation, uint32_t arrayIndex) const
    {
        if (checkResourceIndices(bindLocation, arrayIndex, DescriptorSet::Type::Sampler, "getUav()") == false) return nullptr;
        return mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex][arrayIndex].pUAV;
    }

    bool ParameterBlock::setTexture(const std::string& name, const Texture::SharedPtr& pTexture)
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);

#if _LOG_ENABLED
        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Texture, ReflectionResourceType::ShaderAccess::Undefined, false, name, "setTexture()") == false)
        {
            return false;
        }
#endif

        DescriptorSet::Type type = getSetTypeFromVar(pVar, DescriptorSet::Type::TextureSrv, DescriptorSet::Type::TextureUav);
        setResourceSrvUavCommon(name, pVar->getDescOffset(), type, pTexture, "setTexture()");
        return true;
    }

    Texture::SharedPtr ParameterBlock::getTexture(const std::string& name) const
    {
        const ReflectionVar::SharedConstPtr pVar = mpReflector->getResource(name);
#if _LOG_ENABLED
        if (verifyResourceVar(pVar.get(), ReflectionResourceType::Type::Texture, ReflectionResourceType::ShaderAccess::Undefined, false, name, "getTexture()") == false)
        {
            return nullptr;
        }
#endif
        DescriptorSet::Type type = getSetTypeFromVar(pVar, DescriptorSet::Type::TextureSrv, DescriptorSet::Type::TextureUav);
        return getResourceSrvUavCommon<Texture>(name, pVar->getDescOffset(), type, "getTexture()");
    }

    template<typename ViewType>
    Resource::SharedPtr getResourceFromView(const ViewType* pView)
    {
        return (pView) ? const_cast<Resource*>(pView->getResource())->shared_from_this() : nullptr;
    }

    bool ParameterBlock::setSrv(const BindLocation& bindLocation, uint32_t arrayIndex, const ShaderResourceView::SharedPtr& pSrv)
    {
        if (checkResourceIndices(bindLocation, arrayIndex, DescriptorSet::Type::Count, "setSrv()") == false) return false;
        auto& desc = mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex][arrayIndex];
#if _LOG_ENABLED
        if (desc.pSRV == nullptr)
        {
            logWarning("Can't find SRV with set index " + std::to_string(bindLocation.setIndex) + ", range index " + std::to_string(bindLocation.rangeIndex) + ", array index " + std::to_string(arrayIndex) + ". Ignoring setSrv() call");
            return false;
        }
#endif
        desc.pSRV = pSrv ? pSrv : ShaderResourceView::getNullView();
        desc.pResource = getResourceFromView(pSrv.get());
        mRootSets[bindLocation.setIndex].pSet = nullptr;
        return true;
    }

    bool ParameterBlock::setUav(const BindLocation& bindLocation, uint32_t arrayIndex, const UnorderedAccessView::SharedPtr& pUav)
    {
        if (checkResourceIndices(bindLocation, arrayIndex, DescriptorSet::Type::Count, "setUav()") == false) return false;
        auto& desc = mAssignedResources[bindLocation.setIndex][bindLocation.rangeIndex][arrayIndex];
#if _LOG_ENABLED
        if (desc.pUAV == nullptr)
        {
            logWarning("Can't find UAV with set index " + std::to_string(bindLocation.setIndex) + ", range index " + std::to_string(bindLocation.rangeIndex) + ", array index " + std::to_string(arrayIndex) + ". Ignoring setUav() call");
            return false;
        }
#endif
        desc.pUAV = pUav ? pUav : UnorderedAccessView::getNullView();
        desc.pResource = getResourceFromView(pUav.get());
        mRootSets[bindLocation.setIndex].pSet = nullptr;
        return true;
    }
    
    static bool isUavSetType(DescriptorSet::Type type)
    {
        switch (type)
        {
        case DescriptorSet::Type::StructuredBufferUav:
        case DescriptorSet::Type::TextureUav:
        case DescriptorSet::Type::TypedBufferUav:
            return true;
        default:
            return false;
        }
    }

    static bool prepareResource(CopyContext* pContext, Resource* pResource, DescriptorSet::Type type)
    {
        if (!pResource) return false;

        ConstantBuffer* pCB = dynamic_cast<ConstantBuffer*>(pResource);
        if (pCB) return pCB->uploadToGPU();

        bool dirty = false;
        bool isUav = isUavSetType(type);

        // If it's a typed buffer, upload it to the GPU
        TypedBufferBase* pTypedBuffer = dynamic_cast<TypedBufferBase*>(pResource);
        if (pTypedBuffer)
        {
            dirty = pTypedBuffer->uploadToGPU();
        }
        StructuredBuffer* pStructured = dynamic_cast<StructuredBuffer*>(pResource);
        if (pStructured)
        {
            dirty = pStructured->uploadToGPU();

            if (isUav && pStructured->hasUAVCounter())
            {
                pContext->resourceBarrier(pStructured->getUAVCounter().get(), Resource::State::UnorderedAccess);
            }
        }

        pContext->resourceBarrier(pResource, isUav ? Resource::State::UnorderedAccess : Resource::State::ShaderResource);
        if (isUav)
        {
            if (pTypedBuffer) pTypedBuffer->setGpuCopyDirty();
            if (pStructured)  pStructured->setGpuCopyDirty();
        }
        return dirty;
    }

    bool ParameterBlock::prepareForDraw(CopyContext* pContext)
    {
        // Prepare the resources
        for (size_t s = 0; s < mAssignedResources.size(); s++)
        {
            const auto& set = mAssignedResources[s];
            for (const auto& range : set)
            {
                for (const auto& desc : range)
                {
                    if (prepareResource(pContext, desc.pResource.get(), desc.type))
                    {
                        mRootSets[s].pSet = nullptr;
                    }
                }
            }
        }

        // Allocate the missing sets
        for (uint32_t i = 0; i < mRootSets.size(); i++)
        {
            mRootSets[i].dirty = (mRootSets[i].pSet == nullptr);
            if (mRootSets[i].pSet == nullptr)
            {
                DescriptorSet::Layout layout;
                const auto& set = mpReflector->getDescriptorSetLayouts()[i];
                mRootSets[i].pSet = DescriptorSet::create(gpDevice->getGpuDescriptorPool(), set);
                if (mRootSets[i].pSet == nullptr)
                {
                    return false;
                }
            }
        }

        // bind the resources
        for (uint32_t s = 0; s < mAssignedResources.size(); s++)
        {
            if (mRootSets[s].dirty == false) continue;
            const auto& pDescSet = mRootSets[s].pSet;

            const auto& set = mAssignedResources[s];
            for (uint32_t r = 0 ; r < set.size() ; r++)
            {
                const auto& range = set[r];
                for (uint32_t d = 0; d < range.size(); d++)
                {
                    const auto& desc = range[d];
                    switch (desc.type)
                    {
                    case DescriptorSet::Type::Cbv:
                    {
                        ConstantBuffer* pCB = dynamic_cast<ConstantBuffer*>(desc.pResource.get());
                        ConstantBufferView::SharedPtr pView = pCB ? pCB->getCbv() : ConstantBufferView::getNullView();
                        pDescSet->setCbv(r, d, pView);
                    }
                    break;
                    case DescriptorSet::Type::Sampler:
                        assert(desc.pSampler);
                        pDescSet->setSampler(r, d, desc.pSampler.get());
                        break;
                    case DescriptorSet::Type::StructuredBufferSrv:
                    case DescriptorSet::Type::TypedBufferSrv:
                    case DescriptorSet::Type::TextureSrv:
                        assert(desc.pSRV);
                        pDescSet->setSrv(r, d, desc.pSRV.get());
                        break;
                    case DescriptorSet::Type::StructuredBufferUav:
                    case DescriptorSet::Type::TypedBufferUav:
                    case DescriptorSet::Type::TextureUav:
                        assert(desc.pUAV);
                        pDescSet->setUav(r, d, desc.pUAV.get());
                        break;

                    default:
                        should_not_get_here();
                    }
                }
            }
        }
         return true;
    }
}