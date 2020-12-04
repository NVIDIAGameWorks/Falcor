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
#include "ParameterBlock.h"
#include "Utils/StringUtils.h"
#include "Core/API/CopyContext.h"
#include "Core/API/Device.h"

#include <slang/slang.h>

namespace Falcor
{
    namespace
    {
        std::string getErrorPrefix(const char* funcName, const std::string& varName)
        {
            return std::string(funcName) + " is trying to access the shader variable '" + varName + "'";
        }

        const ReflectionResourceType* getResourceReflection(const ShaderVar& var, const Resource* pResource, const std::string& varName, const char* funcName)
        {
            if (!var.isValid())
            {
                logError("'" + varName + "' was not found. Ignoring '" + funcName + "' call.");
                return nullptr;
            }

            const ReflectionResourceType* pType = var.getType()->unwrapArray()->asResourceType();
            if (!pType)
            {
                logError("'" + varName + "' is not a resource. Ignoring '" + funcName + "' call.");
                return nullptr;
            }

#if _LOG_ENABLED
            ResourceBindFlags requiredFlag = ResourceBindFlags::None;
            if(pType->getType() != ReflectionResourceType::Type::Sampler)
            {
                switch (pType->getShaderAccess())
                {
                case ReflectionResourceType::ShaderAccess::Read:
                    requiredFlag = (pType->getType() == ReflectionResourceType::Type::ConstantBuffer) ? ResourceBindFlags::Constant : ResourceBindFlags::ShaderResource;
                    break;
                case ReflectionResourceType::ShaderAccess::ReadWrite:
                    requiredFlag = ResourceBindFlags::UnorderedAccess;
                    break;
                default:
                    should_not_get_here();
                }

                if (pResource && !is_set(pResource->getBindFlags(), requiredFlag))
                {
                    logError(getErrorPrefix(funcName, varName) + ", but the resource is missing the " + to_string(requiredFlag) + " bind-flag");
                    return nullptr;
                }
            }
#endif
            return pType;
        }

        bool verifyTextureVar(const ShaderVar& var, const Texture* pTexture, const std::string& varName, const char* funcName)
        {
            auto pType = getResourceReflection(var, pTexture, varName, funcName);
            if (!pType) return false;

#if _LOG_ENABLED
            if (pType->getType() != ReflectionResourceType::Type::Texture)
            {
                logError(getErrorPrefix(funcName, varName) + ", but the variable was declared in the shader as a " + to_string(pType->getType()) + " and not as a texture");
                return false;
            }
            assert(pType->getDimensions() != ReflectionResourceType::Dimensions::Buffer);
#endif
            return true;
        }

        bool verifySamplerVar(const ShaderVar& var, const std::string& varName, const char* funcName)
        {
            auto pType = getResourceReflection(var, nullptr, varName, funcName);
            if (!pType) return false;

#if _LOG_ENABLED
            if (pType->getType() != ReflectionResourceType::Type::Sampler)
            {
                logError(getErrorPrefix(funcName, varName) + ", but the variable was declared in the shader as a " + to_string(pType->getType()) + " and not as a sampler");
                return false;
            }
#endif
            return true;
        }

        bool verifyBufferVar(const ShaderVar& var, const Buffer* pBuffer, const std::string& varName, const char* funcName)
        {
            auto pType = getResourceReflection(var, pBuffer, varName, funcName);
            if (!pType) return false;

#if _LOG_ENABLED
            switch (pType->getType())
            {
            case ReflectionResourceType::Type::ConstantBuffer:
            case ReflectionResourceType::Type::RawBuffer:
                break; // Already verified in getResourceReflection()
            case ReflectionResourceType::Type::StructuredBuffer:
                if (pBuffer && !pBuffer->isStructured())
                {
                    logError(getErrorPrefix(funcName, varName) + ", but the variable is a StructuredBuffer and the buffer wasn't create as one");
                    return false;
                }
                break;
            case ReflectionResourceType::Type::TypedBuffer:
                if (pBuffer && !pBuffer->isTyped())
                {
                    logError(getErrorPrefix(funcName, varName) + ", but the variable is a TypedBuffer and the buffer wasn't create as one");
                    return false;
                }
                break;
            default:
                logError(getErrorPrefix(funcName, varName) + ", but the variable '" + varName + "' was declared in the shader as a " + to_string(pType->getType()) + " and not as a buffer");
                return false;
            }

            assert(pType->getDimensions() == ReflectionResourceType::Dimensions::Buffer);
#endif
            return true;
        }

        template<typename ViewType>
        Resource::SharedPtr getResourceFromView(const ViewType* pView)
        {
            if (!pView) return nullptr;
            auto pResource = pView->getResource();
            if (!pResource) return nullptr;
            return pResource->shared_from_this();
        }

        const std::array<DescriptorSet::Type, 4> kRootSrvDescriptorTypes =
        {
            DescriptorSet::Type::RawBufferSrv,
            DescriptorSet::Type::TypedBufferSrv,
            DescriptorSet::Type::StructuredBufferSrv,
            DescriptorSet::Type::AccelerationStructureSrv,
        };

        const std::array<DescriptorSet::Type, 3> kRootUavDescriptorTypes =
        {
            DescriptorSet::Type::RawBufferUav,
            DescriptorSet::Type::TypedBufferUav,
            DescriptorSet::Type::StructuredBufferUav,
        };

        const std::array<DescriptorSet::Type, 5> kSrvDescriptorTypes =
        {
            DescriptorSet::Type::TextureSrv,
            DescriptorSet::Type::RawBufferSrv,
            DescriptorSet::Type::TypedBufferSrv,
            DescriptorSet::Type::StructuredBufferSrv,
            DescriptorSet::Type::AccelerationStructureSrv,
        };

        const std::array<DescriptorSet::Type, 4> kUavDescriptorTypes =
        {
            DescriptorSet::Type::TextureUav,
            DescriptorSet::Type::RawBufferUav,
            DescriptorSet::Type::TypedBufferUav,
            DescriptorSet::Type::StructuredBufferUav,
        };

        const std::array<DescriptorSet::Type, 1> kCbvDescriptorType = { DescriptorSet::Type::Cbv };
        const std::array<DescriptorSet::Type, 1> kSamplerDescriptorType = { DescriptorSet::Type::Sampler };

        template<size_t N>
        bool isSetType(DescriptorSet::Type type, const std::array<DescriptorSet::Type, N>& allowedTypes)
        {
            for (auto t : allowedTypes)
                if (type == t) return true;
            return false;
        }

        bool isSrvType(const ReflectionType::SharedConstPtr& pType)
        {
            auto resourceType = pType->unwrapArray()->asResourceType();
            if (resourceType->getType() == ReflectionResourceType::Type::Sampler ||
                resourceType->getType() == ReflectionResourceType::Type::ConstantBuffer) return false;

            switch (resourceType->getShaderAccess())
            {
            case ReflectionResourceType::ShaderAccess::Read:
                return true;
            case ReflectionResourceType::ShaderAccess::ReadWrite:
                return false;
            default:
                should_not_get_here();
                return false;
            }
        }

        bool isUavType(const ReflectionType::SharedConstPtr& pType)
        {
            auto resourceType = pType->unwrapArray()->asResourceType();
            if (resourceType->getType() == ReflectionResourceType::Type::Sampler ||
                resourceType->getType() == ReflectionResourceType::Type::ConstantBuffer) return false;

            switch (resourceType->getShaderAccess())
            {
            case ReflectionResourceType::ShaderAccess::Read:
                return false;
            case ReflectionResourceType::ShaderAccess::ReadWrite:
                return true;
            default:
                should_not_get_here();
                return false;
            }
        }

        bool isCbvType(const ReflectionType::SharedConstPtr& pType)
        {
            auto resourceType = pType->unwrapArray()->asResourceType();
            if (resourceType->getType() == ReflectionResourceType::Type::ConstantBuffer)
            {
                assert(resourceType->getShaderAccess() == ReflectionResourceType::ShaderAccess::Read);
                return true;
            }
            return false;
        }
    }

    ParameterBlock::~ParameterBlock()
    {
#if _ENABLE_CUDA
        if (mUnderlyingCUDABuffer.kind == CUDABufferKind::Host)
        {
            if (auto pData = mUnderlyingCUDABuffer.pData)
            {
                free(pData);
            }
        }
#endif
    }

    ParameterBlock::SharedPtr ParameterBlock::create(const std::shared_ptr<const ProgramVersion>& pProgramVersion, const ReflectionType::SharedConstPtr& pElementType)
    {
        if (!pElementType) throw std::exception("Can't create a parameter block without type information");
        auto pReflection = ParameterBlockReflection::create(pProgramVersion.get(), pElementType);
        return create(pReflection);
    }

    ParameterBlock::SharedPtr ParameterBlock::create(const ParameterBlockReflection::SharedConstPtr& pReflection)
    {
        assert(pReflection);
        return SharedPtr(new ParameterBlock(pReflection->getProgramVersion(), pReflection));
    }

    ParameterBlock::SharedPtr ParameterBlock::create(const std::shared_ptr<const ProgramVersion>& pProgramVersion, const std::string& typeName)
    {
        assert(pProgramVersion);
        return ParameterBlock::create(pProgramVersion, pProgramVersion->getReflector()->findType(typeName));
    }

    ParameterBlock::ParameterBlock(const std::shared_ptr<const ProgramVersion>& pProgramVersion, const ParameterBlockReflection::SharedConstPtr& pReflection)
        : mpReflector(pReflection)
        , mpProgramVersion(pProgramVersion)
        , mData(pReflection->getElementType()->getByteSize(), 0)
    {
        ReflectionStructType::BuildState state;
        auto pElementType = getElementType();
        assert(pElementType);

        // TODO: this counting should move to `ParameterBlockReflection`
        auto rangeCount = pElementType->getResourceRangeCount();
        for(uint32_t rr = 0; rr < rangeCount; ++rr)
        {
            auto range = pElementType->getResourceRange(rr);

            switch(range.descriptorType)
            {
            case DescriptorSet::Type::Cbv:
                state.cbCount += range.count;
                break;
            case DescriptorSet::Type::TextureSrv:
            case DescriptorSet::Type::RawBufferSrv:
            case DescriptorSet::Type::TypedBufferSrv:
            case DescriptorSet::Type::StructuredBufferSrv:
            case DescriptorSet::Type::AccelerationStructureSrv:
                state.srvCount += range.count;
                break;
            case DescriptorSet::Type::TextureUav:
            case DescriptorSet::Type::RawBufferUav:
            case DescriptorSet::Type::TypedBufferUav:
            case DescriptorSet::Type::StructuredBufferUav:
                state.uavCount += range.count;
                break;
            case DescriptorSet::Type::Sampler:
                state.samplerCount += range.count;
                break;
            case DescriptorSet::Type::Dsv:
            case DescriptorSet::Type::Rtv:
                break;
            default:
                should_not_get_here();
                break;
            }
        }

        mParameterBlocks.resize(state.cbCount);
        mSRVs.resize(state.srvCount);
        mUAVs.resize(state.uavCount);
        mSamplers.resize(state.samplerCount);
        mSets.resize(pReflection->getDescriptorSetCount());

        createConstantBuffers(getRootVar());
    }

    void ParameterBlock::createConstantBuffers(const ShaderVar& var)
    {
        auto pType = var.getType();
        if(pType->getResourceRangeCount() == 0) return;

        switch(pType->getKind())
        {
        case ReflectionType::Kind::Struct:
            {
                auto pStructType = pType->asStructType();
                uint32_t memberCount = pStructType->getMemberCount();
                for(uint32_t i = 0; i < memberCount; ++i) createConstantBuffers(var[i]);
            }
            break;
        case ReflectionType::Kind::Resource:
            {
                auto pResourceType = pType->asResourceType();
                switch(pResourceType->getType())
                {
                case ReflectionResourceType::Type::ConstantBuffer:
                    {
                        auto pCB = ParameterBlock::create(pResourceType->getParameterBlockReflector());
                        var.setParameterBlock(pCB);
                    }
                    break;

                default:
                    break;
                }
            }
            break;

        default:
            break;
        }
    }

    ShaderVar ParameterBlock::getRootVar() const
    {
        return ShaderVar(const_cast<ParameterBlock*>(this));
    }

    ShaderVar ParameterBlock::findMember(const std::string& varName) const
    {
        return getRootVar().findMember(varName);
    }

    ShaderVar ParameterBlock::findMember(uint32_t index) const
    {
        return getRootVar().findMember(index);
    }

    static ShaderVar getBufferBindLocation(ParameterBlock const* pObject, const std::string& name, ReflectionResourceType::Type bufferType)
    {
        auto var = pObject->getRootVar()[name];
        if (!var.isValid())
        {
            logError("Couldn't find a " + to_string(bufferType) + " named " + name);
            return nullptr;
        }
        else if (var.getType()->unwrapArray()->asResourceType()->getType() != bufferType)
        {
            logError("Found a variable named '" + name + "' but it is not a " + to_string(bufferType));
            return nullptr;
        }

        return var;
    }

    bool ParameterBlock::checkResourceIndices(const BindLocation& bindLocation, const char* funcName) const
    {
#if _LOG_ENABLED
        auto rangeIndex = bindLocation.getResourceRangeIndex();
        auto arrayIndex = bindLocation.getResourceArrayIndex();
        auto pElementType = getElementType();

        if(rangeIndex >= pElementType->getResourceRangeCount())
        {
            logError("Can't find resource at range index " + std::to_string(rangeIndex) + ". Ignoring " + funcName + " call");
            return false;
        }

        auto& range = pElementType->getResourceRange(rangeIndex);
        if( arrayIndex >= range.count )
        {
            logError("Resource at range index " + std::to_string(rangeIndex)
                + " has a count of " + std::to_string(range.count)
                + ", so array index " + std::to_string(arrayIndex) + " is out of range. Ignoring " + funcName + " call");
            return false;
        }
#endif
        return true;
    }

    template<size_t N>
    bool ParameterBlock::checkDescriptorType(const BindLocation& bindLocation, const std::array<DescriptorSet::Type, N>& allowedTypes, const char* funcName) const
    {
#if _LOG_ENABLED
        auto rangeIndex = bindLocation.getResourceRangeIndex();
        auto& resourceRange = mpReflector->getResourceRange(rangeIndex);

        if (!isSetType(resourceRange.descriptorType, allowedTypes))
        {
            logError("Resource at range index " + std::to_string(rangeIndex) + ", array index " + std::to_string(bindLocation.getResourceArrayIndex()) + " has non-matching descriptor type. Ignoring " + funcName + " call");
            return false;
        }
#endif
        return true;
    }

    bool ParameterBlock::checkDescriptorSrvUavCommon(
        const BindLocation& bindLocation,
        const Resource::SharedPtr& pResource,
        const std::variant<ShaderResourceView::SharedPtr, UnorderedAccessView::SharedPtr>& pView,
        const char* funcName) const
    {
#if _LOG_ENABLED
        if (!checkResourceIndices(bindLocation, funcName)) return false;

        const auto& bindingInfo = mpReflector->getResourceRangeBindingInfo(bindLocation.getResourceRangeIndex());
        bool isUav = std::holds_alternative<UnorderedAccessView::SharedPtr>(pView);

        if (bindingInfo.isDescriptorSet())
        {
            if (!(isUav ? checkDescriptorType(bindLocation, kUavDescriptorTypes, funcName) : checkDescriptorType(bindLocation, kSrvDescriptorTypes, funcName))) return false;
            // TODO: Check that resource type/dimension matches the descriptor type.
        }
        else if (bindingInfo.isRootDescriptor())
        {
            if (!(isUav ? checkDescriptorType(bindLocation, kRootUavDescriptorTypes, funcName) : checkDescriptorType(bindLocation, kRootSrvDescriptorTypes, funcName))) return false;

            // For root descriptors, also check that the resource is compatible.
            if (!checkRootDescriptorResourceCompatibility(pResource, funcName)) return false;

            // TODO: Check that view points to the start of the buffer.
            // We bind resources to root descriptor by base address, so an offset of zero is assumed.
        }
        else
        {
            logError("Resource at range index " + std::to_string(bindLocation.getResourceRangeIndex()) + ", array index " + std::to_string(bindLocation.getResourceArrayIndex()) + " is not a descriptor. Ignoring " + funcName + " call.");
            return false;
        }
#endif
        return true;
    }

    bool ParameterBlock::checkRootDescriptorResourceCompatibility(const Resource::SharedPtr& pResource, const std::string& funcName) const
    {
        if (!pResource) return true;
#if _LOG_ENABLED
        if (pResource->getType() != Resource::Type::Buffer)
        {
            logError("Resource bound to root descriptor must be a buffer. Ignoring " + funcName + " call.");
            return false;
        }
        auto pBuffer = pResource->asBuffer();
        assert(pBuffer);

        // Check that typed buffer has 32-bit float/uint/sint format.
        // There is no format conversion for buffers accessed through root descriptors.
        if (pBuffer->isTyped())
        {
            switch (pBuffer->getFormat())
            {
            case ResourceFormat::R32Float:
            case ResourceFormat::RG32Float:
            case ResourceFormat::RGB32Float:
            case ResourceFormat::RGBA32Float:
            case ResourceFormat::R32Int:
            case ResourceFormat::R32Uint:
            case ResourceFormat::RG32Int:
            case ResourceFormat::RG32Uint:
            case ResourceFormat::RGB32Int:
            case ResourceFormat::RGB32Uint:
            case ResourceFormat::RGBA32Int:
            case ResourceFormat::RGBA32Uint:
                break;
            default:
                logError("Typed buffer bound to root descriptor must have 32-bit float/uint/sint format. Ignoring " + funcName + "  call.");
                return false;
            }
        }

        // Check that buffer does not have a counter. UAV counters are unsupported with root descriptors.
        if (pBuffer->getUAVCounter())
        {
            logError("Buffer bound to root descriptor must not have a UAV counter. Ignoring " + funcName + "  call.");
            return false;
        }
#endif
        return true;
    }

    size_t ParameterBlock::getFlatIndex(const BindLocation& bindLocation) const
    {
        auto rangeIndex = bindLocation.getResourceRangeIndex();
        auto arrayIndex = bindLocation.getResourceArrayIndex();
        return getElementType()->getResourceRange(rangeIndex).baseIndex + arrayIndex;
    }

    bool ParameterBlock::setParameterBlock(const std::string& name, const ParameterBlock::SharedPtr& pCB)
    {
        auto var = getBufferBindLocation(this, name, ReflectionResourceType::Type::ConstantBuffer);
        if(!var.isValid())
        {
            logError("Parameter block '" + name + "' was not found. Ignoring setParameterBlock() call.");
            return false;
        }
        return var.setParameterBlock(pCB);
    }

    uint32_t ParameterBlock::getDescriptorSetIndex(const BindLocation& bindLocation)
    {
        return mpReflector->getResourceRangeBindingInfo(bindLocation.getResourceRangeIndex()).descriptorSetIndex;
    }

    void ParameterBlock::markDescriptorSetDirty(const BindLocation& bindLocation)
    {
        markDescriptorSetDirty(getDescriptorSetIndex(bindLocation));
    }

    void ParameterBlock::markDescriptorSetDirty(uint32_t index) const
    {
        if (index == ParameterBlockReflection::kInvalidIndex) return;
        assert(index < mSets.size());

        auto epoch = mEpochOfLastChange++;
        mSets[index].pSet = nullptr;
        mSets[index].epochOfLastChange = epoch;
    }

    void ParameterBlock::markUniformDataDirty() const
    {
        auto epoch = mEpochOfLastChange++;
        mEpochOfLastUniformDataChange = epoch;
        mUnderlyingConstantBuffer.epochOfLastObservedChange = epoch-1;
        mUnderlyingConstantBuffer.pCBV = nullptr;

        // When the underlying constant buffer is invalidated, that
        // also means that whatever descriptor set the constant buffer
        // gets bound through (if one exists) should also be invalidated.
        //
        markDescriptorSetDirty(mpReflector->getDefaultConstantBufferBindingInfo().descriptorSetIndex);
    }

    bool ParameterBlock::setParameterBlock(const BindLocation& bindLocation, const ParameterBlock::SharedPtr& pBlock)
    {
        if (!checkResourceIndices(bindLocation, "setParameterBlock")) return false;
        if (!checkDescriptorType(bindLocation, kCbvDescriptorType, "setParameterBlock")) return false;
        auto& assigned = mParameterBlocks[getFlatIndex(bindLocation)];

#if 0
#if _LOG_ENABLED
        if (res.requiredSize > pBlock->getSize())
        {
            logError("Can't attach the constant buffer. Size mismatch.");
            return false;
        }
#endif
#endif
        if (assigned.pBlock == pBlock) return true;
        assigned.pBlock = pBlock;
        assigned.epochOfLastObservedChange = pBlock ? pBlock->mEpochOfLastChange : 0;
        return true;
    }

    ParameterBlock::SharedPtr ParameterBlock::getParameterBlock(const std::string& name) const
    {
        auto var = getBufferBindLocation(this, name, ReflectionResourceType::Type::ConstantBuffer);
        if (!var.isValid())
        {
            logError("Parameter block '" + name + "' was not found. Ignoring getParameterBlock() call.");
            return nullptr;
        }
        return var.getParameterBlock();
    }

    ParameterBlock::SharedPtr ParameterBlock::getParameterBlock(const BindLocation& bindLocation) const
    {
        if (!checkResourceIndices(bindLocation, "getParameterBlock")) return nullptr;
        if (!checkDescriptorType(bindLocation, kCbvDescriptorType, "getParameterBlock")) return nullptr;
        return mParameterBlocks[getFlatIndex(bindLocation)].pBlock;
    }

    bool ParameterBlock::setResourceSrvUavCommon(const BindLocation& bindLoc, const Resource::SharedPtr& pResource, const char* funcName)
    {
        // Check if the bind location is a root descriptor or a descriptor set.
        // If binding to a descriptor set, we'll create a default view for the resource. For root descriptors, we don't need a view.
        const auto& bindingInfo = mpReflector->getResourceRangeBindingInfo(bindLoc.getResourceRangeIndex());
        const bool isRoot = bindingInfo.isRootDescriptor();
        const size_t flatIndex = getFlatIndex(bindLoc);

        const ReflectionResourceType* pResouceReflection = bindLoc.getType()->asResourceType();
        assert(pResouceReflection && pResouceReflection->getDimensions() == bindingInfo.dimension);

        if (isUavType(bindLoc.getType()))
        {
            auto pUAV = (pResource && !isRoot) ? pResource->getUAV() : UnorderedAccessView::getNullView(bindingInfo.dimension);
            if (!checkDescriptorSrvUavCommon(bindLoc, pResource, pUAV, funcName)) return false;
            auto& assignedUAV = mUAVs[flatIndex];
            if (assignedUAV.pResource == pResource && assignedUAV.pView == pUAV) return true;
            assignedUAV.pView = pUAV;
            assignedUAV.pResource = pResource;
        }
        else if (isSrvType(bindLoc.getType()))
        {
            auto pSRV = (pResource && !isRoot) ? pResource->getSRV() : ShaderResourceView::getNullView(bindingInfo.dimension);
            if (!checkDescriptorSrvUavCommon(bindLoc, pResource, pSRV, funcName)) return false;
            auto& assignedSRV = mSRVs[flatIndex];
            if (assignedSRV.pResource == pResource && assignedSRV.pView == pSRV) return true;
            assignedSRV.pView = pSRV;
            assignedSRV.pResource = pResource;
        }
        else
        {
            logError("Error trying to bind resource to non SRV/UAV variable. Ignoring call.");
            return false;
        }

        markDescriptorSetDirty(bindLoc);
        return true;
    }

    Resource::SharedPtr ParameterBlock::getResourceSrvUavCommon(const BindLocation& bindLoc, const char* funcName) const
    {
        if (!checkResourceIndices(bindLoc, funcName)) return nullptr;
        size_t flatIndex = getFlatIndex(bindLoc);

        if (isUavType(bindLoc.getType()))
        {
            if (!checkDescriptorType(bindLoc, kUavDescriptorTypes, funcName)) return nullptr;
            return mUAVs[flatIndex].pResource;
        }
        else if (isSrvType(bindLoc.getType()))
        {
            if (!checkDescriptorType(bindLoc, kSrvDescriptorTypes, funcName)) return nullptr;
            return mSRVs[flatIndex].pResource;
        }
        else
        {
            logError("Error trying to get resource from non SRV/UAV variable. Ignoring call.");
            return nullptr;
        }

        should_not_get_here();
        return nullptr;
    }

    std::pair<Resource::SharedPtr, bool> ParameterBlock::getRootDescriptor(uint32_t resourceRangeIndex, uint32_t arrayIndex) const
    {
        assert(mpReflector->getResourceRangeBindingInfo(resourceRangeIndex).isRootDescriptor());
        auto& resourceRange = mpReflector->getResourceRange(resourceRangeIndex);

        bool isUav = isSetType(resourceRange.descriptorType, kRootUavDescriptorTypes);
        bool isSrv = isSetType(resourceRange.descriptorType, kRootSrvDescriptorTypes);
        assert(isUav || isSrv);

        size_t flatIndex = resourceRange.baseIndex + arrayIndex;
        auto pResource = isUav ? mUAVs[flatIndex].pResource : mSRVs[flatIndex].pResource;

        return { pResource, isUav };
    }

    bool ParameterBlock::setBuffer(const std::string& name, const Buffer::SharedPtr& pBuf)
    {
        // Find the buffer
        auto var = getRootVar()[name];
#if _LOG_ENABLED
        if (!verifyBufferVar(var, pBuf.get(), name, "setBuffer()")) return false;
#endif
        return var.setBuffer(pBuf);
    }

    bool ParameterBlock::setBuffer(const BindLocation& bindLocation, const Buffer::SharedPtr& pBuf)
    {
        if (!bindLocation.isValid()) return false;

        return setResourceSrvUavCommon(bindLocation, pBuf, "setBuffer()");
    }

    Buffer::SharedPtr ParameterBlock::getBuffer(const std::string& name) const
    {
        // Find the buffer
        auto var = getRootVar()[name];
#if _LOG_ENABLED
        if (!verifyBufferVar(var, nullptr, name, "getBuffer()")) return nullptr;
#endif
        return var.getBuffer();
    }

    Buffer::SharedPtr ParameterBlock::getBuffer(const BindLocation& bindLocation) const
    {
        if (!bindLocation.isValid()) return nullptr;

        return getResourceSrvUavCommon(bindLocation, "getBuffer()")->asBuffer();
    }

    bool ParameterBlock::setSampler(const BindLocation& bindLocation, const Sampler::SharedPtr& pSampler)
    {
        if (!checkResourceIndices(bindLocation, "setSampler()")) return false;
        if (!checkDescriptorType(bindLocation, kSamplerDescriptorType, "setSampler()")) return false;

        size_t flatIndex = getFlatIndex(bindLocation);
        auto& pBoundSampler = mSamplers[flatIndex];

        if (pBoundSampler == pSampler) return true;

        pBoundSampler = pSampler ? pSampler : Sampler::getDefault();
        markDescriptorSetDirty(bindLocation);
        return true;
    }

    bool ParameterBlock::setSampler(const std::string& name, const Sampler::SharedPtr& pSampler)
    {
        auto var = getRootVar()[name];
#if _LOG_ENABLED
        if (!verifySamplerVar(var, name, "setSampler()")) return false;
#endif
        return var.setSampler(pSampler);
    }

    Sampler::SharedPtr ParameterBlock::getSampler(const std::string& name) const
    {
        auto var = getRootVar()[name];
#if _LOG_ENABLED
        if (!verifySamplerVar(var, name, "getSampler()")) return nullptr;
#endif
        return var.getSampler();
    }

    const Sampler::SharedPtr& ParameterBlock::getSampler(const BindLocation& bindLocation) const
    {
        static Sampler::SharedPtr pNull = nullptr;
        if (!checkResourceIndices(bindLocation, "getSampler()")) return pNull;
        if (!checkDescriptorType(bindLocation, kSamplerDescriptorType, "getSampler()")) return pNull;
        size_t flatIndex = getFlatIndex(bindLocation);
        return mSamplers[flatIndex];
    }

    ShaderResourceView::SharedPtr ParameterBlock::getSrv(const BindLocation& bindLocation) const
    {
        if (!checkResourceIndices(bindLocation, "getSrv()")) return nullptr;
        if (!checkDescriptorType(bindLocation, kSrvDescriptorTypes, "getSrv()")) return nullptr;
        size_t flatIndex = getFlatIndex(bindLocation);
        return mSRVs[flatIndex].pView;
    }

    UnorderedAccessView::SharedPtr ParameterBlock::getUav(const BindLocation& bindLocation) const
    {
        if (!checkResourceIndices(bindLocation, "getUav()")) return nullptr;
        if (!checkDescriptorType(bindLocation, kUavDescriptorTypes, "getUav()")) return nullptr;
        size_t flatIndex = getFlatIndex(bindLocation);
        return mUAVs[flatIndex].pView;
    }

    bool ParameterBlock::setTexture(const std::string& name, const Texture::SharedPtr& pTexture)
    {
        auto var = getRootVar()[name];
#if _LOG_ENABLED
        if (!verifyTextureVar(var, pTexture.get(), name, "setTexture()")) return false;
#endif
        return var.setTexture(pTexture);
    }

    bool ParameterBlock::setTexture(const BindLocation& bindLocation, const Texture::SharedPtr& pTexture)
    {
        return setResourceSrvUavCommon(bindLocation, pTexture, "setTexture()");
    }

    Texture::SharedPtr ParameterBlock::getTexture(const std::string& name) const
    {
        auto var = getRootVar()[name];
#if _LOG_ENABLED
        if (!verifyTextureVar(var, nullptr, name, "getTexture()")) return nullptr;
#endif
        return var.getTexture();
    }

    Texture::SharedPtr ParameterBlock::getTexture(const BindLocation& bindLocation) const
    {
        return getResourceSrvUavCommon(bindLocation, "getTexture()")->asTexture();
    }

    bool ParameterBlock::setSrv(const BindLocation& bindLocation, const ShaderResourceView::SharedPtr& pSrv)
    {
        auto pResource = getResourceFromView(pSrv.get());
        if (!checkDescriptorSrvUavCommon(bindLocation, pResource, pSrv, "setSrv()")) return false;

        size_t flatIndex = getFlatIndex(bindLocation);
        auto& assignedSRV = mSRVs[flatIndex];

        const auto& bindingInfo = mpReflector->getResourceRangeBindingInfo(bindLocation.getResourceRangeIndex());
        const ReflectionResourceType* pResouceReflection = bindLocation.getType()->asResourceType();
        assert(pResouceReflection && pResouceReflection->getDimensions() == bindingInfo.dimension);

        const ShaderResourceView::SharedPtr pView = pSrv ? pSrv : ShaderResourceView::getNullView(bindingInfo.dimension);
        if (assignedSRV.pResource == pResource && assignedSRV.pView == pView) return true;

        assignedSRV.pView = pView;
        assignedSRV.pResource = pResource;

        markDescriptorSetDirty(bindLocation);
        return true;
    }

    bool ParameterBlock::setUav(const BindLocation& bindLocation, const UnorderedAccessView::SharedPtr& pUav)
    {
        auto pResource = getResourceFromView(pUav.get());
        if (!checkDescriptorSrvUavCommon(bindLocation, pResource, pUav, "setUav()")) return false;

        size_t flatIndex = getFlatIndex(bindLocation);
        auto& assignedUAV = mUAVs[flatIndex];

        const auto& bindingInfo = mpReflector->getResourceRangeBindingInfo(bindLocation.getResourceRangeIndex());
        const ReflectionResourceType* pResouceReflection = bindLocation.getType()->asResourceType();
        assert(pResouceReflection && pResouceReflection->getDimensions() == bindingInfo.dimension);

        UnorderedAccessView::SharedPtr pView = pUav ? pUav : UnorderedAccessView::getNullView(bindingInfo.dimension);
        if (assignedUAV.pResource == pResource && assignedUAV.pView == pView) return true;

        assignedUAV.pView = pView;
        assignedUAV.pResource = getResourceFromView(pView.get());

        markDescriptorSetDirty(bindLocation);
        return true;
    }

    const Buffer::SharedPtr& ParameterBlock::getUnderlyingConstantBuffer() const
    {
        updateSpecialization();

        auto pSlangTypeLayout = mpSpecializedReflector->getElementType()->getSlangTypeLayout();

        auto requiredSize = pSlangTypeLayout->getSize();
        if(auto pSlangPendingTypeLayout = pSlangTypeLayout->getPendingDataTypeLayout() )
        {
            // Note: In this case, the type being stored in this block has been specialized
            // (because concrete types have been plugged in for its interface-type fields),
            // and as a result we need some extra bytes in our underlying constant buffer.
            //
            // The Slang reflection API can tell us the size of the unspecialized part with
            // `getSize()` above, and can tell us the size required for all the "pending"
            // data that couldn't be laid out until specialization was performed (using
            // `pSlangPendingTypeLayout->getSize()`, but just adding those two together
            // doesn't account for the possibility of padding required for alignment.
            //
            // There are actually Slang reflection APIs that could tell us what we
            // need to know (the starting offset of this extra "pending" data),
            // but the way the Falcor `ParameterBlockReflection` is structured today,
            // we haven't held onto the relevant Slang reflection objects.
            //
            // For now we just add in an extra 16 bytes to cover the possibility of
            // alignment.
            //
            // TODO: Use the Slang reflection API in a more refined fashion.
            //
            requiredSize += pSlangPendingTypeLayout->getSize() + 16;
        }

        if( !mUnderlyingConstantBuffer.pBuffer || mUnderlyingConstantBuffer.pBuffer->getSize() < requiredSize )
        {
            mUnderlyingConstantBuffer.pBuffer = Buffer::create(requiredSize, Buffer::BindFlags::Constant, Buffer::CpuAccess::Write);
        }

        return mUnderlyingConstantBuffer.pBuffer;
    }

    ConstantBufferView::SharedPtr ParameterBlock::getUnderlyingConstantBufferView()
    {
        if (mUnderlyingConstantBuffer.pCBV == nullptr)
        {
            mUnderlyingConstantBuffer.pCBV = ConstantBufferView::create(getUnderlyingConstantBuffer());
        }
        return mUnderlyingConstantBuffer.pCBV;
    }

    template<typename VarType>
    ReflectionBasicType::Type getReflectionTypeFromCType()
    {
#define c_to_prog(cType, progType) if(typeid(VarType) == typeid(cType)) return ReflectionBasicType::Type::progType;
        c_to_prog(bool,  Bool);
        c_to_prog(bool2, Bool2);
        c_to_prog(bool3, Bool3);
        c_to_prog(bool4, Bool4);

        c_to_prog(int32_t, Int);
        c_to_prog(int2,    Int2);
        c_to_prog(int3,    Int3);
        c_to_prog(int4,    Int4);

        c_to_prog(uint32_t, Uint);
        c_to_prog(uint2,    Uint2);
        c_to_prog(uint3,    Uint3);
        c_to_prog(uint4,    Uint4);

        c_to_prog(float,  Float);
        c_to_prog(float2, Float2);
        c_to_prog(float3, Float3);
        c_to_prog(float4, Float4);

        c_to_prog(glm::mat2,   Float2x2);
        c_to_prog(glm::mat2x3, Float2x3);
        c_to_prog(glm::mat2x4, Float2x4);

        c_to_prog(glm::mat3  , Float3x3);
        c_to_prog(glm::mat3x2, Float3x2);
        c_to_prog(glm::mat3x4, Float3x4);

        c_to_prog(glm::mat4,   Float4x4);
        c_to_prog(glm::mat4x2, Float4x2);
        c_to_prog(glm::mat4x3, Float4x3);

#undef c_to_prog
        should_not_get_here();
        return ReflectionBasicType::Type::Unknown;
    }

    UniformShaderVarOffset ParameterBlock::getVariableOffset(const std::string& varName) const
    {
        return getElementType()->getZeroOffset()[varName];
    }

    static bool hasChangedSince(ParameterBlock::ChangeEpoch current, ParameterBlock::ChangeEpoch lastObserved)
    {
        return current > lastObserved;
    }

    void ParameterBlock::checkForIndirectChanges(ParameterBlockReflection const* pReflector) const
    {
        // First off, we will recursively check any parameter blocks attached to use
        // for `ConstantBuffer<T>`/`cbuffer` or interface-type parameters, since
        // changes to their state will indirectly affect validity of the descriptor
        // sets attached to `this`.
        //
        auto resourceRangeCount = pReflector->getResourceRangeCount();
        for(uint32_t r = 0; r < resourceRangeCount; ++r)
        {
            auto rangeBindingInfo = pReflector->getResourceRangeBindingInfo(r);
            switch(rangeBindingInfo.flavor)
            {
            default:
                continue;

            case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface:
            case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ConstantBuffer:
                break;
            }

            auto pSubObjectReflector = rangeBindingInfo.pSubObjectReflector.get();
            assert(pSubObjectReflector);

            auto rangeInfo = pReflector->getResourceRange(r);
            for(uint32_t i = 0; i < rangeInfo.count; ++i)
            {
                auto& assigned = getAssignedParameterBlock(r, i);
                auto& pSubObject = assigned.pBlock;

                pSubObject->checkForIndirectChanges(pSubObjectReflector);
            }
        }

        // Next, we will check for any cases where one of our descriptor sets
        // would be invalidated due to a change in a sub-block.
        //
        auto setCount = pReflector->getDescriptorSetCount();
        for(uint32_t s = 0; s < setCount; ++s)
        {
            auto& setInfo = pReflector->getDescriptorSetInfo(s);
            for( auto subObjectRange : setInfo.subObjects )
            {
                auto resourceRangeIndex = subObjectRange.resourceRangeIndexOfSubObject;
                auto setIndexInSubObject = subObjectRange.setIndexInSubObject;

                auto subObjectCount = pReflector->getResourceRange(resourceRangeIndex).count;
                auto pSubObjectReflector = pReflector->getResourceRangeBindingInfo(resourceRangeIndex).pSubObjectReflector.get();
                for(uint32_t i = 0; i < subObjectCount; ++i)
                {
                    auto& assigned = getAssignedParameterBlock(resourceRangeIndex, i);
                    auto& pSubObject = assigned.pBlock;

                    auto lastObservedChangeEpoch = assigned.epochOfLastObservedChange;
                    auto currentChangeEpoch = pSubObject->mSets[setIndexInSubObject].epochOfLastChange;

                    if(currentChangeEpoch > lastObservedChangeEpoch)
                    {
                        markDescriptorSetDirty(s);
                    }
                }
            }
        }

        // Next, we will check for any cases where our default constant
        // buffer would be invalidated due to changes in a sub-block.
        // This can only occur for sub-blocks of interface type.
        //
        for(uint32_t r = 0; r < resourceRangeCount; ++r)
        {
            auto rangeBindingInfo = pReflector->getResourceRangeBindingInfo(r);
            if(rangeBindingInfo.flavor != ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface)
                continue;

            auto pSubObjectReflector = rangeBindingInfo.pSubObjectReflector.get();
            assert(pSubObjectReflector);

            auto rangeInfo = pReflector->getResourceRange(r);
            for(uint32_t i = 0; i < rangeInfo.count; ++i)
            {
                auto& assigned = getAssignedParameterBlock(r, i);
                auto& pSubObject = assigned.pBlock;

                auto lastObservedChangeEpoch = assigned.epochOfLastObservedChange;
                auto currentChangeEpoch = pSubObject->mEpochOfLastUniformDataChange;

                if(currentChangeEpoch > lastObservedChangeEpoch)
                {
                    markUniformDataDirty();
                }
            }
        }

        // Finally, we will update our tracking information to show that we have
        // updated to the latest changed in each sub-block.
        //
        for(uint32_t r = 0; r < resourceRangeCount; ++r)
        {
            auto rangeBindingInfo = pReflector->getResourceRangeBindingInfo(r);
            switch(rangeBindingInfo.flavor)
            {
            default:
                continue;

            case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface:
            case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ConstantBuffer:
                break;
            }

            auto pSubObjectReflector = rangeBindingInfo.pSubObjectReflector.get();
            assert(pSubObjectReflector);

            auto rangeInfo = pReflector->getResourceRange(r);
            for(uint32_t i = 0; i < rangeInfo.count; ++i)
            {
                auto& assigned = getAssignedParameterBlock(r, i);
                auto& pSubObject = assigned.pBlock;

                assigned.epochOfLastObservedChange = pSubObject->mEpochOfLastChange;
            }
        }
    }

    ParameterBlock::ChangeEpoch ParameterBlock::computeEpochOfLastChange(ParameterBlock* pBlock)
    {
        pBlock->checkForIndirectChanges(pBlock->mpReflector.get());
        return getEpochOfLastChange(pBlock);
    }

    void ParameterBlock::validateUnderlyingConstantBuffer(
        ParameterBlockReflection const* pReflector)
    {
        // There is no need to allocate and fill in the underlying constant
        // buffer if we will actually be using root constants.
        //
        if(pReflector->getDefaultConstantBufferBindingInfo().useRootConstants)
            return;

        // We will check if the uniform data (e.g., ths stuff stored
        // in `mData`) has changed since we last wrote into the underlying
        // constant buffer.
        //
        // Note: We do *not* call `checkForIndirectChanges()` here, so
        // we are relying on the caller to have already done that.
        //
        // If no changes have been made to uniform data since we filled
        // in the underlying buffer, then we won't have to do anything.
        //
        auto epochOfLastUniformDataChange = mEpochOfLastUniformDataChange;
        if(mUnderlyingConstantBuffer.epochOfLastObservedChange == epochOfLastUniformDataChange)
            return;

        // Otherwise, we need to ensure that the underlying constant
        // buffer has been allocated, then map it and fill it in.
        //
        auto pBuffer = getUnderlyingConstantBuffer().get();
        char* pData = (char*) pBuffer->map(Buffer::MapType::WriteDiscard);
        writeIntoBuffer(pReflector, pData, pBuffer->getSize());
        pBuffer->unmap();

        // Once we've filled in the underlying constant buffer, we need
        // to note the epoch at which we last observed the uniform state,
        // to avoid redundantly filling in the buffer yet again.
        //
        mUnderlyingConstantBuffer.epochOfLastObservedChange = epochOfLastUniformDataChange;

        // Finally, we need to clobber the constant buffer view object,
        // because it is created on demand, and the old view will still
        // reference an old/stale version of the buffer (since `WriteDiscard`
        // will lead to the buffer having a new memory address).
        //
        mUnderlyingConstantBuffer.pCBV = nullptr;
    }

    void ParameterBlock::writeIntoBuffer(
        ParameterBlockReflection const*   pReflector,
        char*                           pBuffer,
        size_t                          bufferSize)
    {
        auto dataSize = mData.size();
        assert(dataSize <= bufferSize);
        memcpy(pBuffer, mData.data(), dataSize);

        auto resourceRangeCount = pReflector->getResourceRangeCount();
        for(uint32_t rr = 0; rr < resourceRangeCount; ++rr)
        {
            auto rangeBindingInfo = pReflector->getResourceRangeBindingInfo(rr);
            if(rangeBindingInfo.flavor != ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface)
                continue;

            auto pSubObjectReflector = rangeBindingInfo.pSubObjectReflector.get();
            assert(pSubObjectReflector);

            auto rangeInfo = pReflector->getResourceRange(rr);
            for(uint32_t ii = 0; ii < rangeInfo.count; ++ii)
            {
                assert(ii == 0);
                auto pSubObject = getParameterBlock(rr, ii);

                size_t subObjectOffset = rangeBindingInfo.regIndex;
                assert(subObjectOffset <= bufferSize);
                pSubObject->writeIntoBuffer(
                    pSubObjectReflector,
                    pBuffer + subObjectOffset,
                    bufferSize - subObjectOffset);
            }
        }
    }

    template<typename VarType>
    bool checkVariableType(const ReflectionType* pShaderType, const std::string& name, const std::string& bufferName)
    {
#if _LOG_ENABLED
        auto callType = getReflectionTypeFromCType<VarType>();
        const ReflectionBasicType* pBasicType = pShaderType->asBasicType();
        ReflectionBasicType::Type shaderType = pBasicType ? pBasicType->getType() : ReflectionBasicType::Type::Unknown;
        // Check that the types match
        if(callType != shaderType)
        {
            std::string msg("Error when setting variable '");
            msg += name + "' to buffer '" + bufferName + "'.\n";
            msg += "Type mismatch.\nsetVariable() was called with Type " + to_string(callType) + ".\nVariable was declared with Type " + to_string(shaderType) + ".\n\n";
            logError(msg);
            assert(0);
            return false;
        }
#endif
        return true;
    }

    template<typename VarType>
    bool checkVariableByOffset(size_t offset, size_t count, const ReflectionType* pReflection)
    {
#if _LOG_ENABLED
        // Make sure the first element matches what is expected
        TypedShaderVarOffset bindLoc = pReflection->findMemberByOffset(offset);
        if (!bindLoc.isValid())
        {
            logError("Trying to set a variable at offset " + std::to_string(offset) + " but this offset is not used in the buffer");
            return false;
        }

        return true;
#else
        return true;
#endif
    }

    template<typename VarType>
    bool ParameterBlock::setVariable(UniformShaderVarOffset offset, const VarType& value)
    {
        if(checkVariableByOffset<VarType>(offset.getByteOffset(), 1, getElementType().get()))
        {
            const uint8_t* pVar = mData.data() + offset.getByteOffset();
            *(VarType*)pVar = value;
            markUniformDataDirty();
            return true;
        }
        return false;
    }

#define set_constant_by_offset(_t) template dlldecl bool ParameterBlock::setVariable(UniformShaderVarOffset offset, const _t& value)

    set_constant_by_offset(bool);
    set_constant_by_offset(bool2);
    set_constant_by_offset(bool3);
    set_constant_by_offset(bool4);

    set_constant_by_offset(uint32_t);
    set_constant_by_offset(uint2);
    set_constant_by_offset(uint3);
    set_constant_by_offset(uint4);

    set_constant_by_offset(int32_t);
    set_constant_by_offset(int2);
    set_constant_by_offset(int3);
    set_constant_by_offset(int4);

    set_constant_by_offset(float);
    set_constant_by_offset(float2);
    set_constant_by_offset(float3);
    set_constant_by_offset(float4);

    set_constant_by_offset(glm::mat2);
    set_constant_by_offset(glm::mat2x3);
    set_constant_by_offset(glm::mat2x4);

    set_constant_by_offset(glm::mat3);
    set_constant_by_offset(glm::mat3x2);
    set_constant_by_offset(glm::mat3x4);

    set_constant_by_offset(glm::mat4);
    set_constant_by_offset(glm::mat4x2);
    set_constant_by_offset(glm::mat4x3);

    set_constant_by_offset(uint64_t);

#undef set_constant_by_offset

    bool ParameterBlock::setBlob(const void* pSrc, size_t offset, size_t size)
    {
        if((_LOG_ENABLED != 0) && (offset + size > mData.size()))
        {
            std::string Msg("Error when setting blob to parameter block. Blob too large and will result in overflow. Ignoring call.");
            logError(Msg);
            return false;
        }
        std::memcpy(mData.data() + offset, pSrc, size);
        markUniformDataDirty();
        return true;
    }

    bool ParameterBlock::setBlob(const void* pSrc, UniformShaderVarOffset loc, size_t size)
    {
        return setBlob(pSrc, loc.getByteOffset(), size);
    }

    size_t ParameterBlock::getElementSize() const
    {
        return getElementType()->getByteSize();
    }

    void ParameterBlock::renderUI(Gui::Widgets& widget)
    {
        VariablesBufferUI variablesBufferUI(*this);
        variablesBufferUI.renderUI(widget);
    }

    static void prepareResource(
        CopyContext* pContext,
        Resource* pResource,
        bool isUav)
    {
        if (!pResource) return;

        // If it's a buffer with a UAV counter, insert a UAV barrier
        const Buffer* pBuffer = pResource->asBuffer().get();
        if (isUav && pBuffer && pBuffer->getUAVCounter())
        {
            pContext->resourceBarrier(pBuffer->getUAVCounter().get(), Resource::State::UnorderedAccess);
            pContext->uavBarrier(pBuffer->getUAVCounter().get());
        }

        bool insertBarrier = true;
#ifdef FALCOR_D3D12
        insertBarrier = (is_set(pResource->getBindFlags(), Resource::BindFlags::AccelerationStructure) == false);
#endif
        if (insertBarrier)
        {
            insertBarrier = !pContext->resourceBarrier(pResource, isUav ? Resource::State::UnorderedAccess : Resource::State::ShaderResource);
        }

        // Insert UAV barrier automatically if the resource is an UAV that is already in UnorderedAccess state.
        // Otherwise the user would have to insert barriers explicitly between passes accessing UAVs, which is easily forgotten.
        if (insertBarrier && isUav) pContext->uavBarrier(pResource);
    }

    ParameterBlock::SharedPtr const& ParameterBlock::getParameterBlock(uint32_t resourceRangeIndex, uint32_t arrayIndex) const
    {
        auto rangeInfo = mpReflector->getElementType()->getResourceRange(resourceRangeIndex);
        return mParameterBlocks[rangeInfo.baseIndex + arrayIndex].pBlock;
    }

    ParameterBlock::AssignedParameterBlock const& ParameterBlock::getAssignedParameterBlock(uint32_t resourceRangeIndex, uint32_t arrayIndex) const
    {
        auto rangeInfo = mpReflector->getElementType()->getResourceRange(resourceRangeIndex);
        return mParameterBlocks[rangeInfo.baseIndex + arrayIndex];
    }

    bool ParameterBlock::bindIntoDescriptorSet(
        const ParameterBlockReflection* pReflector,
        DescriptorSet::SharedPtr        pDescSet,
        uint32_t                        setIndex,
        uint32_t&                       ioDestRangeIndex)
    {
        if( pReflector->hasDefaultConstantBuffer() )
        {
            auto defaultConstantBufferInfo = pReflector->getDefaultConstantBufferBindingInfo();
            if(setIndex == defaultConstantBufferInfo.descriptorSetIndex )
            {
                uint32_t destRangeIndex = ioDestRangeIndex++;

                ConstantBufferView::SharedPtr pView = getUnderlyingConstantBufferView();
                pDescSet->setCbv(destRangeIndex, 0, pView.get());
            }
        }

        return bindResourcesIntoDescriptorSet(pReflector, pDescSet, setIndex, ioDestRangeIndex);
    }

    bool ParameterBlock::bindResourcesIntoDescriptorSet(
        const ParameterBlockReflection* pReflector,
        DescriptorSet::SharedPtr        pDescSet,
        uint32_t                        setIndex,
        uint32_t&                       ioDestRangeIndex)
    {
        auto& setInfo = pReflector->getDescriptorSetInfo(setIndex);

        for(auto resourceRangeIndex : setInfo.resourceRangeIndices)
        {
            // TODO: Should this use the specialized reflector's element type instead?
            auto resourceRange = getElementType()->getResourceRange(resourceRangeIndex);
            const auto& bindingInfo = pReflector->getResourceRangeBindingInfo(resourceRangeIndex);

            DescriptorSet::Type descriptorType = resourceRange.descriptorType;
            size_t descriptorCount = resourceRange.count;

            uint32_t destRangeIndex = ioDestRangeIndex++;

            // TODO(tfoley): We could swap the loop and `switch`
            // and do something like a single `setCbvs(...)` call per range.
            for(uint32_t descriptorIndex = 0; descriptorIndex < descriptorCount; descriptorIndex++)
            {
                size_t flatIndex = resourceRange.baseIndex + descriptorIndex;
                switch( descriptorType )
                {
                case DescriptorSet::Type::Cbv:
#if 0
                    {
                        // TODO(tfoley): Shouldn't actually be setting CBV here,
                        // since that should be the responsibility of the nested
                        // `ParameterBlock`.
                        //
                        ParameterBlock* pCB = mCBs[flatIndex].get();
                        ConstantBufferView::SharedPtr pView = pCB ? pCB->getCbv() : ConstantBufferView::getNullView();
                        pDescSet->setCbv(destRangeIndex, descriptorIndex, pView.get());
                    }
#endif
                    break;

                case DescriptorSet::Type::Sampler:
                    {
                        auto pSampler = mSamplers[flatIndex];
                        if(!pSampler) pSampler = Sampler::getDefault();
                        pDescSet->setSampler(destRangeIndex, descriptorIndex, pSampler.get());
                    }
                    break;
                case DescriptorSet::Type::TextureSrv:
                case DescriptorSet::Type::RawBufferSrv:
                case DescriptorSet::Type::TypedBufferSrv:
                case DescriptorSet::Type::StructuredBufferSrv:
                case DescriptorSet::Type::AccelerationStructureSrv:
                    {
                        auto pView = mSRVs[flatIndex].pView;
                        assert(bindingInfo.dimension != ReflectionResourceType::Dimensions::Unknown);
                        if(!pView) pView = ShaderResourceView::getNullView(bindingInfo.dimension);
                        pDescSet->setSrv(destRangeIndex, descriptorIndex, pView.get());
                    }
                    break;
                case DescriptorSet::Type::TextureUav:
                case DescriptorSet::Type::RawBufferUav:
                case DescriptorSet::Type::TypedBufferUav:
                case DescriptorSet::Type::StructuredBufferUav:
                    {
                        auto pView = mUAVs[flatIndex].pView;
                        assert(bindingInfo.dimension != ReflectionResourceType::Dimensions::Unknown);
                        if(!pView) pView = UnorderedAccessView::getNullView(bindingInfo.dimension);
                        pDescSet->setUav(destRangeIndex, descriptorIndex, pView.get());
                    }
                    break;

                default:
                    should_not_get_here();
                    return false;
                }
            }
        }

        // Recursively bind resources in sub-objects.
        for( auto subObjectRange : setInfo.subObjects )
        {
            auto resourceRangeIndex = subObjectRange.resourceRangeIndexOfSubObject;
            auto setIndexInSubObject = subObjectRange.setIndexInSubObject;

            auto subObjectCount = pReflector->getResourceRange(resourceRangeIndex).count;
            auto pSubObjectReflector = pReflector->getResourceRangeBindingInfo(resourceRangeIndex).pSubObjectReflector.get();
            for(uint32_t i = 0; i < subObjectCount; ++i)
            {
                // TODO: if `subObjectCount` is > 1, then we really need
                // to pass down an array index to apply when setting things...
                assert(subObjectCount == 1);

                auto pSubObject = getParameterBlock(resourceRangeIndex, i);
                if( pReflector->getResourceRangeBindingInfo(resourceRangeIndex).flavor == ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface )
                {
                    pSubObject->bindResourcesIntoDescriptorSet(
                        pSubObjectReflector,
                        pDescSet,
                        setIndexInSubObject,
                        ioDestRangeIndex);
                }
                else
                {
                    pSubObject->bindIntoDescriptorSet(
                        pSubObjectReflector,
                        pDescSet,
                        setIndexInSubObject,
                        ioDestRangeIndex);
                }
            }
        }

        return true;
    }

    bool ParameterBlock::updateSpecialization() const
    {
        auto pSlangTypeLayout = getElementType()->getSlangTypeLayout();

        // If the element type has no unspecialized existential/interface types
        // in it, then there is nothing to be done.
        //
        if( pSlangTypeLayout && pSlangTypeLayout->getSize(slang::ParameterCategory::ExistentialTypeParam) == 0 )
        {
            mpSpecializedReflector = mpReflector;
            return false;
        }

        // TODO: we should have some kind of caching step here, where we
        // collect the existential type arguments and see if they've changed
        // since the last time.

        return updateSpecializationImpl();
    }

    void ParameterBlock::collectSpecializationArgs(ParameterBlock::SpecializationArgs& ioArgs) const
    {
        auto pReflector = getReflection();
        uint32_t resourceRangeCount = pReflector->getResourceRangeCount();
        for( uint32_t r = 0; r < resourceRangeCount; ++r )
        {
            auto rangeInfo = pReflector->getResourceRange(r);
            auto rangeBindingInfo = pReflector->getResourceRangeBindingInfo(r);
            switch( rangeBindingInfo.flavor )
            {
            case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface:
                {
                    // For now we only support declarations of single interface-type
                    // parameters and not arrays.
                    //
                    // TODO: In order to support the array case we need to either:
                    //
                    // 1. Enforce that all the bound objects have the same concrete
                    //   type, and then specialize on that one known type.
                    //
                    // 2. Collect all of the concrete types that are bound and create
                    //   a "tagged union" type over them that we specialize to instead.
                    //
                    assert(rangeInfo.count == 1);

                    auto pSubObject = getParameterBlock(r, 0);

                    // TODO: We should actually be querying the specialized element type
                    // for the object, and not just its direct type.

                    auto pSubObjectType = pSubObject->getElementType()->getSlangTypeLayout()->getType();

                    slang::SpecializationArg specializationArg;
                    specializationArg.kind = slang::SpecializationArg::Kind::Type;
                    specializationArg.type = pSubObjectType;
                    ioArgs.push_back(specializationArg);
                }
                break;

            case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ConstantBuffer:
            case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ParameterBlock:
                {
                    assert(rangeInfo.count == 1);
                    auto pSubObject = getParameterBlock(r, 0);
                    pSubObject->collectSpecializationArgs(ioArgs);
                }
                break;

            default:
                break;
            }
        }
    }

    bool ParameterBlock::updateSpecializationImpl() const
    {
        // We want to compute the specialized layout that this object
        // should use...

        auto pSlangTypeLayout = getElementType()->getSlangTypeLayout();
        auto pSlangType = pSlangTypeLayout->getType();

        SpecializationArgs specializationArgs;
        collectSpecializationArgs(specializationArgs);

        // We call back into Slang to generate a specialization of
        // the element type and compute its layout.
        //
        // TODO: We should probably have a cache of specialized layouts
        // here, so that if we go back to a previous configuration we
        // can re-use its layout.
        //

        auto pSlangSession = mpProgramVersion->getSlangSession();

        ComPtr<ISlangBlob> pDiagnostics;
        auto pSpecializedSlangType = pSlangSession->specializeType(
            pSlangType,
            specializationArgs.data(),
            specializationArgs.size(),
            pDiagnostics.writeRef());
        if( !pSpecializedSlangType )
        {
            assert(pDiagnostics);
            logError((const char*) pDiagnostics->getBufferPointer());
            return false;
        }

        if( pDiagnostics )
        {
            logWarning((const char*) pDiagnostics->getBufferPointer());
        }

        auto pSpecializedSlangTypeLayout = pSlangSession->getTypeLayout(pSpecializedSlangType);

        mpSpecializedReflector = ParameterBlockReflection::create(mpProgramVersion.get(), pSpecializedSlangTypeLayout);

        return true;
    }

    bool ParameterBlock::prepareDefaultConstantBufferAndResources(
        CopyContext*                        pContext,
        ParameterBlockReflection const*     pReflector)
    {
        if( pReflector->hasDefaultConstantBuffer() )
        {
            validateUnderlyingConstantBuffer(pReflector);
        }

        return prepareResources(pContext, pReflector);
    }

    bool ParameterBlock::prepareResources(
        CopyContext*                    pContext,
        ParameterBlockReflection const* pReflector)
    {
        // Prepare all bound resources by inserting appropriate barriers/transitions as needed.
        for(auto& srv : mSRVs)
        {
            prepareResource(pContext, srv.pResource.get(), false);
        }

        for(auto& uav : mUAVs)
        {
            prepareResource(pContext, uav.pResource.get(), true);
        }

        // Recursively prepare the resources in all sub-blocks bound to this parameter block.
        auto resourceRangeCount = pReflector->getResourceRangeCount();
        for(uint32_t resourceRangeIndex = 0; resourceRangeIndex < resourceRangeCount; ++resourceRangeIndex)
        {
            auto& bindingInfo = pReflector->getResourceRangeBindingInfo(resourceRangeIndex);
            if (bindingInfo.flavor == ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Simple ||
                bindingInfo.flavor == ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::RootDescriptor)
            {
                continue;
            }

            auto& resourceRange = pReflector->getResourceRange(resourceRangeIndex);
            auto pSubObjectReflector = bindingInfo.pSubObjectReflector.get();
            auto objectCount = resourceRange.count;

            for(uint32_t i = 0; i < objectCount; ++i)
            {
                auto& assigned = getAssignedParameterBlock(resourceRangeIndex, i);
                auto pSubBlock = assigned.pBlock;

                switch(bindingInfo.flavor)
                {
                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ConstantBuffer:
                    if( !pSubBlock->prepareDefaultConstantBufferAndResources(pContext, pSubObjectReflector) )
                    {
                        return false;
                    }
                    break;

                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ParameterBlock:
                    if( !pSubBlock->prepareDescriptorSets(pContext, pSubObjectReflector) )
                    {
                        return false;
                    }
                    break;

                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface:
                    if( !pSubBlock->prepareResources(pContext, pSubObjectReflector) )
                    {
                        return false;
                    }
                    break;

                default:
                    should_not_get_here();
                    return false;
                }
            }
        }

        return true;
    }

    bool ParameterBlock::prepareDescriptorSets(CopyContext* pContext)
    {
        // Note: allocating and filling in descriptor sets will always
        // use the specialized reflector, which is based on how the
        // object has been laid out based on the known types of
        // interface-type parameters.
        //
        updateSpecialization();
        auto pReflector = mpSpecializedReflector.get();

        return prepareDescriptorSets(pContext, pReflector);
    }

    bool ParameterBlock::prepareDescriptorSets(
        CopyContext*                    pContext,
        const ParameterBlockReflection* pReflector)
    {
        // We first need to check for "indirect" changes, where a write to
        // a sub-block (e.g., a constant buffer) will require invalidation
        // and re-creation of descriptor sets for this parameter block.
        //
        checkForIndirectChanges(pReflector);

        // Next we need to recursively "prepare" the resources for this
        // block. This step combines three kinds of work:
        //
        // 1. Ensuring that the underlying constant buffer (if required)
        //    has been allocated and filled in.
        //
        // 2. Ensuring that for any resources that have been written
        //    to or that went through a layout/state change, we insert
        //    the appropriate barriers/transitions.
        //
        // 3. Ensuring that any sub-parameter-blocks that will
        //    bind additional descriptor sets also prepare themsevles,
        //    recursively.
        //
        if( !prepareDefaultConstantBufferAndResources(pContext, pReflector) )
        {
            return false;
        }

        // We might have zero or more descriptor sets that need to
        // be re-allocated and filled in, because their contents
        // have been invalidated in one way or another.
        //
        uint32_t setCount = pReflector->getDescriptorSetCount();
        for(uint32_t setIndex = 0; setIndex < setCount; ++setIndex)
        {
            auto& pSet = mSets[setIndex].pSet;
            if(pSet) continue;

            auto pSetLayout = pReflector->getDescriptorSetLayout(setIndex);
            pSet = DescriptorSet::create(gpDevice->getGpuDescriptorPool(), pSetLayout);

            uint32_t destRangeIndex = 0;
            bindIntoDescriptorSet(
                pReflector,
                pSet,
                setIndex,
                destRangeIndex);
        }

        return true;
    }
}
