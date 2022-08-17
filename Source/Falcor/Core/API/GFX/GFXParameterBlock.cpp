/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/API/ParameterBlock.h"
#include "Core/API/Device.h"
#include "Core/API/GFX/GFXAPI.h"
#include "Core/Program/ProgramVersion.h"
#include "Utils/Logger.h"

namespace Falcor
{
    namespace
    {
        gfx::ShaderOffset getGFXShaderOffset(const UniformShaderVarOffset& offset)
        {
            gfx::ShaderOffset result;
            result.bindingArrayIndex = 0;
            result.bindingRangeIndex = 0;
            result.uniformOffset = offset.getByteOffset();
            return result;
        }

        gfx::ShaderOffset getGFXShaderOffset(const ParameterBlock::BindLocation& bindLoc)
        {
            gfx::ShaderOffset gfxOffset = {};
            gfxOffset.bindingArrayIndex = bindLoc.getResourceArrayIndex();
            gfxOffset.bindingRangeIndex = bindLoc.getResourceRangeIndex();
            gfxOffset.uniformOffset = bindLoc.getUniform().getByteOffset();
            return gfxOffset;
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
                FALCOR_UNREACHABLE();
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
                FALCOR_UNREACHABLE();
                return false;
            }
        }

        bool isCbvType(const ReflectionType::SharedConstPtr& pType)
        {
            auto resourceType = pType->unwrapArray()->asResourceType();
            if (resourceType->getType() == ReflectionResourceType::Type::ConstantBuffer)
            {
                FALCOR_ASSERT(resourceType->getShaderAccess() == ReflectionResourceType::ShaderAccess::Read);
                return true;
            }
            return false;
        }
    }

    ParameterBlock::~ParameterBlock() {}

    ParameterBlock::ParameterBlock(const ProgramReflection::SharedConstPtr& pReflector)
        : mpReflector(pReflector->getDefaultParameterBlock())
        , mpProgramVersion(pReflector->getProgramVersion())
    {
        FALCOR_GFX_CALL(gpDevice->getApiHandle()->createMutableRootShaderObject(
            pReflector->getProgramVersion()->getKernels(nullptr)->getApiHandle(),
            mpShaderObject.writeRef()));
        createConstantBuffers(getRootVar());
    }

    ParameterBlock::ParameterBlock(
        const std::shared_ptr<const ProgramVersion>& pProgramVersion,
        const ParameterBlockReflection::SharedConstPtr& pReflection)
        : mpReflector(pReflection)
        , mpProgramVersion(pProgramVersion)
    {
        FALCOR_GFX_CALL(gpDevice->getApiHandle()->createMutableShaderObjectFromTypeLayout(
            pReflection->getElementType()->getSlangTypeLayout(),
            mpShaderObject.writeRef()));
        createConstantBuffers(getRootVar());
    }

    bool ParameterBlock::setBlob(const void* pSrc, UniformShaderVarOffset offset, size_t size)
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(offset);
        return SLANG_SUCCEEDED(mpShaderObject->setData(gfxOffset, pSrc, size));
    }

    bool ParameterBlock::setBlob(const void* pSrc, size_t offset, size_t size)
    {
        gfx::ShaderOffset gfxOffset = {};
        gfxOffset.uniformOffset = offset;
        return SLANG_SUCCEEDED(mpShaderObject->setData(gfxOffset, pSrc, size));
    }

    bool ParameterBlock::setBuffer(const std::string& name, const Buffer::SharedPtr& pBuffer)
    {
        auto var = getRootVar()[name];
        return var.setBuffer(pBuffer);
    }

    bool ParameterBlock::setBuffer(const BindLocation& bindLoc, const Buffer::SharedPtr& pResource)
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLoc);
        if (isUavType(bindLoc.getType()))
        {
            auto pUAV = pResource ? pResource->getUAV() : UnorderedAccessView::getNullView(ReflectionResourceType::Dimensions::Buffer);
            mpShaderObject->setResource(gfxOffset, pUAV->getApiHandle());
            mUAVs[gfxOffset] = pUAV;
        }
        else if (isSrvType(bindLoc.getType()))
        {
            auto pSRV = pResource ? pResource->getSRV() : ShaderResourceView::getNullView(ReflectionResourceType::Dimensions::Buffer);
            mpShaderObject->setResource(gfxOffset, pSRV->getApiHandle());
            mSRVs[gfxOffset] = pSRV;
        }
        else
        {
            logError("Error trying to bind resource to non SRV/UAV variable. Ignoring call.");
            return false;
        }
        return true;
    }

    Buffer::SharedPtr ParameterBlock::getBuffer(const std::string& name) const
    {
        auto var = getRootVar()[name];
        return var.getBuffer();
    }

    Buffer::SharedPtr ParameterBlock::getBuffer(const BindLocation& bindLoc) const
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLoc);
        if (isUavType(bindLoc.getType()))
        {
            auto iter = mUAVs.find(gfxOffset);
            if (iter == mUAVs.end()) return nullptr;
            auto pResource = iter->second->getResource();
            return pResource ? pResource->asBuffer() : nullptr;
        }
        else if (isSrvType(bindLoc.getType()))
        {
            auto iter = mSRVs.find(gfxOffset);
            if (iter == mSRVs.end()) return nullptr;
            auto pResource = iter->second->getResource();
            return pResource ? pResource->asBuffer() : nullptr;
        }
        else
        {
            logError("Error trying to bind resource to non SRV/UAV variable. Ignoring call.");
            return nullptr;
        }
    }

    bool ParameterBlock::setParameterBlock(const std::string& name, const ParameterBlock::SharedPtr& pBlock)
    {
        auto var = getRootVar()[name];
        return var.setParameterBlock(pBlock);
    }

    bool ParameterBlock::setParameterBlock(const BindLocation& bindLocation, const ParameterBlock::SharedPtr& pBlock)
    {
        auto gfxOffset = getGFXShaderOffset(bindLocation);
        mParameterBlocks[gfxOffset] = pBlock;
        return SLANG_SUCCEEDED(mpShaderObject->setObject(gfxOffset, pBlock ? pBlock->mpShaderObject : nullptr));
    }

    ParameterBlock::SharedPtr ParameterBlock::getParameterBlock(const std::string& name) const
    {
        auto var = getRootVar()[name];
        return var.getParameterBlock();
    }

    ParameterBlock::SharedPtr ParameterBlock::getParameterBlock(const BindLocation& bindLocation) const
    {
        auto gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mParameterBlocks.find(gfxOffset);
        if (iter == mParameterBlocks.end()) return nullptr;
        return iter->second;
    }

    template<typename VarType>
    bool ParameterBlock::setVariable(UniformShaderVarOffset offset, const VarType& value)
    {
        auto gfxOffset = getGFXShaderOffset(offset);
        return SLANG_SUCCEEDED(mpShaderObject->setData(gfxOffset, &value, sizeof(VarType)));
    }

#define set_constant_by_offset(_t) template FALCOR_API bool ParameterBlock::setVariable(UniformShaderVarOffset offset, const _t& value)
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

    set_constant_by_offset(rmcv::mat1x4);
    set_constant_by_offset(rmcv::mat2x4);
    set_constant_by_offset(rmcv::mat3x4);
    set_constant_by_offset(rmcv::mat4x4);

    set_constant_by_offset(uint64_t);

#undef set_constant_by_offset

    bool ParameterBlock::setTexture(const std::string& name, const Texture::SharedPtr& pTexture)
    {
        auto var = getRootVar()[name];
        return var.setTexture(pTexture);
    }

    bool ParameterBlock::setTexture(const BindLocation& bindLocation, const Texture::SharedPtr& pTexture)
    {
        const auto& bindingInfo = mpReflector->getResourceRangeBindingInfo(bindLocation.getResourceRangeIndex());
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        if (isUavType(bindLocation.getType()))
        {
            auto pUAV = pTexture ? pTexture->getUAV() : UnorderedAccessView::getNullView(bindingInfo.dimension);
            mpShaderObject->setResource(gfxOffset, pUAV->getApiHandle());
            mUAVs[gfxOffset] = pUAV;
            mResources[gfxOffset] = pTexture;
        }
        else if (isSrvType(bindLocation.getType()))
        {
            auto pSRV = pTexture ? pTexture->getSRV() : ShaderResourceView::getNullView(bindingInfo.dimension);
            mpShaderObject->setResource(gfxOffset, pSRV->getApiHandle());
            mSRVs[gfxOffset] = pSRV;
            mResources[gfxOffset] = pTexture;
        }
        else
        {
            logError("Error trying to bind resource to non SRV/UAV variable. Ignoring call.");
            return false;
        }
        return true;
    }

    Texture::SharedPtr ParameterBlock::getTexture(const std::string& name) const
    {
        auto var = getRootVar()[name];
        return var.getTexture();
    }

    Texture::SharedPtr ParameterBlock::getTexture(const BindLocation& bindLocation) const
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        if (isUavType(bindLocation.getType()))
        {
            auto iter = mUAVs.find(gfxOffset);
            if (iter == mUAVs.end()) return nullptr;
            auto pResource = iter->second->getResource();
            return pResource ? pResource->asTexture() : nullptr;
        }
        else if (isSrvType(bindLocation.getType()))
        {
            auto iter = mSRVs.find(gfxOffset);
            if (iter == mSRVs.end()) return nullptr;
            auto pResource = iter->second->getResource();
            return pResource ? pResource->asTexture() : nullptr;
        }
        else
        {
            logError("Error trying to bind resource to non SRV/UAV variable. Ignoring call.");
            return nullptr;
        }
    }

    bool ParameterBlock::setSrv(const BindLocation& bindLocation, const ShaderResourceView::SharedPtr& pSrv)
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        if (isSrvType(bindLocation.getType()))
        {
            mpShaderObject->setResource(gfxOffset, pSrv ? pSrv->getApiHandle() : nullptr);
            mSRVs[gfxOffset] = pSrv;
            mResources[gfxOffset] = pSrv ? pSrv->getResource() : nullptr;
        }
        else
        {
            logError("Error trying to bind SRV to a non SRV variable. Ignoring call.");
            return false;
        }
        return true;
    }

    bool ParameterBlock::setUav(const BindLocation& bindLocation, const UnorderedAccessView::SharedPtr& pUav)
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        if (isUavType(bindLocation.getType()))
        {
            mpShaderObject->setResource(gfxOffset, pUav ? pUav->getApiHandle() : nullptr);
            mUAVs[gfxOffset] = pUav;
            mResources[gfxOffset] = pUav ? pUav->getResource() : nullptr;
        }
        else
        {
            logError("Error trying to bind UAV to a non UAV variable. Ignoring call.");
            return false;
        }
        return true;
    }

    bool ParameterBlock::setAccelerationStructure(const BindLocation& bindLocation, const RtAccelerationStructure::SharedPtr& pAccl)
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        mAccelerationStructures[gfxOffset] = pAccl;
        return SLANG_SUCCEEDED(mpShaderObject->setResource(gfxOffset, pAccl ? pAccl->getApiHandle() : nullptr));
    }

    ShaderResourceView::SharedPtr ParameterBlock::getSrv(const BindLocation& bindLocation) const
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mSRVs.find(gfxOffset);
        if (iter == mSRVs.end()) return nullptr;
        return iter->second;
    }

    UnorderedAccessView::SharedPtr ParameterBlock::getUav(const BindLocation& bindLocation) const
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mUAVs.find(gfxOffset);
        if (iter == mUAVs.end()) return nullptr;
        return iter->second;
    }

    RtAccelerationStructure::SharedPtr ParameterBlock::getAccelerationStructure(const BindLocation& bindLocation) const
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mAccelerationStructures.find(gfxOffset);
        if (iter == mAccelerationStructures.end()) return nullptr;
        return iter->second;
    }

    bool ParameterBlock::setSampler(const std::string& name, const Sampler::SharedPtr& pSampler)
    {
        auto var = getRootVar()[name];
        return var.setSampler(pSampler);
    }

    bool ParameterBlock::setSampler(const BindLocation& bindLocation, const Sampler::SharedPtr& pSampler)
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        auto pBoundSampler = pSampler ? pSampler : Sampler::getDefault();
        mSamplers[gfxOffset] = pBoundSampler;
        return SLANG_SUCCEEDED(mpShaderObject->setSampler(gfxOffset, pBoundSampler->getApiHandle()));
    }

    const Sampler::SharedPtr& ParameterBlock::getSampler(const BindLocation& bindLocation) const
    {
        static Sampler::SharedPtr pNull = nullptr;

        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mSamplers.find(gfxOffset);
        if (iter == mSamplers.end())
        {
            return pNull;
        }
        return iter->second;
    }

    Sampler::SharedPtr ParameterBlock::getSampler(const std::string& name) const
    {
        auto var = getRootVar()[name];
        return var.getSampler();
    }

    size_t ParameterBlock::getSize() const
    {
        return mpShaderObject->getSize();
    }

    bool ParameterBlock::updateSpecialization() const
    {
        return true;
    }

    bool ParameterBlock::prepareDescriptorSets(CopyContext* pCopyContext)
    {
        // Insert necessary resource barriers for bound resources.
        for (auto& srv : mSRVs)
        {
            prepareResource(pCopyContext, srv.second->getResource().get(), false);
        }
        for (auto& uav : mUAVs)
        {
            prepareResource(pCopyContext, uav.second->getResource().get(), true);
        }
        for (auto& subObj : this->mParameterBlocks)
        {
            subObj.second->prepareDescriptorSets(pCopyContext);
        }
        return true;
    }

    const ParameterBlock::SharedPtr& ParameterBlock::getParameterBlock(uint32_t resourceRangeIndex, uint32_t arrayIndex) const
    {
        static ParameterBlock::SharedPtr pNull = nullptr;

        gfx::ShaderOffset gfxOffset = {};
        gfxOffset.bindingRangeIndex = resourceRangeIndex;
        gfxOffset.bindingArrayIndex = arrayIndex;
        auto iter = mParameterBlocks.find(gfxOffset);
        if (iter == mParameterBlocks.end())
        {
            return pNull;
        }
        return iter->second;
    }

    void ParameterBlock::collectSpecializationArgs(SpecializationArgs& ioArgs) const
    {
    }

    void ParameterBlock::markUniformDataDirty() const
    {
        throw "unimplemented";
    }

    void const* ParameterBlock::getRawData() const
    {
        return mpShaderObject->getRawData();
    }

    const Buffer::SharedPtr& ParameterBlock::getUnderlyingConstantBuffer() const
    {
        throw "unimplemented";
    }

#if FALCOR_HAS_CUDA
    void* ParameterBlock::getCUDAHostBuffer(size_t& outSize)
    {
        return nullptr;
    }

    void* ParameterBlock::getCUDADeviceBuffer(size_t& outSize)
    {
        return nullptr;
    }
#endif
}
