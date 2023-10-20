/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "ParameterBlock.h"
#include "Device.h"
#include "CopyContext.h"
#include "GFXAPI.h"
#include "Core/Error.h"
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

bool isSrvType(const ReflectionType* pType)
{
    FALCOR_ASSERT(pType);
    auto resourceType = pType->unwrapArray()->asResourceType();
    if (!resourceType || resourceType->getType() == ReflectionResourceType::Type::Sampler ||
        resourceType->getType() == ReflectionResourceType::Type::ConstantBuffer)
        return false;

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

bool isUavType(const ReflectionType* pType)
{
    FALCOR_ASSERT(pType);
    auto resourceType = pType->unwrapArray()->asResourceType();
    if (!resourceType || resourceType->getType() == ReflectionResourceType::Type::Sampler ||
        resourceType->getType() == ReflectionResourceType::Type::ConstantBuffer)
        return false;

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

bool isSamplerType(const ReflectionType* pType)
{
    FALCOR_ASSERT(pType);
    auto resourceType = pType->unwrapArray()->asResourceType();
    return (resourceType && resourceType->getType() == ReflectionResourceType::Type::Sampler);
}

bool isAccelerationStructureType(const ReflectionType* pType)
{
    FALCOR_ASSERT(pType);
    auto resourceType = pType->unwrapArray()->asResourceType();
    return (resourceType && resourceType->getType() == ReflectionResourceType::Type::AccelerationStructure);
}

bool isParameterBlockType(const ReflectionType* pType)
{
    FALCOR_ASSERT(pType);
    auto resourceType = pType->unwrapArray()->asResourceType();
    // Parameter blocks are currently classified as constant buffers.
    // See getResourceType() in ProgramReflection.cpp
    return (resourceType && resourceType->getType() == ReflectionResourceType::Type::ConstantBuffer);
}

bool isConstantBufferType(const ReflectionType* pType)
{
    FALCOR_ASSERT(pType);
    auto resourceType = pType->unwrapArray()->asResourceType();
    return (resourceType && resourceType->getType() == ReflectionResourceType::Type::ConstantBuffer);
}

} // namespace

ref<ParameterBlock> ParameterBlock::create(
    ref<Device> pDevice,
    const ref<const ProgramVersion>& pProgramVersion,
    const ref<const ReflectionType>& pElementType
)
{
    FALCOR_CHECK(pElementType, "Can't create a parameter block without type information");
    auto pReflection = ParameterBlockReflection::create(pProgramVersion.get(), pElementType);
    return create(pDevice, pReflection);
}

ref<ParameterBlock> ParameterBlock::create(ref<Device> pDevice, const ref<const ParameterBlockReflection>& pReflection)
{
    FALCOR_ASSERT(pReflection);
    // TODO(@skallweit) we convert the weak pointer to a shared pointer here because we tie
    // the lifetime of the parameter block to the lifetime of the program version.
    // The ownership for programs/versions/kernels and parameter blocks needs to be revisited.
    return ref<ParameterBlock>(new ParameterBlock(pDevice, ref<const ProgramVersion>(pReflection->getProgramVersion()), pReflection));
}

ref<ParameterBlock> ParameterBlock::create(
    ref<Device> pDevice,
    const ref<const ProgramVersion>& pProgramVersion,
    const std::string& typeName
)
{
    FALCOR_ASSERT(pProgramVersion);
    return ParameterBlock::create(pDevice, pProgramVersion, pProgramVersion->getReflector()->findType(typeName));
}

ShaderVar ParameterBlock::getRootVar() const
{
    return ShaderVar(const_cast<ParameterBlock*>(this));
}

ShaderVar ParameterBlock::findMember(std::string_view varName) const
{
    return getRootVar().findMember(varName);
}

ShaderVar ParameterBlock::findMember(uint32_t index) const
{
    return getRootVar().findMember(index);
}

size_t ParameterBlock::getElementSize() const
{
    return mpReflector->getElementType()->getByteSize();
}

TypedShaderVarOffset ParameterBlock::getVariableOffset(std::string_view varName) const
{
    return getElementType()->getZeroOffset()[varName];
}

void ParameterBlock::createConstantBuffers(const ShaderVar& var)
{
    auto pType = var.getType();
    if (pType->getResourceRangeCount() == 0)
        return;

    switch (pType->getKind())
    {
    case ReflectionType::Kind::Struct:
    {
        auto pStructType = pType->asStructType();
        uint32_t memberCount = pStructType->getMemberCount();
        for (uint32_t i = 0; i < memberCount; ++i)
            createConstantBuffers(var[i]);
    }
    break;
    case ReflectionType::Kind::Resource:
    {
        auto pResourceType = pType->asResourceType();
        switch (pResourceType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
        {
            auto pCB = ParameterBlock::create(ref<Device>(mpDevice), pResourceType->getParameterBlockReflector());
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

void ParameterBlock::prepareResource(CopyContext* pContext, Resource* pResource, bool isUav)
{
    if (!pResource)
        return;

    // If it's a buffer with a UAV counter, insert a UAV barrier
    const Buffer* pBuffer = pResource->asBuffer().get();
    if (isUav && pBuffer && pBuffer->getUAVCounter())
    {
        pContext->resourceBarrier(pBuffer->getUAVCounter().get(), Resource::State::UnorderedAccess);
        pContext->uavBarrier(pBuffer->getUAVCounter().get());
    }

    bool insertBarrier = true;
    insertBarrier = (is_set(pResource->getBindFlags(), ResourceBindFlags::AccelerationStructure) == false);
    if (insertBarrier)
    {
        insertBarrier = !pContext->resourceBarrier(pResource, isUav ? Resource::State::UnorderedAccess : Resource::State::ShaderResource);
    }

    // Insert UAV barrier automatically if the resource is an UAV that is already in UnorderedAccess state.
    // Otherwise the user would have to insert barriers explicitly between passes accessing UAVs, which is easily forgotten.
    if (insertBarrier && isUav)
        pContext->uavBarrier(pResource);
}

ParameterBlock::~ParameterBlock() {}

ParameterBlock::ParameterBlock(ref<Device> pDevice, const ref<const ProgramReflection>& pReflector)
    : mpDevice(pDevice.get()), mpProgramVersion(pReflector->getProgramVersion()), mpReflector(pReflector->getDefaultParameterBlock())
{
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createMutableRootShaderObject(
        pReflector->getProgramVersion()->getKernels(mpDevice, nullptr)->getGfxProgram(), mpShaderObject.writeRef()
    ));
    initializeResourceBindings();
    createConstantBuffers(getRootVar());
}

ParameterBlock::ParameterBlock(
    ref<Device> pDevice,
    const ref<const ProgramVersion>& pProgramVersion,
    const ref<const ParameterBlockReflection>& pReflection
)
    : mpDevice(pDevice.get()), mpProgramVersion(pProgramVersion), mpReflector(pReflection)
{
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createMutableShaderObjectFromTypeLayout(
        pReflection->getElementType()->getSlangTypeLayout(), mpShaderObject.writeRef()
    ));
    initializeResourceBindings();
    createConstantBuffers(getRootVar());
}

void ParameterBlock::initializeResourceBindings()
{
    for (uint32_t i = 0; i < mpReflector->getResourceRangeCount(); i++)
    {
        auto info = mpReflector->getResourceRangeBindingInfo(i);
        auto range = mpReflector->getResourceRange(i);
        for (uint32_t arrayIndex = 0; arrayIndex < range.count; arrayIndex++)
        {
            gfx::ShaderOffset offset = {};
            offset.bindingRangeIndex = i;
            offset.bindingArrayIndex = arrayIndex;
            switch (range.descriptorType)
            {
            case ShaderResourceType::Sampler:
                mpShaderObject->setSampler(offset, mpDevice->getDefaultSampler()->getGfxSamplerState());
                break;
            case ShaderResourceType::TextureSrv:
            case ShaderResourceType::TextureUav:
            case ShaderResourceType::RawBufferSrv:
            case ShaderResourceType::RawBufferUav:
            case ShaderResourceType::TypedBufferSrv:
            case ShaderResourceType::TypedBufferUav:
            case ShaderResourceType::StructuredBufferUav:
            case ShaderResourceType::StructuredBufferSrv:
            case ShaderResourceType::AccelerationStructureSrv:
                mpShaderObject->setResource(offset, nullptr);
                break;
            }
        }
    }
}

void ParameterBlock::setBlob(const void* pSrc, const BindLocation& bindLocation, size_t size)
{
    if (!isConstantBufferType(bindLocation.getType()))
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        FALCOR_GFX_CALL(mpShaderObject->setData(gfxOffset, pSrc, size));
    }
    else
    {
        FALCOR_THROW("Error trying to set a blob to a non constant buffer variable.");
    }
}

void ParameterBlock::setBlob(const void* pSrc, size_t offset, size_t size)
{
    gfx::ShaderOffset gfxOffset = {};
    gfxOffset.uniformOffset = offset;
    FALCOR_GFX_CALL(mpShaderObject->setData(gfxOffset, pSrc, size));
}

//
// Uniforms
//

template<typename T>
void setVariableInternal(
    ParameterBlock* pBlock,
    const ParameterBlock::BindLocation& bindLocation,
    const T& value,
    ReflectionBasicType::Type type,
    ReflectionBasicType::Type implicitType = ReflectionBasicType::Type::Unknown
)
{
    const ReflectionBasicType* basicType = bindLocation.getType()->unwrapArray()->asBasicType();
    if (!basicType)
        FALCOR_THROW("Error trying to set a variable that is not a basic type.");
    ReflectionBasicType::Type expectedType = basicType->getType();
    // Check types. Allow implicit conversions from signed to unsigned types.
    if (type != expectedType && implicitType != expectedType)
        FALCOR_THROW(
            "Error trying to set a variable with a different type than the one in the program (expected {}, got {}).",
            enumToString(expectedType),
            enumToString(type)
        );
    size_t size = sizeof(T);
    size_t expectedSize = basicType->getByteSize();
    if (size != expectedSize)
        FALCOR_THROW(
            "Error trying to set a variable with a different size than the one in the program (expected {} bytes, got {}).",
            expectedSize,
            size
        );
    auto gfxOffset = getGFXShaderOffset(bindLocation);
    FALCOR_GFX_CALL(pBlock->getShaderObject()->setData(gfxOffset, &value, size));
}

template<typename T>
void ParameterBlock::setVariable(const BindLocation& bindLocation, const T& value)
{
    FALCOR_UNIMPLEMENTED();
}

#define DEFINE_SET_VARIABLE(ctype, basicType, implicitType)                                           \
    template<>                                                                                        \
    FALCOR_API void ParameterBlock::setVariable(const BindLocation& bindLocation, const ctype& value) \
    {                                                                                                 \
        setVariableInternal<ctype>(this, bindLocation, value, basicType, implicitType);               \
    }

DEFINE_SET_VARIABLE(uint32_t, ReflectionBasicType::Type::Uint, ReflectionBasicType::Type::Int);
DEFINE_SET_VARIABLE(uint2, ReflectionBasicType::Type::Uint2, ReflectionBasicType::Type::Int2);
DEFINE_SET_VARIABLE(uint3, ReflectionBasicType::Type::Uint3, ReflectionBasicType::Type::Int3);
DEFINE_SET_VARIABLE(uint4, ReflectionBasicType::Type::Uint4, ReflectionBasicType::Type::Int4);

DEFINE_SET_VARIABLE(int32_t, ReflectionBasicType::Type::Int, ReflectionBasicType::Type::Uint);
DEFINE_SET_VARIABLE(int2, ReflectionBasicType::Type::Int2, ReflectionBasicType::Type::Uint2);
DEFINE_SET_VARIABLE(int3, ReflectionBasicType::Type::Int3, ReflectionBasicType::Type::Uint3);
DEFINE_SET_VARIABLE(int4, ReflectionBasicType::Type::Int4, ReflectionBasicType::Type::Uint4);

DEFINE_SET_VARIABLE(float, ReflectionBasicType::Type::Float, ReflectionBasicType::Type::Unknown);
DEFINE_SET_VARIABLE(float2, ReflectionBasicType::Type::Float2, ReflectionBasicType::Type::Unknown);
DEFINE_SET_VARIABLE(float3, ReflectionBasicType::Type::Float3, ReflectionBasicType::Type::Unknown);
DEFINE_SET_VARIABLE(float4, ReflectionBasicType::Type::Float4, ReflectionBasicType::Type::Unknown);

// DEFINE_SET_VARIABLE(float1x4, ReflectionBasicType::Type::Float1x4, ReflectionBasicType::Type::Unknown);
DEFINE_SET_VARIABLE(float2x4, ReflectionBasicType::Type::Float2x4, ReflectionBasicType::Type::Unknown);
DEFINE_SET_VARIABLE(float3x4, ReflectionBasicType::Type::Float3x4, ReflectionBasicType::Type::Unknown);
DEFINE_SET_VARIABLE(float4x4, ReflectionBasicType::Type::Float4x4, ReflectionBasicType::Type::Unknown);

DEFINE_SET_VARIABLE(uint64_t, ReflectionBasicType::Type::Uint64, ReflectionBasicType::Type::Int64);

#undef DEFINE_SET_VARIABLE

// Template specialization to allow setting booleans on a parameter block.
// On the host side a bool is 1B and the device 4B. We cast bools to 32-bit integers here.
// Note that this applies to our boolN vectors as well, which are currently 1B per element.

template<>
FALCOR_API void ParameterBlock::setVariable(const BindLocation& bindLocation, const bool& value)
{
    uint v = value ? 1 : 0;
    setVariableInternal(this, bindLocation, v, ReflectionBasicType::Type::Bool);
}

template<>
FALCOR_API void ParameterBlock::setVariable(const BindLocation& bindLocation, const bool2& value)
{
    uint2 v = {value.x ? 1 : 0, value.y ? 1 : 0};
    setVariableInternal(this, bindLocation, v, ReflectionBasicType::Type::Bool2);
}

template<>
FALCOR_API void ParameterBlock::setVariable(const BindLocation& bindLocation, const bool3& value)
{
    uint3 v = {value.x ? 1 : 0, value.y ? 1 : 0, value.z ? 1 : 0};
    setVariableInternal(this, bindLocation, v, ReflectionBasicType::Type::Bool3);
}

template<>
FALCOR_API void ParameterBlock::setVariable(const BindLocation& bindLocation, const bool4& value)
{
    uint4 v = {value.x ? 1 : 0, value.y ? 1 : 0, value.z ? 1 : 0, value.w ? 1 : 0};
    setVariableInternal(this, bindLocation, v, ReflectionBasicType::Type::Bool4);
}

//
// Buffer
//

void ParameterBlock::setBuffer(std::string_view name, const ref<Buffer>& pBuffer)
{
    getRootVar()[name].setBuffer(pBuffer);
}

void ParameterBlock::setBuffer(const BindLocation& bindLoc, const ref<Buffer>& pBuffer)
{
    gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLoc);
    if (isUavType(bindLoc.getType()))
    {
        if (pBuffer && !is_set(pBuffer->getBindFlags(), ResourceBindFlags::UnorderedAccess))
            FALCOR_THROW("Trying to bind buffer '{}' created without UnorderedAccess flag as a UAV.", pBuffer->getName());
        auto pUAV = pBuffer ? pBuffer->getUAV() : nullptr;
        mpShaderObject->setResource(gfxOffset, pUAV ? pUAV->getGfxResourceView() : nullptr);
        mUAVs[gfxOffset] = pUAV;
        mResources[gfxOffset] = pBuffer;
    }
    else if (isSrvType(bindLoc.getType()))
    {
        if (pBuffer && !is_set(pBuffer->getBindFlags(), ResourceBindFlags::ShaderResource))
            FALCOR_THROW("Trying to bind buffer '{}' created without ShaderResource flag as an SRV.", pBuffer->getName());
        auto pSRV = pBuffer ? pBuffer->getSRV() : nullptr;
        mpShaderObject->setResource(gfxOffset, pSRV ? pSRV->getGfxResourceView() : nullptr);
        mSRVs[gfxOffset] = pSRV;
        mResources[gfxOffset] = pBuffer;
    }
    else
    {
        FALCOR_THROW("Error trying to bind buffer to a non SRV/UAV variable.");
    }
}

ref<Buffer> ParameterBlock::getBuffer(std::string_view name) const
{
    return getRootVar()[name].getBuffer();
}

ref<Buffer> ParameterBlock::getBuffer(const BindLocation& bindLoc) const
{
    gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLoc);
    if (isUavType(bindLoc.getType()))
    {
        auto iter = mUAVs.find(gfxOffset);
        if (iter == mUAVs.end())
            return nullptr;
        auto pResource = iter->second->getResource();
        return pResource ? pResource->asBuffer() : nullptr;
    }
    else if (isSrvType(bindLoc.getType()))
    {
        auto iter = mSRVs.find(gfxOffset);
        if (iter == mSRVs.end())
            return nullptr;
        auto pResource = iter->second->getResource();
        return pResource ? pResource->asBuffer() : nullptr;
    }
    else
    {
        FALCOR_THROW("Error trying to get buffer from a non SRV/UAV variable.");
    }
}

//
// Texture
//

void ParameterBlock::setTexture(std::string_view name, const ref<Texture>& pTexture)
{
    getRootVar()[name].setTexture(pTexture);
}

void ParameterBlock::setTexture(const BindLocation& bindLocation, const ref<Texture>& pTexture)
{
    gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
    if (isUavType(bindLocation.getType()))
    {
        if (pTexture && !is_set(pTexture->getBindFlags(), ResourceBindFlags::UnorderedAccess))
            FALCOR_THROW("Trying to bind texture '{}' created without UnorderedAccess flag as a UAV.", pTexture->getName());
        auto pUAV = pTexture ? pTexture->getUAV() : nullptr;
        mpShaderObject->setResource(gfxOffset, pUAV ? pUAV->getGfxResourceView() : nullptr);
        mUAVs[gfxOffset] = pUAV;
        mResources[gfxOffset] = pTexture;
    }
    else if (isSrvType(bindLocation.getType()))
    {
        if (pTexture && !is_set(pTexture->getBindFlags(), ResourceBindFlags::ShaderResource))
            FALCOR_THROW("Trying to bind texture '{}' created without ShaderResource flag as an SRV.", pTexture->getName());
        auto pSRV = pTexture ? pTexture->getSRV() : nullptr;
        mpShaderObject->setResource(gfxOffset, pSRV ? pSRV->getGfxResourceView() : nullptr);
        mSRVs[gfxOffset] = pSRV;
        mResources[gfxOffset] = pTexture;
    }
    else
    {
        FALCOR_THROW("Error trying to bind texture to a non SRV/UAV variable.");
    }
}

ref<Texture> ParameterBlock::getTexture(std::string_view name) const
{
    return getRootVar()[name].getTexture();
}

ref<Texture> ParameterBlock::getTexture(const BindLocation& bindLocation) const
{
    gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
    if (isUavType(bindLocation.getType()))
    {
        auto iter = mUAVs.find(gfxOffset);
        if (iter == mUAVs.end())
            return nullptr;
        auto pResource = iter->second->getResource();
        return pResource ? pResource->asTexture() : nullptr;
    }
    else if (isSrvType(bindLocation.getType()))
    {
        auto iter = mSRVs.find(gfxOffset);
        if (iter == mSRVs.end())
            return nullptr;
        auto pResource = iter->second->getResource();
        return pResource ? pResource->asTexture() : nullptr;
    }
    else
    {
        FALCOR_THROW("Error trying to get texture from a non SRV/UAV variable.");
    }
}

//
// ResourceView
//

void ParameterBlock::setSrv(const BindLocation& bindLocation, const ref<ShaderResourceView>& pSrv)
{
    if (isSrvType(bindLocation.getType()))
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        mpShaderObject->setResource(gfxOffset, pSrv ? pSrv->getGfxResourceView() : nullptr);
        mSRVs[gfxOffset] = pSrv;
        // Note: The resource view does not hold a strong reference to the resource, so we need to keep it alive here.
        mResources[gfxOffset] = ref<Resource>(pSrv ? pSrv->getResource() : nullptr);
    }
    else
    {
        FALCOR_THROW("Error trying to bind an SRV to a non SRV variable.");
    }
}

ref<ShaderResourceView> ParameterBlock::getSrv(const BindLocation& bindLocation) const
{
    if (isSrvType(bindLocation.getType()))
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mSRVs.find(gfxOffset);
        if (iter == mSRVs.end())
            return nullptr;
        return iter->second;
    }
    else
    {
        FALCOR_THROW("Error trying to get an SRV from a non SRV variable.");
    }
}

void ParameterBlock::setUav(const BindLocation& bindLocation, const ref<UnorderedAccessView>& pUav)
{
    if (isUavType(bindLocation.getType()))
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        mpShaderObject->setResource(gfxOffset, pUav ? pUav->getGfxResourceView() : nullptr);
        mUAVs[gfxOffset] = pUav;
        // Note: The resource view does not hold a strong reference to the resource, so we need to keep it alive here.
        mResources[gfxOffset] = ref<Resource>(pUav ? pUav->getResource() : nullptr);
    }
    else
    {
        FALCOR_THROW("Error trying to bind a UAV to a non UAV variable.");
    }
}

ref<UnorderedAccessView> ParameterBlock::getUav(const BindLocation& bindLocation) const
{
    if (isUavType(bindLocation.getType()))
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mUAVs.find(gfxOffset);
        if (iter == mUAVs.end())
            return nullptr;
        return iter->second;
    }
    else
    {
        FALCOR_THROW("Error trying to get a UAV from a non UAV variable.");
    }
}

void ParameterBlock::setAccelerationStructure(const BindLocation& bindLocation, const ref<RtAccelerationStructure>& pAccl)
{
    if (isAccelerationStructureType(bindLocation.getType()))
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        mAccelerationStructures[gfxOffset] = pAccl;
        FALCOR_GFX_CALL(mpShaderObject->setResource(gfxOffset, pAccl ? pAccl->getGfxAccelerationStructure() : nullptr));
    }
    else
    {
        FALCOR_THROW("Error trying to bind an acceleration structure to a non acceleration structure variable.");
    }
}

ref<RtAccelerationStructure> ParameterBlock::getAccelerationStructure(const BindLocation& bindLocation) const
{
    if (isAccelerationStructureType(bindLocation.getType()))
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mAccelerationStructures.find(gfxOffset);
        if (iter == mAccelerationStructures.end())
            return nullptr;
        return iter->second;
    }
    else
    {
        FALCOR_THROW("Error trying to get an acceleration structure from a non acceleration structure variable.");
    }
}

//
// Sampler
//

void ParameterBlock::setSampler(std::string_view name, const ref<Sampler>& pSampler)
{
    getRootVar()[name].setSampler(pSampler);
}

void ParameterBlock::setSampler(const BindLocation& bindLocation, const ref<Sampler>& pSampler)
{
    if (isSamplerType(bindLocation.getType()))
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        const ref<Sampler>& pBoundSampler = pSampler ? pSampler : mpDevice->getDefaultSampler();
        mSamplers[gfxOffset] = pBoundSampler;
        FALCOR_GFX_CALL(mpShaderObject->setSampler(gfxOffset, pBoundSampler->getGfxSamplerState()));
    }
    else
    {
        FALCOR_THROW("Error trying to bind a sampler to a non sampler variable.");
    }
}

ref<Sampler> ParameterBlock::getSampler(std::string_view name) const
{
    return getRootVar()[name].getSampler();
}

ref<Sampler> ParameterBlock::getSampler(const BindLocation& bindLocation) const
{
    if (isSamplerType(bindLocation.getType()))
    {
        gfx::ShaderOffset gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mSamplers.find(gfxOffset);
        if (iter == mSamplers.end())
            return nullptr;
        return iter->second;
    }
    else
    {
        FALCOR_THROW("Error trying to get a sampler from a non sampler variable.");
    }
}

//
// ParameterBlock
//

void ParameterBlock::setParameterBlock(std::string_view name, const ref<ParameterBlock>& pBlock)
{
    getRootVar()[name].setParameterBlock(pBlock);
}

void ParameterBlock::setParameterBlock(const BindLocation& bindLocation, const ref<ParameterBlock>& pBlock)
{
    if (isParameterBlockType(bindLocation.getType()))
    {
        auto gfxOffset = getGFXShaderOffset(bindLocation);
        mParameterBlocks[gfxOffset] = pBlock;
        FALCOR_GFX_CALL(mpShaderObject->setObject(gfxOffset, pBlock ? pBlock->mpShaderObject : nullptr));
    }
    else
    {
        FALCOR_THROW("Error trying to bind a parameter block to a non parameter block variable.");
    }
}

ref<ParameterBlock> ParameterBlock::getParameterBlock(std::string_view name) const
{
    return getRootVar()[name].getParameterBlock();
}

ref<ParameterBlock> ParameterBlock::getParameterBlock(const BindLocation& bindLocation) const
{
    if (isParameterBlockType(bindLocation.getType()))
    {
        auto gfxOffset = getGFXShaderOffset(bindLocation);
        auto iter = mParameterBlocks.find(gfxOffset);
        if (iter == mParameterBlocks.end())
            return nullptr;
        return iter->second;
    }
    else
    {
        FALCOR_THROW("Error trying to get a parameter block from a non parameter block variable.");
    }
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
        prepareResource(pCopyContext, srv.second ? srv.second->getResource() : nullptr, false);
    }
    for (auto& uav : mUAVs)
    {
        prepareResource(pCopyContext, uav.second ? uav.second->getResource() : nullptr, true);
    }
    for (auto& subObj : this->mParameterBlocks)
    {
        subObj.second->prepareDescriptorSets(pCopyContext);
    }
    return true;
}

void ParameterBlock::collectSpecializationArgs(SpecializationArgs& ioArgs) const {}

void const* ParameterBlock::getRawData() const
{
    return mpShaderObject->getRawData();
}

} // namespace Falcor
