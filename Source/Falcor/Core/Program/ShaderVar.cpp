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
#include "ShaderVar.h"
#include "Core/API/ParameterBlock.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
ShaderVar::ShaderVar() : mpBlock(nullptr) {}
ShaderVar::ShaderVar(const ShaderVar& other) : mpBlock(other.mpBlock), mOffset(other.mOffset) {}
ShaderVar::ShaderVar(ParameterBlock* pObject, const TypedShaderVarOffset& offset) : mpBlock(pObject), mOffset(offset) {}
ShaderVar::ShaderVar(ParameterBlock* pObject) : mpBlock(pObject), mOffset(pObject->getElementType().get(), ShaderVarOffset::kZero) {}

//
// Navigation
//

ShaderVar ShaderVar::operator[](std::string_view name) const
{
    FALCOR_CHECK(isValid(), "Cannot lookup on invalid ShaderVar.");
    auto result = findMember(name);
    FALCOR_CHECK(result.isValid(), "No member named '{}' found.", name);
    return result;
}

ShaderVar ShaderVar::operator[](size_t index) const
{
    FALCOR_CHECK(isValid(), "Cannot lookup on invalid ShaderVar.");

    const ReflectionType* pType = getType();

    // If the user is applying `[]` to a `ShaderVar`
    // that represents a constant buffer (or parameter block)
    // then we assume they mean to look up an element
    // inside the buffer/block, and thus implicitly
    // dereference this `ShaderVar`.
    //
    if (auto pResourceType = pType->asResourceType())
    {
        switch (pResourceType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
            return getParameterBlock()->getRootVar()[index];
        default:
            break;
        }
    }

    if (auto pArrayType = pType->asArrayType())
    {
        auto elementCount = pArrayType->getElementCount();
        if (!elementCount || index < elementCount)
        {
            UniformShaderVarOffset elementUniformLocation = mOffset.getUniform() + index * pArrayType->getElementByteStride();
            ResourceShaderVarOffset elementResourceLocation(
                mOffset.getResource().getRangeIndex(),
                mOffset.getResource().getArrayIndex() * elementCount + ResourceShaderVarOffset::ArrayIndex(index)
            );
            TypedShaderVarOffset newOffset =
                TypedShaderVarOffset(pArrayType->getElementType(), ShaderVarOffset(elementUniformLocation, elementResourceLocation));
            return ShaderVar(mpBlock, newOffset);
        }
    }
    else if (auto pStructType = pType->asStructType())
    {
        if (index < pStructType->getMemberCount())
        {
            auto pMember = pStructType->getMember(index);
            // Need to apply the offsets from member
            TypedShaderVarOffset newOffset = TypedShaderVarOffset(pMember->getType(), mOffset + pMember->getBindLocation());
            return ShaderVar(mpBlock, newOffset);
        }
    }

    FALCOR_THROW("No element or member found at index {}", index);
}

ShaderVar ShaderVar::findMember(std::string_view name) const
{
    if (!isValid())
        return *this;
    const ReflectionType* pType = getType();

    // If the user is applying `[]` to a `ShaderVar`
    // that represents a constant buffer (or parameter block)
    // then we assume they mean to look up a member
    // inside the buffer/block, and thus implicitly
    // dereference this `ShaderVar`.
    //
    if (auto pResourceType = pType->asResourceType())
    {
        switch (pResourceType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
            return getParameterBlock()->getRootVar().findMember(name);
        default:
            break;
        }
    }

    if (auto pStructType = pType->asStructType())
    {
        if (auto pMember = pStructType->findMember(name))
        {
            // Need to apply the offsets from member
            TypedShaderVarOffset newOffset = TypedShaderVarOffset(pMember->getType(), mOffset + pMember->getBindLocation());
            return ShaderVar(mpBlock, newOffset);
        }
    }

    return ShaderVar();
}

ShaderVar ShaderVar::findMember(uint32_t index) const
{
    if (!isValid())
        return *this;
    const ReflectionType* pType = getType();

    // If the user is applying `[]` to a `ShaderVar`
    // that represents a constant buffer (or parameter block)
    // then we assume they mean to look up a member
    // inside the buffer/block, and thus implicitly
    // dereference this `ShaderVar`.
    //
    if (auto pResourceType = pType->asResourceType())
    {
        switch (pResourceType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
            return getParameterBlock()->getRootVar().findMember(index);
        default:
            break;
        }
    }

    if (auto pStructType = pType->asStructType())
    {
        if (index < pStructType->getMemberCount())
        {
            auto pMember = pStructType->getMember(index);

            // Need to apply the offsets from member
            TypedShaderVarOffset newOffset = TypedShaderVarOffset(pMember->getType(), mOffset + pMember->getBindLocation());
            return ShaderVar(mpBlock, newOffset);
        }
    }

    return ShaderVar();
}

//
// Variable assignment
//

void ShaderVar::setBlob(void const* data, size_t size) const
{
    // If the var is pointing at a constant buffer, then assume
    // the user actually means to write the blob *into* that buffer.
    //
    const ReflectionType* pType = getType();
    if (auto pResourceType = pType->asResourceType())
    {
        switch (pResourceType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
            return getParameterBlock()->getRootVar().setBlob(data, size);
        default:
            break;
        }
    }

    mpBlock->setBlob(mOffset, data, size);
}

//
// Resource binding
//

void ShaderVar::setBuffer(const ref<Buffer>& pBuffer) const
{
    mpBlock->setBuffer(mOffset, pBuffer);
}

ref<Buffer> ShaderVar::getBuffer() const
{
    return mpBlock->getBuffer(mOffset);
}

void ShaderVar::setTexture(const ref<Texture>& pTexture) const
{
    mpBlock->setTexture(mOffset, pTexture);
}

ref<Texture> ShaderVar::getTexture() const
{
    return mpBlock->getTexture(mOffset);
}

void ShaderVar::setSrv(const ref<ShaderResourceView>& pSrv) const
{
    mpBlock->setSrv(mOffset, pSrv);
}

ref<ShaderResourceView> ShaderVar::getSrv() const
{
    return mpBlock->getSrv(mOffset);
}

void ShaderVar::setUav(const ref<UnorderedAccessView>& pUav) const
{
    mpBlock->setUav(mOffset, pUav);
}

ref<UnorderedAccessView> ShaderVar::getUav() const
{
    return mpBlock->getUav(mOffset);
}

void ShaderVar::setAccelerationStructure(const ref<RtAccelerationStructure>& pAccl) const
{
    mpBlock->setAccelerationStructure(mOffset, pAccl);
}

ref<RtAccelerationStructure> ShaderVar::getAccelerationStructure() const
{
    return mpBlock->getAccelerationStructure(mOffset);
}

void ShaderVar::setSampler(const ref<Sampler>& pSampler) const
{
    mpBlock->setSampler(mOffset, pSampler);
}

ref<Sampler> ShaderVar::getSampler() const
{
    return mpBlock->getSampler(mOffset);
}

void ShaderVar::setParameterBlock(const ref<ParameterBlock>& pBlock) const
{
    mpBlock->setParameterBlock(mOffset, pBlock);
}

ref<ParameterBlock> ShaderVar::getParameterBlock() const
{
    return mpBlock->getParameterBlock(mOffset);
}

//
// Offset access
//

ShaderVar ShaderVar::operator[](const TypedShaderVarOffset& offset) const
{
    if (!isValid())
        return *this;
    const ReflectionType* pType = getType();

    // If the user is applying `[]` to a `ShaderVar`
    // that represents a constant buffer (or parameter block)
    // then we assume they mean to look up an offset
    // inside the buffer/block, and thus implicitly
    // dereference this `ShaderVar`
    if (auto pResourceType = pType->asResourceType())
    {
        switch (pResourceType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
            return getParameterBlock()->getRootVar()[offset];
        default:
            break;
        }
    }

    return ShaderVar(mpBlock, TypedShaderVarOffset(offset.getType(), mOffset + offset));
}

ShaderVar ShaderVar::operator[](const UniformShaderVarOffset& loc) const
{
    if (!isValid())
        return *this;
    const ReflectionType* pType = getType();

    // If the user is applying `[]` to a `ShaderVar`
    // that represents a constant buffer (or parameter block)
    // then we assume they mean to look up an offset
    // inside the buffer/block, and thus implicitly
    // dereference this `ShaderVar`.
    //
    if (auto pResourceType = pType->asResourceType())
    {
        switch (pResourceType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
            return getParameterBlock()->getRootVar()[loc];
        default:
            break;
        }
    }

    auto byteOffset = loc.getByteOffset();
    if (byteOffset == 0)
        return *this;

    if (auto pArrayType = pType->asArrayType())
    {
        auto pElementType = pArrayType->getElementType();
        auto elementCount = pArrayType->getElementCount();
        auto elementStride = pArrayType->getElementByteStride();

        auto elementIndex = byteOffset / elementStride;
        auto offsetIntoElement = byteOffset % elementStride;

        TypedShaderVarOffset elementOffset =
            TypedShaderVarOffset(pElementType, ShaderVarOffset(mOffset.getUniform() + elementIndex * elementStride, mOffset.getResource()));
        ShaderVar elementCursor(mpBlock, elementOffset);
        return elementCursor[UniformShaderVarOffset(offsetIntoElement)];
    }
    else if (auto pStructType = pType->asStructType())
    {
        // We want to search for a member matching this offset
        //
        // TODO: A binary search should be preferred to the linear
        // search here.

        auto memberCount = pStructType->getMemberCount();
        for (uint32_t m = 0; m < memberCount; ++m)
        {
            auto pMember = pStructType->getMember(m);
            auto memberByteOffset = pMember->getByteOffset();
            auto memberByteSize = pMember->getType()->getByteSize();

            if (byteOffset < memberByteOffset)
                continue;
            if (byteOffset >= memberByteOffset + memberByteSize)
                continue;

            auto offsetIntoMember = byteOffset - memberByteOffset;
            TypedShaderVarOffset memberOffset = TypedShaderVarOffset(pMember->getType(), mOffset + pMember->getBindLocation());
            ShaderVar memberCursor(mpBlock, memberOffset);
            return memberCursor[UniformShaderVarOffset(offsetIntoMember)];
        }
    }

    FALCOR_THROW("No element or member found at offset {}", byteOffset);
}

void const* ShaderVar::getRawData() const
{
    return (uint8_t*)(mpBlock->getRawData()) + mOffset.getUniform().getByteOffset();
}

void ShaderVar::setImpl(const ref<Texture>& pTexture) const
{
    mpBlock->setTexture(mOffset, pTexture);
}

void ShaderVar::setImpl(const ref<Sampler>& pSampler) const
{
    mpBlock->setSampler(mOffset, pSampler);
}

void ShaderVar::setImpl(const ref<Buffer>& pBuffer) const
{
    mpBlock->setBuffer(mOffset, pBuffer);
}

void ShaderVar::setImpl(const ref<ParameterBlock>& pBlock) const
{
    mpBlock->setParameterBlock(mOffset, pBlock);
}

FALCOR_SCRIPT_BINDING(ShaderVar)
{
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Buffer)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Texture)

    pybind11::class_<ShaderVar> shaderVar(m, "ShaderVar");

    shaderVar.def("__getitem__", [](ShaderVar& self, std::string_view name) { return self[name]; });
    shaderVar.def("__getattr__", [](ShaderVar& self, std::string_view name) { return self[name]; });

#define def_setter(type)                                                                                          \
    shaderVar.def("__setitem__", [](ShaderVar& self, std::string_view name, type value) { self[name] = value; }); \
    shaderVar.def("__setattr__", [](ShaderVar& self, std::string_view name, type value) { self[name] = value; });

    def_setter(ref<Buffer>);
    def_setter(ref<Texture>);
    def_setter(ref<Sampler>);

    def_setter(uint2);
    def_setter(uint3);
    def_setter(uint4);

    def_setter(int2);
    def_setter(int3);
    def_setter(int4);

    def_setter(bool);
    def_setter(bool2);
    def_setter(bool3);
    def_setter(bool4);

    def_setter(float2);
    def_setter(float3);
    def_setter(float4);

    def_setter(float3x4);
    def_setter(float4x4);

#undef def_setter

    // We need to handle integers and floats specially.
    // Python only has an `int` and `float` type that can have different bit-width.
    // We use reflection data to convert the python types to the correct types before assigning.

    auto set_int = [](ShaderVar& self, std::string_view name, pybind11::int_ value)
    {
        const ReflectionBasicType* basicType = self[name].getType()->unwrapArray()->asBasicType();
        FALCOR_CHECK(basicType, "Error trying to set a variable that is not a basic type.");
        switch (basicType->getType())
        {
        case ReflectionBasicType::Type::Int:
            self[name] = value.cast<int32_t>();
            break;
        case ReflectionBasicType::Type::Uint:
            self[name] = value.cast<uint32_t>();
            break;
        default:
            FALCOR_THROW("Error trying to set a variable that is not an integer type.");
            break;
        }
    };

    shaderVar.def("__setitem__", set_int);
    shaderVar.def("__setattr__", set_int);

    auto set_float = [](ShaderVar& self, std::string_view name, pybind11::float_ value)
    {
        const ReflectionBasicType* basicType = self[name].getType()->unwrapArray()->asBasicType();
        FALCOR_CHECK(basicType, "Error trying to set a variable that is not a basic type.");
        switch (basicType->getType())
        {
        case ReflectionBasicType::Type::Float:
            self[name] = value.cast<float>();
            break;
        default:
            FALCOR_THROW("Error trying to set a variable that is not an float type.");
            break;
        }
    };

    shaderVar.def("__setitem__", set_float);
    shaderVar.def("__setattr__", set_float);
}

} // namespace Falcor
