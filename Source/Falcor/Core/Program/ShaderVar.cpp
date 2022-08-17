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
#include "ShaderVar.h"
#include "Core/API/ParameterBlock.h"

namespace Falcor
{
    ShaderVar::ShaderVar() : mpBlock(nullptr) {}
    ShaderVar::ShaderVar(const ShaderVar& other) : mpBlock(other.mpBlock), mOffset(other.mOffset) {}
    ShaderVar::ShaderVar(ParameterBlock* pObject, const TypedShaderVarOffset& offset) : mpBlock(pObject), mOffset(offset) {}
    ShaderVar::ShaderVar(ParameterBlock* pObject) : mpBlock(pObject), mOffset(pObject->getElementType().get(), ShaderVarOffset::kZero) {}

    ShaderVar ShaderVar::findMember(const std::string& name) const
    {
        if (!isValid()) return *this;
        auto pType = getType();

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
                TypedShaderVarOffset newOffset = TypedShaderVarOffset(pMember->getType().get(), mOffset + pMember->getBindLocation());
                return ShaderVar(mpBlock, newOffset);
            }
        }

        return ShaderVar();
    }

    ShaderVar ShaderVar::findMember(uint32_t index) const
    {
        if (!isValid()) return *this;
        auto pType = getType();

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
            if( index < pStructType->getMemberCount() )
            {
                auto pMember = pStructType->getMember(index);

                // Need to apply the offsets from member
                TypedShaderVarOffset newOffset = TypedShaderVarOffset(pMember->getType().get(), mOffset + pMember->getBindLocation());
                return ShaderVar( mpBlock, newOffset);
            }
        }

        return ShaderVar();
    }

    ShaderVar ShaderVar::operator[](const std::string& name) const
    {
        auto result = findMember(name);
        if( !result.isValid() && isValid() )
        {
            reportError("No member named '" + name + "' found.\n");
        }
        return result;
    }

    ShaderVar ShaderVar::operator[](const char* name) const
    {
        // #SHADER_VAR we can use std::string_view to do lookups into the map
        // TODO: We should have the ability to do this lookup
        // without ever having to construct a `std::string`.
        //
        // The sticking point is the `ReflectionStructType::findMember`
        // operation, which currently needs to do lookup in a `std::map<std::string, ...>`,
        // so that we can't use a `const char*` as the key for lookup
        // without incurring the cost of a constructing a `std::string`.
        //
        // To get around this limitation we'd need to implement a more clever/complicated
        // map that uses a raw `const char*` as the key, and then compares keys with
        // `strcmp`, while ensuring that the keys actually stored in the map are allocated
        // and retained somewhere (which they should be because they are the names of
        // the members of the `ReflectionStructType`).
        //
        // For now we just punt and go with the slow option even
        // if the user is using a static string.
        //
        return (*this)[std::string(name)];
    }

    ShaderVar ShaderVar::operator[](size_t index) const
    {
        if (!isValid()) return *this;
        auto pType = getType();

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
                ResourceShaderVarOffset elementResourceLocation(mOffset.getResource().getRangeIndex(), mOffset.getResource().getArrayIndex() * elementCount + ResourceShaderVarOffset::ArrayIndex(index));
                TypedShaderVarOffset newOffset = TypedShaderVarOffset(pArrayType->getElementType().get(), ShaderVarOffset(elementUniformLocation, elementResourceLocation));
                return ShaderVar(mpBlock, newOffset);
            }
        }
        else if (auto pStructType = pType->asStructType())
        {
            if (index < pStructType->getMemberCount())
            {
                auto pMember = pStructType->getMember(index);
                // Need to apply the offsets from member
                TypedShaderVarOffset newOffset = TypedShaderVarOffset(pMember->getType().get(), mOffset + pMember->getBindLocation());
                return ShaderVar(mpBlock, newOffset);
            }
        }

        reportError("No element or member found at index " + std::to_string(index));
        return ShaderVar();
    }

    ShaderVar ShaderVar::operator[](TypedShaderVarOffset const& offset) const
    {
        if (!isValid()) return *this;
        auto pType = getType();

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

        return ShaderVar(mpBlock, TypedShaderVarOffset(offset.getType().get(), mOffset + offset));
    }

    ShaderVar ShaderVar::operator[](UniformShaderVarOffset const& loc) const
    {
        if (!isValid()) return *this;
        auto pType = getType();

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
        if (byteOffset == 0) return *this;

        if (auto pArrayType = pType->asArrayType())
        {
            auto pElementType = pArrayType->getElementType();
            auto elementCount = pArrayType->getElementCount();
            auto elementStride = pArrayType->getElementByteStride();

            auto elementIndex = byteOffset / elementStride;
            auto offsetIntoElement = byteOffset % elementStride;

            TypedShaderVarOffset elementOffset = TypedShaderVarOffset(pElementType.get(), ShaderVarOffset(mOffset.getUniform() + elementIndex * elementStride, mOffset.getResource()));
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

                if (byteOffset < memberByteOffset)                   continue;
                if (byteOffset >= memberByteOffset + memberByteSize) continue;

                auto offsetIntoMember = byteOffset - memberByteOffset;
                TypedShaderVarOffset memberOffset = TypedShaderVarOffset(pMember->getType().get(), mOffset + pMember->getBindLocation());
                ShaderVar memberCursor(mpBlock, memberOffset);
                return memberCursor[UniformShaderVarOffset(offsetIntoMember)];
            }
        }

        reportError("no member at offset");
        return ShaderVar();
    }

    bool ShaderVar::isValid() const
    {
        return mOffset.isValid();
    }

    bool ShaderVar::setTexture(const Texture::SharedPtr& pTexture) const
    {
        return mpBlock->setTexture(mOffset, pTexture);
    }

    bool ShaderVar::setSampler(const Sampler::SharedPtr& pSampler) const
    {
        return mpBlock->setSampler(mOffset, pSampler);
    }

    bool ShaderVar::setBuffer(const Buffer::SharedPtr& pBuffer) const
    {
        return mpBlock->setBuffer(mOffset, pBuffer);
    }

    bool ShaderVar::setSrv(const ShaderResourceView::SharedPtr& pSrv) const
    {
        return mpBlock->setSrv(mOffset, pSrv);
    }

    bool ShaderVar::setUav(const UnorderedAccessView::SharedPtr& pUav) const
    {
        return mpBlock->setUav(mOffset, pUav);
    }

    bool ShaderVar::setAccelerationStructure(const RtAccelerationStructure::SharedPtr& pAccl) const
    {
        return mpBlock->setAccelerationStructure(mOffset, pAccl);
    }

    bool ShaderVar::setParameterBlock(const std::shared_ptr<ParameterBlock>& pBlock) const
    {
        return mpBlock->setParameterBlock(mOffset, pBlock);
    }

    bool ShaderVar::setImpl(const Texture::SharedPtr& pTexture) const
    {
        return mpBlock->setTexture(mOffset, pTexture);
    }

    bool ShaderVar::setImpl(const Sampler::SharedPtr& pSampler) const
    {
        return mpBlock->setSampler(mOffset, pSampler);
    }

    bool ShaderVar::setImpl(const Buffer::SharedPtr& pBuffer) const
    {
        return mpBlock->setBuffer(mOffset, pBuffer);
    }

    bool ShaderVar::setImpl(const std::shared_ptr<ParameterBlock>& pBlock) const
    {
        return mpBlock->setParameterBlock(mOffset, pBlock);
    }

    bool ShaderVar::setBlob(void const* data, size_t size) const
    {
        // If the var is pointing at a constant buffer, then assume
        // the user actually means to write the blob *into* that buffer.
        //
        auto pType = getType();
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

        return mpBlock->setBlob(mOffset, data, size);
    }

    ShaderVar::operator Buffer::SharedPtr() const
    {
        return mpBlock->getBuffer(mOffset);
    }

    ShaderVar::operator Texture::SharedPtr() const
    {
        return mpBlock->getTexture(mOffset);
    }

    ShaderVar::operator Sampler::SharedPtr() const
    {
        return mpBlock->getSampler(mOffset);
    }

    ShaderVar::operator UniformShaderVarOffset() const
    {
        return mOffset.getUniform();
    }

    std::shared_ptr<ParameterBlock> ShaderVar::getParameterBlock() const
    {
        return mpBlock->getParameterBlock(mOffset);
    }

    Buffer::SharedPtr ShaderVar::getBuffer() const
    {
        return mpBlock->getBuffer(mOffset);
    }

    Texture::SharedPtr ShaderVar::getTexture() const
    {
        return mpBlock->getTexture(mOffset);
    }

    Sampler::SharedPtr ShaderVar::getSampler() const
    {
        return mpBlock->getSampler(mOffset);
    }

    ShaderResourceView::SharedPtr ShaderVar::getSrv() const
    {
        return mpBlock->getSrv(mOffset);
    }

    UnorderedAccessView::SharedPtr ShaderVar::getUav() const
    {
        return mpBlock->getUav(mOffset);
    }

    RtAccelerationStructure::SharedPtr ShaderVar::getAccelerationStructure() const
    {
        return mpBlock->getAccelerationStructure(mOffset);
    }

    size_t ShaderVar::getByteOffset() const
    {
        return mOffset.getUniform().getByteOffset();
    }

    ReflectionType::SharedConstPtr ShaderVar::getType() const
    {
        return mOffset.getType();
    }

    void const* ShaderVar::getRawData() const
    {
        return (uint8_t*)(mpBlock->getRawData()) + mOffset.getUniform().getByteOffset();
    }

}
