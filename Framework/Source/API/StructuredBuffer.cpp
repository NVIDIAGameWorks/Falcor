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
#include "API/StructuredBuffer.h"
#include "API/Buffer.h"
#include <cstring>

namespace Falcor
{
    template<typename VarType> 
    bool checkVariableByOffset(size_t offset, size_t count, const ReflectionResourceType* pReflection);
    template<typename VarType> 
    bool checkVariableType(const ReflectionType* pShaderType, const std::string& name, const std::string& bufferName);

#define verify_element_index() if(elementIndex >= mElementCount) {logWarning(std::string(__FUNCTION__) + ": elementIndex is out-of-bound. Ignoring call."); return;}

    StructuredBuffer::StructuredBuffer(const std::string& name, const ReflectionResourceType::SharedConstPtr& pReflector, size_t elementCount, Resource::BindFlags bindFlags)
        : VariablesBuffer(name, pReflector, pReflector->getSize(), elementCount, bindFlags, Buffer::CpuAccess::None)
    {
        if (hasUAVCounter())
        {
            static const uint32_t zero = 0;
            mpUAVCounter = Buffer::create(sizeof(uint32_t), Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
        }
    }

    StructuredBuffer::SharedPtr StructuredBuffer::create(const std::string& name, const ReflectionResourceType::SharedConstPtr& pReflection, size_t elementCount, Resource::BindFlags bindFlags)
    {
        assert(elementCount > 0);
        auto pBuffer = SharedPtr(new StructuredBuffer(name, pReflection, elementCount, bindFlags));
        return pBuffer;
    }

    StructuredBuffer::SharedPtr StructuredBuffer::create(const Program::SharedPtr& pProgram, const std::string& name, size_t elementCount, Resource::BindFlags bindFlags)
    {
        const auto& pProgReflector = pProgram->getActiveVersion()->getReflector();
        const auto& pDefaultBlock = pProgReflector->getDefaultParameterBlock();
        const ReflectionVar* pVar = pDefaultBlock ? pDefaultBlock->getResource(name).get() : nullptr;

        if (pVar)
        {
            ReflectionResourceType::SharedConstPtr pType = pVar->getType()->unwrapArray()->asResourceType()->inherit_shared_from_this::shared_from_this();
            if(pType && pType->getType() == ReflectionResourceType::Type::StructuredBuffer)
            {
                return create(pVar->getName(), pType, elementCount, bindFlags);
            }
        }
        logError("Can't find a structured buffer named \"" + name + "\" in the program");
        return nullptr;
    }

    void StructuredBuffer::readFromGPU(size_t offset, size_t size)
    {
        if(size == -1)
        {
            size = mSize - offset;
        }
        if(size + offset > mSize)
        {
            logWarning("StructuredBuffer::readFromGPU() - trying to read more data than what the buffer contains. Call is ignored.");
            return;
        }

        if(mGpuCopyDirty)
        {
            mGpuCopyDirty = false;
            const uint8_t* pData = (uint8_t*)map(Buffer::MapType::Read);
            std::memcpy(mData.data(), pData, mData.size());
            unmap();
        }
    }

    bool StructuredBuffer::hasUAVCounter() const
    {
        assert(mpReflector->getStructuredBufferType() != ReflectionResourceType::StructuredType::Invalid);
        return (mpReflector->getStructuredBufferType() != ReflectionResourceType::StructuredType::Default);
    }

    StructuredBuffer::~StructuredBuffer() = default;

    void StructuredBuffer::readBlob(void* pDest, size_t offset, size_t size)   
    {    
        if(size + offset > mSize)
        {
            logWarning("StructuredBuffer::readBlob() - trying to read more data than what the buffer contains. Call is ignored.");
            return;
        }
        readFromGPU();
        std::memcpy(pDest, mData.data() + offset, size);
    }

    template<typename VarType> 
    void StructuredBuffer::getVariable(size_t offset, size_t elementIndex, VarType& value)
    {
        verify_element_index();
        if(_LOG_ENABLED == 0 || checkVariableByOffset<VarType>(offset, 0, mpReflector.get()))
        {
            readFromGPU();
            const uint8_t* pVar = mData.data() + offset + elementIndex * mElementSize;
            value = *(const VarType*)pVar;
        }
    }

#define get_constant_offset(_t) template void StructuredBuffer::getVariable(size_t offset, size_t elementIndex, _t& value)

    get_constant_offset(bool);
    get_constant_offset(glm::bvec2);
    get_constant_offset(glm::bvec3);
    get_constant_offset(glm::bvec4);

    get_constant_offset(uint32_t);
    get_constant_offset(glm::uvec2);
    get_constant_offset(glm::uvec3);
    get_constant_offset(glm::uvec4);

    get_constant_offset(int32_t);
    get_constant_offset(glm::ivec2);
    get_constant_offset(glm::ivec3);
    get_constant_offset(glm::ivec4);

    get_constant_offset(float);
    get_constant_offset(glm::vec2);
    get_constant_offset(glm::vec3);
    get_constant_offset(glm::vec4);

    get_constant_offset(glm::mat2);
    get_constant_offset(glm::mat2x3);
    get_constant_offset(glm::mat2x4);

    get_constant_offset(glm::mat3);
    get_constant_offset(glm::mat3x2);
    get_constant_offset(glm::mat3x4);

    get_constant_offset(glm::mat4);
    get_constant_offset(glm::mat4x2);
    get_constant_offset(glm::mat4x3);

    get_constant_offset(uint64_t);

#undef get_constant_offset

    template<typename VarType>
    void StructuredBuffer::getVariable(const std::string& name, size_t elementIndex, VarType& value)
    {
       const auto& pVar = mpReflector->findMember(name);
       
        if ((_LOG_ENABLED == 0) || (pVar && checkVariableType<VarType>(pVar->getType().get(), name, mName)))
        {
            getVariable(pVar->getOffset(), elementIndex, value);
        }
    }

#define get_constant_string(_t) template void StructuredBuffer::getVariable(const std::string& name, size_t elementIndex, _t& value)

    get_constant_string(bool);
    get_constant_string(glm::bvec2);
    get_constant_string(glm::bvec3);
    get_constant_string(glm::bvec4);

    get_constant_string(uint32_t);
    get_constant_string(glm::uvec2);
    get_constant_string(glm::uvec3);
    get_constant_string(glm::uvec4);

    get_constant_string(int32_t);
    get_constant_string(glm::ivec2);
    get_constant_string(glm::ivec3);
    get_constant_string(glm::ivec4);

    get_constant_string(float);
    get_constant_string(glm::vec2);
    get_constant_string(glm::vec3);
    get_constant_string(glm::vec4);

    get_constant_string(glm::mat2);
    get_constant_string(glm::mat2x3);
    get_constant_string(glm::mat2x4);

    get_constant_string(glm::mat3);
    get_constant_string(glm::mat3x2);
    get_constant_string(glm::mat3x4);

    get_constant_string(glm::mat4);
    get_constant_string(glm::mat4x2);
    get_constant_string(glm::mat4x3);

    get_constant_string(uint64_t);
#undef get_constant_string

    template<typename VarType>
    void StructuredBuffer::getVariableArray(size_t offset, size_t count, size_t elementIndex, VarType value[])
    {
        verify_element_index();

        if (_LOG_ENABLED == 0 || checkVariableByOffset<VarType>(offset, count, mpReflector.get()))
        {
            readFromGPU();
            const uint8_t* pVar = mData.data() + offset;
            const VarType* pMat = (VarType*)(pVar + elementIndex * mElementSize);
            for (size_t i = 0; i < count; i++)
            {
                value[i] = pMat[i];
            }
        }
    }

#define get_constant_array_offset(_t) template void StructuredBuffer::getVariableArray(size_t offset, size_t count, size_t elementIndex, _t value[])

    get_constant_array_offset(bool);
    get_constant_array_offset(glm::bvec2);
    get_constant_array_offset(glm::bvec3);
    get_constant_array_offset(glm::bvec4);

    get_constant_array_offset(uint32_t);
    get_constant_array_offset(glm::uvec2);
    get_constant_array_offset(glm::uvec3);
    get_constant_array_offset(glm::uvec4);

    get_constant_array_offset(int32_t);
    get_constant_array_offset(glm::ivec2);
    get_constant_array_offset(glm::ivec3);
    get_constant_array_offset(glm::ivec4);

    get_constant_array_offset(float);
    get_constant_array_offset(glm::vec2);
    get_constant_array_offset(glm::vec3);
    get_constant_array_offset(glm::vec4);

    get_constant_array_offset(glm::mat2);
    get_constant_array_offset(glm::mat2x3);
    get_constant_array_offset(glm::mat2x4);

    get_constant_array_offset(glm::mat3);
    get_constant_array_offset(glm::mat3x2);
    get_constant_array_offset(glm::mat3x4);

    get_constant_array_offset(glm::mat4);
    get_constant_array_offset(glm::mat4x2);
    get_constant_array_offset(glm::mat4x3);

    get_constant_array_offset(uint64_t);

#undef get_constant_array_offset

    template<typename VarType>
    void StructuredBuffer::getVariableArray(const std::string& name, size_t count, size_t elementIndex, VarType value[])
    {
        const auto& pVar = mpReflector->findMember(name);
        if ((_LOG_ENABLED == 0) || (pVar && checkVariableType<VarType>(pVar->getType().get(), name, mName)))
        {
            getVariableArray(pVar->getOffset(), count, elementIndex, value);
        }
    }

#define get_constant_array_string(_t) template void StructuredBuffer::getVariableArray(const std::string& name, size_t count, size_t elementIndex, _t value[])

    get_constant_array_string(bool);
    get_constant_array_string(glm::bvec2);
    get_constant_array_string(glm::bvec3);
    get_constant_array_string(glm::bvec4);

    get_constant_array_string(uint32_t);
    get_constant_array_string(glm::uvec2);
    get_constant_array_string(glm::uvec3);
    get_constant_array_string(glm::uvec4);

    get_constant_array_string(int32_t);
    get_constant_array_string(glm::ivec2);
    get_constant_array_string(glm::ivec3);
    get_constant_array_string(glm::ivec4);

    get_constant_array_string(float);
    get_constant_array_string(glm::vec2);
    get_constant_array_string(glm::vec3);
    get_constant_array_string(glm::vec4);

    get_constant_array_string(glm::mat2);
    get_constant_array_string(glm::mat2x3);
    get_constant_array_string(glm::mat2x4);

    get_constant_array_string(glm::mat3);
    get_constant_array_string(glm::mat3x2);
    get_constant_array_string(glm::mat3x4);

    get_constant_array_string(glm::mat4);
    get_constant_array_string(glm::mat4x2);
    get_constant_array_string(glm::mat4x3);

    get_constant_array_string(uint64_t);
#undef get_constant_array_string
}
