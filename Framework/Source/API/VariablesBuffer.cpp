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
#include "VariablesBuffer.h"
#include "Graphics/Program/ProgramVersion.h"
#include "Buffer.h"
#include "glm/glm.hpp"
#include "Texture.h"
#include "Graphics/Program/ProgramReflection.h"
#include "API/Device.h"

namespace Falcor
{
    VariablesBuffer::~VariablesBuffer() = default;

    template<typename VarType>
    ReflectionType::Type getReflectionTypeFromCType()
    {
#define c_to_prog(cType, progType) if(typeid(VarType) == typeid(cType)) return ReflectionType::Type::progType;
        c_to_prog(bool,  Bool);
        c_to_prog(bvec2, Bool2);
        c_to_prog(bvec3, Bool3);
        c_to_prog(bvec4, Bool4);

        c_to_prog(int32_t, Int);
        c_to_prog(ivec2, Int2);
        c_to_prog(ivec3, Int3);
        c_to_prog(ivec4, Int4);

        c_to_prog(uint32_t, Uint);
        c_to_prog(uvec2, Uint2);
        c_to_prog(uvec3, Uint3);
        c_to_prog(uvec4, Uint4);

        c_to_prog(float,     Float);
        c_to_prog(glm::vec2, Float2);
        c_to_prog(glm::vec3, Float3);
        c_to_prog(glm::vec4, Float4);

        c_to_prog(glm::mat2,   Float2x2);
        c_to_prog(glm::mat2x3, Float2x3);
        c_to_prog(glm::mat2x4, Float2x4);

        c_to_prog(glm::mat3  , Float3x3);
        c_to_prog(glm::mat3x2, Float3x2);
        c_to_prog(glm::mat3x4, Float3x4);
        
        c_to_prog(glm::mat4, Float4x4);
        c_to_prog(glm::mat4x2, Float4x2);
        c_to_prog(glm::mat4x3, Float4x3);

#undef c_to_prog
        should_not_get_here();
        return ReflectionType::Type::Unknown;
    }

    VariablesBuffer::VariablesBuffer(const std::string& name, const ReflectionType::SharedConstPtr& pReflectionType, size_t elementSize, size_t elementCount, BindFlags bindFlags, CpuAccess cpuAccess) :
       mName(name), mpReflector(pReflectionType), Buffer(elementSize * elementCount, bindFlags, cpuAccess), mElementCount(elementCount), mElementSize(elementSize)
    {
        Buffer::apiInit(false);
        mData.assign(mSize, 0);
    }

    size_t VariablesBuffer::getVariableOffset(const std::string& varName) const
    {
        size_t offset = 0;
        const auto& pVar = mpReflector->findMember(varName);
        return pVar->getOffset();
    }

    bool VariablesBuffer::uploadToGPU(size_t offset, size_t size)
    {
        if(mDirty == false)
        {
            return false;
        }

        if(size == -1)
        {
            size = mSize - offset;
        }

        if(size + offset > mSize)
        {
            logWarning("VariablesBuffer::uploadToGPU() - trying to upload more data than what the buffer contains. Call is ignored.");
            return false;
        }

        updateData(mData.data(), offset, size);
        mDirty = false;
        return true;
    }

    template<typename VarType>
    bool checkVariableType(ReflectionType::Type shaderType, const std::string& name, const std::string& bufferName)
    {
#if _LOG_ENABLED
        auto callType = getReflectionTypeFromCType<VarType>();
        // Check that the types match
        if(callType != shaderType)
        {
            std::string msg("Error when setting variable \"");
            msg += name + "\" to buffer \"" + bufferName + "\".\n";
            msg += "Type mismatch.\nsetVariable() was called with Type " + to_string(callType) + ".\nVariable was declared with Type " + to_string(shaderType) + ".\n\n";
            logError(msg);
            assert(0);
            return false;
        }
#endif
        return true;
    }

//     template<typename VarType>
//     bool checkVariableByOffset(size_t offset, size_t count, const ProgramReflection::BufferReflection* pBufferDesc)
//     {
// #if _LOG_ENABLED
//         ProgramReflection::Variable::Type callType = getReflectionTypeFromCType<VarType>();
//         // Find the variable
//         for(auto& a = pBufferDesc->varBegin() ; a != pBufferDesc->varEnd() ; a++)
//         {
//             const auto& varDesc = a->second;
//             const auto& varName = a->first;
//             size_t arrayIndex = 0;
//             bool checkThis = (varDesc.location == offset);
// 
//             // If this is an array, check if we set an element inside it
//             if(varDesc.arrayStride > 0 && offset > varDesc.location)
//             {
//                 size_t stride = offset - varDesc.location;
//                 if((stride % varDesc.arrayStride) == 0)
//                 {
//                     arrayIndex = stride / varDesc.arrayStride;
//                     if(arrayIndex < varDesc.arraySize)
//                     {
//                         checkThis = true;
//                     }
//                 }
//             }
// 
//             if(checkThis)
//             {
//                 if(varDesc.arraySize == 0)
//                 {
//                     if(count > 1)
//                     {
//                         std::string Msg("Error when setting constant by offset. Found constant \"" + varName + "\" which is not an array, but trying to set more than 1 element");
//                         logError(Msg);
//                         return false;
//                     }
//                 }
//                 else if(arrayIndex + count > varDesc.arraySize)
//                 {
//                     std::string Msg("Error when setting constant by offset. Found constant \"" + varName + "\" with array size " + std::to_string(varDesc.arraySize));
//                     Msg += ". Trying to set " + std::to_string(count) + " elements, starting at index " + std::to_string(arrayIndex) + ", which will cause out-of-bound access. Ignoring call.";
//                     logError(Msg);
//                     return false;
//                 }
//                 return checkVariableType<VarType>(varDesc.type, varName + "(Set by offset)", pBufferDesc->getName());
//             }
//         }
//         std::string msg("Error when setting constant by offset. No constant found at offset ");
//         msg += std::to_string(offset) + ". Ignoring call";
//         logError(msg);
//         return false;
// #else
//         return true;
// #endif
//     }

#define verify_element_index() if(elementIndex >= mElementCount) {logWarning(std::string(__FUNCTION__) + ": elementIndex is out-of-bound. Ignoring call."); return;}

    template<typename VarType> 
    void VariablesBuffer::setVariable(size_t offset, size_t elementIndex, const VarType& value)
    {
        verify_element_index();
//        if(checkVariableByOffset<VarType>(offset, 1, mpReflector.get()))
        {
            const uint8_t* pVar = mData.data() + offset + elementIndex * mElementSize;
            *(VarType*)pVar = value;
            mDirty = true;
        }
    }

#define set_constant_by_offset(_t) template void VariablesBuffer::setVariable(size_t offset, size_t elementIndex, const _t& value)
    set_constant_by_offset(bool);
    set_constant_by_offset(glm::bvec2);
    set_constant_by_offset(glm::bvec3);
    set_constant_by_offset(glm::bvec4);

    set_constant_by_offset(uint32_t);
    set_constant_by_offset(glm::uvec2);
    set_constant_by_offset(glm::uvec3);
    set_constant_by_offset(glm::uvec4);

    set_constant_by_offset(int32_t);
    set_constant_by_offset(glm::ivec2);
    set_constant_by_offset(glm::ivec3);
    set_constant_by_offset(glm::ivec4);

    set_constant_by_offset(float);
    set_constant_by_offset(glm::vec2);
    set_constant_by_offset(glm::vec3);
    set_constant_by_offset(glm::vec4);

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

    template<typename VarType>
    void VariablesBuffer::setVariable(const std::string& name, size_t element, const VarType& value)
    {
        const auto& pVar = mpReflector->findMember(name);
        bool valid = true;
        if((_LOG_ENABLED == 0) || (pVar && checkVariableType<VarType>(pVar->getType()->getType(), name, mName)))
        {
            setVariable<VarType>(pVar->getOffset(), element, value);
        }
    }

#define set_constant_by_name(_t) template void VariablesBuffer::setVariable(const std::string& name, size_t element, const _t& value)

    set_constant_by_name(bool);
    set_constant_by_name(glm::bvec2);
    set_constant_by_name(glm::bvec3);
    set_constant_by_name(glm::bvec4);

    set_constant_by_name(uint32_t);
    set_constant_by_name(glm::uvec2);
    set_constant_by_name(glm::uvec3);
    set_constant_by_name(glm::uvec4);

    set_constant_by_name(int32_t);
    set_constant_by_name(glm::ivec2);
    set_constant_by_name(glm::ivec3);
    set_constant_by_name(glm::ivec4);

    set_constant_by_name(float);
    set_constant_by_name(glm::vec2);
    set_constant_by_name(glm::vec3);
    set_constant_by_name(glm::vec4);

    set_constant_by_name(glm::mat2);
    set_constant_by_name(glm::mat2x3);
    set_constant_by_name(glm::mat2x4);

    set_constant_by_name(glm::mat3);
    set_constant_by_name(glm::mat3x2);
    set_constant_by_name(glm::mat3x4);

    set_constant_by_name(glm::mat4);
    set_constant_by_name(glm::mat4x2);
    set_constant_by_name(glm::mat4x3);

    set_constant_by_name(uint64_t);
#undef set_constant_by_name

    template<typename VarType> 
    void VariablesBuffer::setVariableArray(size_t offset, size_t elementIndex, const VarType* pValue, size_t count)
    {
        verify_element_index();
//        if(checkVariableByOffset<VarType>(offset, count, mpReflector.get()))
        {
            const uint8_t* pVar = mData.data() + offset;
            VarType* pData = (VarType*)pVar + elementIndex * mElementSize;
            for(size_t i = 0; i < count; i++)
            {
                pData[i] = pValue[i];
            }
            mDirty = true;
        }
    }

#define set_constant_array_by_offset(_t) template void VariablesBuffer::setVariableArray(size_t offset, size_t elementIndex, const _t* pValue, size_t count)

    set_constant_array_by_offset(bool);
    set_constant_array_by_offset(glm::bvec2);
    set_constant_array_by_offset(glm::bvec3);
    set_constant_array_by_offset(glm::bvec4);

    set_constant_array_by_offset(uint32_t);
    set_constant_array_by_offset(glm::uvec2);
    set_constant_array_by_offset(glm::uvec3);
    set_constant_array_by_offset(glm::uvec4);

    set_constant_array_by_offset(int32_t);
    set_constant_array_by_offset(glm::ivec2);
    set_constant_array_by_offset(glm::ivec3);
    set_constant_array_by_offset(glm::ivec4);

    set_constant_array_by_offset(float);
    set_constant_array_by_offset(glm::vec2);
    set_constant_array_by_offset(glm::vec3);
    set_constant_array_by_offset(glm::vec4);

    set_constant_array_by_offset(glm::mat2);
    set_constant_array_by_offset(glm::mat2x3);
    set_constant_array_by_offset(glm::mat2x4);

    set_constant_array_by_offset(glm::mat3);
    set_constant_array_by_offset(glm::mat3x2);
    set_constant_array_by_offset(glm::mat3x4);

    set_constant_array_by_offset(glm::mat4);
    set_constant_array_by_offset(glm::mat4x2);
    set_constant_array_by_offset(glm::mat4x3);

    set_constant_array_by_offset(uint64_t);

#undef set_constant_array_by_offset

    template<typename VarType>
    void VariablesBuffer::setVariableArray(const std::string& name, size_t elementIndex, const VarType* pValue, size_t count)
    {
        const auto& pVar = mpReflector->findMember(name);
        if( _LOG_ENABLED == 0 || (pVar && checkVariableType<VarType>(pVar->getType()->getType(), name, mName)))
        {
#if _LOG_ENABLED
            if (pVar->getType()->getArraySize() == 0)
            {
                logWarning("Can't use VariablesBuffer::setVariableArray() on " + name + ". It is not an array.");
                return;
            }
            if (count - elementIndex > pVar->getType()->getArraySize())
            {
                logWarning("VariablesBuffer::setVariableArray() - setting to many elements. Clamping...");
                count = pVar->getType()->getArraySize() - elementIndex;
            }
#endif
            setVariableArray(pVar->getOffset(), elementIndex, pValue, count);
        }
    }
    
#define set_constant_array_by_string(_t) template void VariablesBuffer::setVariableArray(const std::string& name, size_t elementIndex, const _t* pValue, size_t count)

    set_constant_array_by_string(bool);
    set_constant_array_by_string(glm::bvec2);
    set_constant_array_by_string(glm::bvec3);
    set_constant_array_by_string(glm::bvec4);

    set_constant_array_by_string(uint32_t);
    set_constant_array_by_string(glm::uvec2);
    set_constant_array_by_string(glm::uvec3);
    set_constant_array_by_string(glm::uvec4);

    set_constant_array_by_string(int32_t);
    set_constant_array_by_string(glm::ivec2);
    set_constant_array_by_string(glm::ivec3);
    set_constant_array_by_string(glm::ivec4);

    set_constant_array_by_string(float);
    set_constant_array_by_string(glm::vec2);
    set_constant_array_by_string(glm::vec3);
    set_constant_array_by_string(glm::vec4);

    set_constant_array_by_string(glm::mat2);
    set_constant_array_by_string(glm::mat2x3);
    set_constant_array_by_string(glm::mat2x4);

    set_constant_array_by_string(glm::mat3);
    set_constant_array_by_string(glm::mat3x2);
    set_constant_array_by_string(glm::mat3x4);

    set_constant_array_by_string(glm::mat4);
    set_constant_array_by_string(glm::mat4x2);
    set_constant_array_by_string(glm::mat4x3);

    set_constant_array_by_string(uint64_t);

#undef set_constant_array_by_string

    void VariablesBuffer::setBlob(const void* pSrc, size_t offset, size_t size)
    {
        if((_LOG_ENABLED != 0) && (offset + size > mSize))
        {
            std::string Msg("Error when setting blob to buffer\"");
            Msg += mName + "\". Blob to large and will result in overflow. Ignoring call.";
            logError(Msg);
            return;
        }
        memcpy(mData.data() + offset, pSrc, size);
        mDirty = true;
    }
}
