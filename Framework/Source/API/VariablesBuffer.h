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
#pragma once
#include <string>
#include "ProgramReflection.h"
#include "Texture.h"
#include "Buffer.h"
#include "Graphics/Program.h"

namespace Falcor
{
    class Sampler;

    /** Manages shader buffers containing named data, such as Constant/Uniform Buffers and Structured Buffers.
        When accessing a variable by name, you can only use a name which points to a basic Type, or an array of basic Type (so if you want the start of a structure, ask for the first field in the struct).
        Note that Falcor has 2 flavors of setting variable by names - SetVariable() and SetVariableArray(). Naming rules for N-dimensional arrays of a basic Type are a little different between the two.
        SetVariable() must include N indices. SetVariableArray() can include N indices, or N-1 indices (implicit [0] as last index).
    */
    class VariablesBuffer : public Buffer, public inherit_shared_from_this<Buffer, VariablesBuffer>
    {
    public:
        using SharedPtr = std::shared_ptr<VariablesBuffer>;
        using SharedConstPtr = std::shared_ptr<const VariablesBuffer>;

        static const size_t kInvalidOffset = ProgramReflection::kInvalidLocation;

        VariablesBuffer(const ProgramReflection::BufferReflection::SharedConstPtr& pReflector, size_t elementSize, size_t elementCount, BindFlags bindFlags, CpuAccess cpuAccess);

        virtual ~VariablesBuffer() = 0;

        /** Apply the changes to the actual GPU buffer.
            Note that it is possible to use this function to update only part of the GPU copy of the buffer. This might lead to inconsistencies between the GPU and CPU buffer, so make sure you know what you are doing.
            \param[in] offset Offset into the buffer to write to
            \param[in] size Number of bytes to upload. If this value is -1, will update the [Offset, EndOfBuffer] range.
        */
        virtual bool uploadToGPU(size_t offset = 0, size_t size = -1);

        /** Get the reflection object describing the CB
        */
        ProgramReflection::BufferReflection::SharedConstPtr getBufferReflector() const { return mpReflector; }

        /** Set a block of data into the constant buffer.\n
            If Offset + Size will result in buffer overflow, the call will be ignored and log an error.
            \param[in] pSrc Pointer to the source data.
            \param[in] offset Destination offset inside the buffer.
            \param[in] size Number of bytes in the source data.
        */
        void setBlob(const void* pSrc, size_t offset, size_t size);

        /** Get a variable offset inside the buffer. See notes about naming in the VariablesBuffer class description. Constant name can be provided with an implicit array-index, similar to VariablesBuffer#SetVariableArray.
        */
        size_t getVariableOffset(const std::string& varName) const;

        size_t getElementCount() const { return mElementCount; }

        size_t getElementSize() const { return mElementSize; }

    protected:
        template<typename T>
        void setVariable(const std::string& name, size_t elementIndex, const T& value);

        template<typename T>
        void setVariableArray(size_t offset, size_t elementIndex, const T* pValue, size_t count);

        template<typename T>
        void setVariable(size_t offset, size_t elementIndex, const T& value);

        template<typename T>
        void setVariableArray(const std::string& name, size_t elementIndex, const T* pValue, size_t count);

        void setTexture(const std::string& name, const Texture* pTexture, const Sampler* pSampler);

        void setTextureArray(const std::string& name, const Texture* pTexture[], const Sampler* pSampler, size_t count);

        void setTexture(size_t Offset, const Texture* pTexture, const Sampler* pSampler);

        void setTextureInternal(size_t offset, const Texture* pTexture, const Sampler* pSampler);

        ProgramReflection::BufferReflection::SharedConstPtr mpReflector;
        std::vector<uint8_t> mData;
        mutable bool mDirty = true;
        size_t mElementCount;
        size_t mElementSize;
    };

    template<typename VarType>
    ProgramReflection::Variable::Type getReflectionTypeFromCType()
    {
#define c_to_prog(cType, progType) if(typeid(VarType) == typeid(cType)) return ProgramReflection::Variable::Type::progType;
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

        c_to_prog(uint64_t, GpuPtr);
#undef c_to_prog
        should_not_get_here();
        return ProgramReflection::Variable::Type::Unknown;
    }


    template<typename VarType>
    bool checkVariableType(ProgramReflection::Variable::Type shaderType, const std::string& name, const std::string& bufferName)
    {
#if _LOG_ENABLED
        ProgramReflection::Variable::Type callType = getReflectionTypeFromCType<VarType>();
        // Check that the types match
        if(callType != shaderType && shaderType != ProgramReflection::Variable::Type::Unknown)
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

    template<typename VarType>
    bool checkVariableByOffset(size_t offset, size_t count, const ProgramReflection::BufferReflection* pBufferDesc)
    {
#if _LOG_ENABLED
        // Find the variable
        for(auto a = pBufferDesc->varBegin() ; a != pBufferDesc->varEnd() ; a++)
        {
            const auto& varDesc = a->second;
            const auto& varName = a->first;
            size_t arrayIndex = 0;
            bool checkThis = (varDesc.location == offset);

            // If this is an array, check if we set an element inside it
            if(varDesc.arrayStride > 0 && offset > varDesc.location)
            {
                size_t stride = offset - varDesc.location;
                if((stride % varDesc.arrayStride) == 0)
                {
                    arrayIndex = stride / varDesc.arrayStride;
                    if(arrayIndex < varDesc.arraySize)
                    {
                        checkThis = true;
                    }
                }
            }

            if(checkThis)
            {
                if(varDesc.arraySize == 0)
                {
                    if(count > 1 && varName.find('[') == std::string::npos)
                    {
                        std::string Msg("Error when setting constant by offset. Found constant \"" + varName + "\" which is not an array, but trying to set more than 1 element");
                        logError(Msg);
                        return false;
                    }
                }
                else if(arrayIndex + count > varDesc.arraySize)
                {
                    std::string Msg("Error when setting constant by offset. Found constant \"" + varName + "\" with array size " + std::to_string(varDesc.arraySize));
                    Msg += ". Trying to set " + std::to_string(count) + " elements, starting at index " + std::to_string(arrayIndex) + ", which will cause out-of-bound access. Ignoring call.";
                    logError(Msg);
                    return false;
                }
                return checkVariableType<VarType>(varDesc.type, varName + "(Set by offset)", pBufferDesc->getName());
            }
        }
        std::string msg("Error when setting constant by offset. No constant found at offset ");
        msg += std::to_string(offset) + ". Ignoring call";
        logError(msg);
        return false;
#else
        return true;
#endif
    }

}

