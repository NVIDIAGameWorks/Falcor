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
#include "Graphics/Program/ProgramReflection.h"
#include "Texture.h"
#include "Buffer.h"
#include "Graphics/Program//Program.h"

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

        static const size_t kInvalidOffset = -1;// ProgramReflection::kInvalidLocation;

        VariablesBuffer(const std::string& name, const ReflectionResourceType::SharedConstPtr& pReflectionType, size_t elementSize, size_t elementCount, BindFlags bindFlags, CpuAccess cpuAccess);

        virtual ~VariablesBuffer() = 0;

        /** Apply the changes to the actual GPU buffer.
            Note that it is possible to use this function to update only part of the GPU copy of the buffer. This might lead to inconsistencies between the GPU and CPU buffer, so make sure you know what you are doing.
            \param[in] offset Offset into the buffer to write to
            \param[in] size Number of bytes to upload. If this value is -1, will update the [Offset, EndOfBuffer] range.
        */
        virtual bool uploadToGPU(size_t offset = 0, size_t size = -1);

        /** Get the reflection object describing the CB
        */
        ReflectionType::SharedConstPtr getBufferReflector() const { return mpReflector; }

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

        ReflectionResourceType::SharedConstPtr mpReflector;
        std::vector<uint8_t> mData;
        mutable bool mDirty = true;
        size_t mElementCount;
        size_t mElementSize;
        std::string mName;
    };
}

