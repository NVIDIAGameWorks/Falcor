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
#include <unordered_map>
#include "VariablesBuffer.h"

namespace Falcor
{
    class Buffer;
    class ProgramVersion;
    class Texture;
    class Sampler;

    /** Manages an array-of-structs style shader buffer, known as Structured Buffers in DirectX.
        Even though the buffer is created with a specific reflection object, it can be used with other programs as long as the buffer declarations are the same across programs.
    */
    class StructuredBuffer : public VariablesBuffer, public inherit_shared_from_this<VariablesBuffer, StructuredBuffer>
    {
    public:
        class SharedPtr : public std::shared_ptr<StructuredBuffer>
        {
        public:
            class Element
            {
            public:
                class Var
                {
                public:
                    Var(StructuredBuffer* pBuf, size_t offset, size_t element) : mpBuf(pBuf), mElement(element), mOffset(offset) {}
                    template<typename T> void operator=(const T& val) { mpBuf->setVariable(mOffset, mElement, val); }
                    template<typename T> operator T() const { T val;  mpBuf->getVariable(mOffset, mElement, val); return val; }
                protected:
                    size_t mElement;
                    size_t mOffset;
                    StructuredBuffer* mpBuf;
                };

                Element(StructuredBuffer* pBuf, size_t element) : mpBuf(pBuf), mElement(element) {}
                Var operator[](size_t offset) { return Var(mpBuf, offset, mElement); }
                Var operator[](const std::string& var) { return Var(mpBuf, mpBuf->getVariableOffset(var), mElement); }
                size_t getElement() const { return mElement; }
            private:
                StructuredBuffer* mpBuf;
                size_t mElement;
            };

            SharedPtr() = default;
            SharedPtr(std::shared_ptr<StructuredBuffer> pBuf) : std::shared_ptr<StructuredBuffer>(pBuf) {}
            SharedPtr(StructuredBuffer* pBuf) : std::shared_ptr<StructuredBuffer>(pBuf) {}

            Element operator[](size_t elemIndex) { return Element(get(), elemIndex); }
        };

        using SharedConstPtr = std::shared_ptr<const StructuredBuffer>;

        /** Create a structured buffer.
            \param[in] pReflectionType A reflection type object containing the buffer layout
            \param[in] elementCount - the number of struct elements in the buffer
            \param[in] bindFlags The bind flags for the resource
            \return A new buffer object if the operation was successful, otherwise nullptr
        */
        static SharedPtr create(const std::string& name, const ReflectionResourceType::SharedConstPtr& pReflectionType, size_t elementCount = 1, Resource::BindFlags bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);
        
        /** Create a structured buffer. Fetches the requested buffer reflector from the active program version and create the buffer from it
            \param[in] pProgram A program object which defines the buffer
            \param[in] elementCount The number of struct elements in the buffer
            \param[in] bindFlags The bind flags for the resource
            \return A new buffer object if the operation was successful, otherwise nullptr
        */
        static SharedPtr create(const Program::SharedPtr& pProgram, const std::string& name, size_t elementCount = 1, Resource::BindFlags bindFlags = Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);

        ~StructuredBuffer();

        /** Read a variable from the buffer.
            The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged and the call will be ignored.
            \param[in] name The variable name. See notes about naming in the ConstantBuffer class description.
            \param[out] value The value read from the buffer
        */
        template<typename T>
        void getVariable(const std::string& name, size_t elementIndex, T& value);

        /** Read an array of variables from the buffer.
            The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged and the call will be ignored.
            \param[in] offset The byte offset of the variable inside the buffer
            \param[in] count Number of elements to read
            \param[out] value Pointer to an array of values to read into
        */
        template<typename T>
        void getVariableArray(size_t offset, size_t count, size_t elementIndex, T value[]);

        /** Read a variable from the buffer.
            The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged and the call will be ignored.
            \param[in] offset The byte offset of the variable inside the buffer
            \param[out] value The value read from the buffer
        */
        template<typename T>
        void getVariable(size_t offset, size_t elementIndex, T& value);

        /** Read an array of variables from the buffer.
            The function will validate that the value Type matches the declaration in the shader. If there's a mismatch, an error will be logged and the call will be ignored.
            \param[in] name The variable name. See notes about naming in the ConstantBuffer class description.
            \param[in] count Number of elements to read
            \param[out] value Pointer to an array of values to read into
        */
        template<typename T>
        void getVariableArray(const std::string& name, size_t count, size_t elementIndex, T value[]);

        /** Read a block of data from the buffer.
            If Offset + Size will result in buffer overflow, the call will be ignored and log an error.
            \param pDst Pointer to a buffer to write the data into
            \param offset Byte offset to start reading from the buffer
            \param size Number of bytes to read from the buffer
        */
        void readBlob(void* pDst, size_t offset, size_t size);

        /** Read the buffer data from the GPU.\n
            Note that it is possible to use this function to update only part of the CPU copy of the buffer. This might lead to inconsistencies between the GPU and CPU buffer, so make sure you know what you are doing.
            \param[in] offset Offset into the buffer to read from
            \param[in] size   Number of bytes to read. If this value is -1, will update the [Offset, EndOfBuffer] range.
        */
        void readFromGPU(size_t offset = 0, size_t size = -1);

        /** Set the GPUCopyDirty flag
        */
        void setGpuCopyDirty() const { mGpuCopyDirty = true; }

        /** If the buffer can be used as a UAV, checks whether it has an associated counter.
        */
        bool hasUAVCounter() const;

        /** Get the UAV counter buffer.
        */
        const Buffer::SharedPtr& getUAVCounter() const { return mpUAVCounter; }

    private:
        StructuredBuffer(const std::string& name, const ReflectionResourceType::SharedConstPtr& pReflectionType, size_t elementCount, Resource::BindFlags bindFlags);
        mutable bool mGpuCopyDirty = false;

        Buffer::SharedPtr mpUAVCounter;
    };
}