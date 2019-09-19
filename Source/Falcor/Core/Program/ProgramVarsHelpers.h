/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/API/Texture.h"
#include "Core/API/Sampler.h"
#include "Core/API/Buffer.h"
#include "Core/BufferTypes/StructuredBuffer.h"
#include "Core/BufferTypes/TypedBuffer.h"

namespace Falcor
{
    class ConstantBuffer;
    class ProgramVars;
    class ParameterBlock;

    class dlldecl IProgramVars
    {
    public:
        virtual ~IProgramVars() = default;
        virtual bool setConstantBuffer(const std::string& name, const ConstantBuffer::SharedPtr& pCB) = 0;
        virtual ConstantBuffer::SharedPtr getConstantBuffer(const std::string& name) const = 0;
        virtual bool setRawBuffer(const std::string& name, const Buffer::SharedPtr& pBuf) = 0;
        virtual bool setTypedBuffer(const std::string& name, const TypedBufferBase::SharedPtr& pBuf) = 0;
        virtual bool setStructuredBuffer(const std::string& name, const StructuredBuffer::SharedPtr& pBuf) = 0;
        virtual bool setSampler(const std::string& name, const Sampler::SharedPtr& pSampler) = 0;
        virtual bool setParameterBlock(const std::string& name, const std::shared_ptr<ParameterBlock>& pBlock) = 0;

        virtual Buffer::SharedPtr getRawBuffer(const std::string& name) const = 0;
        virtual TypedBufferBase::SharedPtr getTypedBuffer(const std::string& name) const = 0;
        virtual StructuredBuffer::SharedPtr getStructuredBuffer(const std::string& name) const = 0;
        virtual bool setTexture(const std::string& name, const Texture::SharedPtr& pTexture) = 0;
        virtual Texture::SharedPtr getTexture(const std::string& name) const = 0;
        virtual Sampler::SharedPtr getSampler(const std::string& name) const = 0;

        // Delete some functions. If they are not deleted, the compiler will try to convert the uints to string, resulting in runtime error
        // MAKE SURE TO ALSO DELETE THEM IN THE DERIVED CLASS!
        virtual Sampler::SharedPtr getSampler(uint32_t) const = delete;
        virtual bool setSampler(uint32_t, const Sampler::SharedPtr&) = delete;
        virtual bool setConstantBuffer(uint32_t, const ConstantBuffer::SharedPtr&) = delete;
        virtual ConstantBuffer::SharedPtr getConstantBuffer(uint32_t) const = delete;
    };

    class dlldecl IndexedVars
    {
    public:
        virtual IProgramVars* getVars() const = 0;

        /** A secondary intermediary class that allows a double [][] operator to be used on the SharedPtr.
        */
        class Var
        {
        public:
            /** Constructor gets called when mySharedPtr["myIdx1"]["myVar"] is encountered
            */
            Var(ConstantBuffer::SharedPtr pCB, const std::string& name) : mpCB(pCB), mName(name) {}

            /** Assignment operator gets called when mySharedPtr["myIdx1"]["myVar"] = T(someData); is encountered
            */
            template<typename T> void operator=(const T& val) { if (mpCB) mpCB[mName] = val; }

            /** Allows mySharedPtr["myIdx1"]["myVar"].setBlob( blobData )...
                In theory, block binary transfers could be done with operator=, but without careful coding that could *accidentally* do implicit binary transfers
            */
            template<typename T> void setBlob(const T& blob, size_t blobSz = sizeof(T)) { if (mpCB) mpCB[mName].setBlob(blob, blobSz); }
        protected:
            ConstantBuffer::SharedPtr mpCB;  ///< Storing the constant buffer from mySharedPtr["myIdx1"]
            const std::string mName;         ///< When calling mySharedPtr["myIdx1"]["myVar"] this stores "myVar"
        };

        /** An intermediary class that allows the [] operator to be used on the SharedPtr.
        */
        class Index
        {
        public:
            /** Constructor gets called when mySharedPtr["myIdx1"] is encountered
            */
            Index(IProgramVars* pVars, const std::string& name) : mpVars(pVars), mName(name) { }

            /** When a second array operator is encountered, instantiate a Var object to handle mySharedPtr["myIdx1"]["myVar"]
            */
            Var operator[](const std::string& var) { return mpVars ? Var(mpVars->getConstantBuffer(mName), var) : Var(nullptr, var); }

            Var operator[](uint32_t index) = delete; // No set by index. This is here because if we didn't explicitly delete it, the compiler will try to convert to int into a string, resulting in runtime error

            /** When encountering an assignment operator (i.e., mySharedPtr["myIdx1"] = pSomeResource;)
                set the appropriate resource for the following types:  texture, sampler, various buffers
                Note: When compiling in Release mode, these fail silently if your specified shader resource
                is of the wrong type.  In Debug mode, you hit an assert in the appropriate operator=() so you
                can check which variable is screwed up.  TODO: Log errors rather than assert()ing.
            */
            void operator=(const Texture::SharedPtr& pTexture) { mpVars->setTexture(mName, pTexture); }
            void operator=(const Sampler::SharedPtr& pSampler) { mpVars->setSampler(mName, pSampler); }

            void operator=(const TypedBufferBase::SharedPtr& pBuffer) { mpVars->setTypedBuffer(mName, pBuffer); }
            void operator=(const StructuredBuffer::SharedPtr& pBuffer) { mpVars->setStructuredBuffer(mName, pBuffer); }
            void operator=(const Buffer::SharedPtr pBuffer) { mpVars->setRawBuffer(mName, pBuffer); }
            void operator=(const std::shared_ptr<ParameterBlock>& pBlock) { mpVars->setParameterBlock(mName, pBlock); }

            /** Allow conversion of this intermediary type to a constant buffer, e.g., for allowing:
                ConstantBuffer::SharedPtr cb = mySharedPtr["myIdx1"];
            */
            operator ConstantBuffer::SharedPtr() { return mpVars->getConstantBuffer(mName); }

            void setBlob(const void* pSrc, size_t offset, size_t size)
            {
                auto pCb = mpVars->getConstantBuffer(mName);
                if (pCb) pCb->setBlob(pSrc, offset, size);
            }

            template<typename BlobData>
            void setBlob(const BlobData& src, size_t offset = 0) { return setBlob(&src, offset, sizeof(BlobData)); }

        protected:
            IProgramVars* mpVars;    ///< The variables wrapper we're using
            const std::string mName;  ///< When calling mySharedPtr["myIdx1"], stores "myIdx1"
        };

        /** Calling [] on the SharedPtr.  Create an intermediate object to process further operators
        */
        Index operator[](const std::string& var) const { return Index(getVars(), var); }
    };

    template<typename Class>
    class VarsSharedPtr : public std::shared_ptr<Class>, public IndexedVars
    {
    public:
        VarsSharedPtr() : std::shared_ptr<Class>(), IndexedVars() {}
        explicit VarsSharedPtr(Class* pProgVars) : std::shared_ptr<Class>(pProgVars) {}
        constexpr VarsSharedPtr(nullptr_t) : std::shared_ptr<Class>(nullptr) {}
        VarsSharedPtr(const std::shared_ptr<Class>& other) : std::shared_ptr<Class>(other) {}

        IProgramVars* getVars() const override { return std::shared_ptr<Class>::get()->getVars().get(); }
    };
}
