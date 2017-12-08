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
#include "Graphics/Program/ProgramReflection.h"
#include "API/ConstantBuffer.h"
#include "API/StructuredBuffer.h"
#include "API/TypedBuffer.h"

namespace Falcor
{
    class RootSignature;
    class ProgramVars;

    class ParameterBlock : public std::enable_shared_from_this<ParameterBlock>
    {
    public:
        template<typename T>
        class SharedPtrT : public std::shared_ptr<T>
        {
        public:
            SharedPtrT() : std::shared_ptr<T>() {}
            SharedPtrT(T* pParamBlock) : std::shared_ptr<T>(pParamBlock) {}
            ConstantBuffer::SharedPtr operator[](const std::string& cbName) const { return get()->getConstantBuffer(cbName); }
            ConstantBuffer::SharedPtr operator[](uint32_t index) = delete; // No set by index. This is here because if we didn't explicitly delete it, the compiler will try to convert to int into a string, resulting in runtime error
        };

        using SharedPtr = SharedPtrT<ParameterBlock>;
        using SharedConstPtr = SharedPtrT<const ParameterBlock>;
        ~ParameterBlock();

        /** Create a new object
        */
        static SharedPtr create(const ParameterBlockReflection::SharedConstPtr& pReflection, const RootSignature* pRootSig, bool createBuffers);

        /** Bind a constant buffer object by name.
        If the name doesn't exists or the CBs size doesn't match the required size, the call will fail.
        If a buffer was previously bound it will be released.
        \param[in] name The name of the constant buffer in the program
        \param[in] pCB The constant buffer object
        \return false is the call failed, otherwise true
        */
        bool setConstantBuffer(const std::string& name, const ConstantBuffer::SharedPtr& pCB);

        /** Bind a constant buffer object by index.
        If the no CB exists in the specified index or the CB size doesn't match the required size, the call will fail.
        If a buffer was previously bound it will be released.
        \param[in] setIndex The set-index in the block
        \param[in] descIndex The descriptor index inside the set
        \param[in] pCB The constant buffer object
        \return false is the call failed, otherwise true
        */
        bool setConstantBuffer(uint32_t setIndex, uint32_t rangeIndex, uint32_t arrayIndex, const ConstantBuffer::SharedPtr& pCB);

        /** Get a constant buffer object.
        \param[in] name The name of the buffer
        \return If the name is valid, a shared pointer to the CB. Otherwise returns nullptr
        */
        ConstantBuffer::SharedPtr getConstantBuffer(const std::string& name) const;

        /** Get a constant buffer object.
        \param[in] setIndex The set-index in the block
        \param[in] descIndex The descriptor index inside the set
        \return If the indices is valid, a shared pointer to the buffer. Otherwise returns nullptr
        */
        ConstantBuffer::SharedPtr getConstantBuffer(uint32_t setIndex, uint32_t rangeIndex, uint32_t arrayIndex) const;

        /** Set a raw-buffer. Based on the shader reflection, it will be bound as either an SRV or a UAV
        \param[in] name The name of the buffer
        \param[in] pBuf The buffer object
        */
        bool setRawBuffer(const std::string& name, Buffer::SharedPtr pBuf);

        /** Set a typed buffer. Based on the shader reflection, it will be bound as either an SRV or a UAV
        \param[in] name The name of the buffer
        \param[in] pBuf The buffer object
        */
        bool setTypedBuffer(const std::string& name, TypedBufferBase::SharedPtr pBuf);
        
        /** Set a structured buffer. Based on the shader reflection, it will be bound as either an SRV or a UAV
        \param[in] name The name of the buffer
        \param[in] pBuf The buffer object
        */
        bool setStructuredBuffer(const std::string& name, StructuredBuffer::SharedPtr pBuf);

        /** Get a raw-buffer object.
        \param[in] name The name of the buffer
        \return If the name is valid, a shared pointer to the buffer object. Otherwise returns nullptr
        */
        Buffer::SharedPtr getRawBuffer(const std::string& name) const;

        /** Get a typed buffer object.
        \param[in] name The name of the buffer
        \return If the name is valid, a shared pointer to the buffer object. Otherwise returns nullptr
        */
        TypedBufferBase::SharedPtr getTypedBuffer(const std::string& name) const;

        /** Get a structured buffer object.
        \param[in] name The name of the buffer
        \return If the name is valid, a shared pointer to the buffer object. Otherwise returns nullptr
        */
        StructuredBuffer::SharedPtr getStructuredBuffer(const std::string& name) const;

        /** Bind a texture. Based on the shader reflection, it will be bound as either an SRV or a UAV
        \param[in] name The name of the texture object in the shader
        \param[in] pTexture The texture object to bind
        */
        bool setTexture(const std::string& name, const Texture::SharedPtr& pTexture);

        /** Get a texture object.
        \param[in] name The name of the texture
        \return If the name is valid, a shared pointer to the texture object. Otherwise returns nullptr
        */
        Texture::SharedPtr getTexture(const std::string& name) const;

        /** Bind an SRV.
        \param[in] setIndex The set-index in the block
        \param[in] descIndex The descriptor index inside the set
        \param[in] pSrv The shader-resource-view object to bind
        */
        bool setSrv(uint32_t setIndex, uint32_t rangeIndex, uint32_t arrayIndex, const ShaderResourceView::SharedPtr& pSrv);

        /** Bind a UAV.
        \param[in] setIndex The set-index in the block
        \param[in] descIndex The descriptor index inside the set
        \param[in] pSrv The unordered-access-view object to bind
        */
        bool setUav(uint32_t setIndex, uint32_t rangeIndex, uint32_t arrayIndex, const UnorderedAccessView::SharedPtr& pUav);

        /** Get an SRV object.
        \param[in] setIndex The set-index in the block
        \param[in] descIndex The descriptor index inside the set
        \return If the indices is valid, a shared pointer to the SRV. Otherwise returns nullptr
        */
        ShaderResourceView::SharedPtr getSrv(uint32_t setIndex, uint32_t rangeIndex, uint32_t arrayIndex) const;

        /** Get a UAV object
        \param[in] setIndex The set-index in the block
        \param[in] descIndex The descriptor index inside the set
        \return If the index is valid, a shared pointer to the UAV. Otherwise returns nullptr
        */
        UnorderedAccessView::SharedPtr getUav(uint32_t setIndex, uint32_t rangeIndex, uint32_t arrayIndex) const;

        /** Bind a sampler to the program in the global namespace.
        \param[in] name The name of the sampler object in the shader
        \param[in] pSampler The sampler object to bind
        \return false if the sampler was not found in the program, otherwise true
        */
        bool setSampler(const std::string& name, const Sampler::SharedPtr& pSampler);

        /** Bind a sampler to the program in the global namespace.
        \param[in] setIndex The set-index in the block
        \param[in] descIndex The descriptor index inside the set
        \return false if the sampler was not found in the program, otherwise true
        */
        bool setSampler(uint32_t setIndex, uint32_t rangeIndex, uint32_t arrayIndex, const Sampler::SharedPtr& pSampler);

        /** Gets a sampler object.
        \return If the index is valid, a shared pointer to the sampler. Otherwise returns nullptr
        */
        Sampler::SharedPtr getSampler(const std::string& name) const;

        /** Gets a sampler object.
        \param[in] setIndex The set-index in the block
        \param[in] descIndex The descriptor index inside the set
        \return If the index is valid, a shared pointer to the sampler. Otherwise returns nullptr
        */
        Sampler::SharedPtr getSampler(uint32_t setIndex, uint32_t rangeIndex, uint32_t arrayIndex) const;

        /** Get the program reflection interface
        */
        ParameterBlockReflection::SharedConstPtr getReflection() const { return mpReflector; }

        bool prepareForDraw(CopyContext* pContext);
       
        // Delete some functions. If they are not deleted, the compiler will try to convert the uints to string, resulting in runtime error
        Sampler::SharedPtr getSampler(uint32_t) const = delete;
        bool setSampler(uint32_t, const Sampler::SharedPtr&) = delete;
        bool setConstantBuffer(uint32_t, const ConstantBuffer::SharedPtr&) = delete;
        ConstantBuffer::SharedPtr getConstantBuffer(uint32_t) const = delete;

        // #PARAMBLOCK I don't like it. This should be private
        struct RootSet
        {
            DescriptorSet::SharedPtr pSet;
            bool dirty = true;
        };
        std::vector<RootSet>& getRootSets() { return mRootSets; }
    private:
        ParameterBlock(const ParameterBlockReflection::SharedConstPtr& pReflection, const RootSignature* pRootSig, bool createBuffers);
        ParameterBlockReflection::SharedConstPtr mpReflector;
        friend class ProgramVars;

        struct AssignedResource
        {
            Resource::SharedPtr pResource = nullptr;
            AssignedResource();
            AssignedResource(const AssignedResource& other);
            ~AssignedResource();

            DescriptorSet::Type type;
            union
            {
                ConstantBuffer::SharedPtr      pCB;
                ShaderResourceView::SharedPtr  pSRV;
                UnorderedAccessView::SharedPtr pUAV;
                Sampler::SharedPtr pSampler;
            };
            size_t requiredSize = 0;
        };
        using ResourceVec = std::vector<AssignedResource>;
        using SetResourceVec = std::vector<ResourceVec>;
        std::vector<SetResourceVec> mAssignedResources;
        bool checkResourceIndices(uint32_t setIndex, uint32_t rangeIndex, uint32_t arrayIndex, DescriptorSet::Type type, const std::string& funcName) const;

        std::vector<RootSet> mRootSets;
        void setResourceSrvUavCommon(const std::string& name, uint32_t descOffset, DescriptorSet::Type type, const Resource::SharedPtr& pResource, const std::string& funcName);
        template<typename ResourceType>
        typename ResourceType::SharedPtr getResourceSrvUavCommon(const std::string& name, uint32_t descOffset, DescriptorSet::Type type, const std::string& funcName) const;
    };
}