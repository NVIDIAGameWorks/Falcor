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
#include "API/Texture.h"
#include "API/ConstantBuffer.h"
#include "API/Sampler.h"
#include <unordered_map>
#include "ProgramReflection.h"
#include "API/StructuredBuffer.h"
#include "API/TypedBuffer.h"
#include "API/LowLevel/RootSignature.h"
#include "Graphics/Program/ParameterBlock.h"

namespace Falcor
{
    class ProgramVersion;
    class ComputeContext;
    class DescriptorSet;

    /** This class manages a program's reflection and variable assignment.
        It's a high-level abstraction of variables-related concepts such as CBs, texture and sampler assignments, root-signature, descriptor tables, etc.
    */
    class ProgramVars
    {
    public:
        template<typename T>
        class SharedPtrT : public std::shared_ptr<T>
        {
        public:
            SharedPtrT() : std::shared_ptr<T>() {}
            SharedPtrT(T* pProgVars) : std::shared_ptr<T>(pProgVars) {}
            ConstantBuffer::SharedPtr operator[](const std::string& cbName) { return std::shared_ptr<T>::get()->getConstantBuffer(cbName); }
            ConstantBuffer::SharedPtr operator[](uint32_t index) = delete; // No set by index. This is here because if we didn't explicitly delete it, the compiler will try to convert to int into a string, resulting in runtime error
        };

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
            Please note that the register space and index are the global indices used in the program. Do not confuse those indices with ParameterBlock::BindLocation.
            \param[in] regSpace The register space
            \param[in] baseRegIndex The base register index
            \param[in] arrayIndex The array index, or 0 for non-array variables
            \param[in] pCB The constant buffer object
            \return false is the call failed, otherwise true
        */
        bool setConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ConstantBuffer::SharedPtr& pCB);

        /** Get a constant buffer object.
            \param[in] name The name of the buffer
            \return If the name is valid, a shared pointer to the CB. Otherwise returns nullptr
        */
        ConstantBuffer::SharedPtr getConstantBuffer(const std::string& name) const;

        /** Get a constant buffer object.
            Please note that the register space and index are the global indices used in the program. Do not confuse those indices with ParameterBlock::BindLocation.
            \param[in] regSpace The register space
            \param[in] baseRegIndex The base register index
            \param[in] arrayIndex The array index, or 0 for non-array variables
            \return If the indices are valid, a shared pointer to the buffer. Otherwise returns nullptr
        */
        ConstantBuffer::SharedPtr getConstantBuffer(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const;

        /** Set a raw-buffer. Based on the shader reflection, it will be bound as either an SRV or a UAV
            \param[in] name The name of the buffer
            \param[in] pBuf The buffer object
            \return false is the call failed, otherwise true
        */
        bool setRawBuffer(const std::string& name, Buffer::SharedPtr pBuf);
        
        /** Set a typed buffer. Based on the shader reflection, it will be bound as either an SRV or a UAV
            \param[in] name The name of the buffer
            \param[in] pBuf The buffer object
            \return false is the call failed, otherwise true
        */
        bool setTypedBuffer(const std::string& name, TypedBufferBase::SharedPtr pBuf);

        /** Set a structured buffer. Based on the shader reflection, it will be bound as either an SRV or a UAV
            \param[in] name The name of the buffer
            \param[in] pBuf The buffer object
            \return false is the call failed, otherwise true
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
            \return false is the call failed, otherwise true
        */
        bool setTexture(const std::string& name, const Texture::SharedPtr& pTexture);

        /** Get a texture object.
            \param[in] name The name of the texture
            \return If the name is valid, a shared pointer to the texture object. Otherwise returns nullptr
        */
        Texture::SharedPtr getTexture(const std::string& name) const;

        /** Bind an SRV.
            Please note that the register space and index are the global indices used in the program. Do not confuse those indices with ParameterBlock::BindLocation.
            \param[in] regSpace The register space
            \param[in] baseRegIndex The base register index
            \param[in] arrayIndex The array index, or 0 for non-array variables
            \param[in] pSrv The shader-resource-view. If it's nullptr, will set a view to a default (black) texture.
            \return false is the call failed, otherwise true
        */
        bool setSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const ShaderResourceView::SharedPtr& pSrv);

        /** Bind a UAV.
            Please note that the register space and index are the global indices used in the program. Do not confuse those indices with ParameterBlock::BindLocation.
            \param[in] regSpace The register space
            \param[in] baseRegIndex The base register index
            \param[in] arrayIndex The array index, or 0 for non-array variables
            \param[in] pUav The unordered-access-view. If it's nullptr, will set a view to a default (black) texture.
            \return false is the call failed, otherwise true
        */
        bool setUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const UnorderedAccessView::SharedPtr& pUav);

        /** Get an SRV object.
            Please note that the register space and index are the global indices used in the program. Do not confuse those indices with ParameterBlock::BindLocation.
            \param[in] regSpace Register space the SRV is located in
            \param[in] baseRegIndex Register index the SRV is located at
            \param[in] arrayIndex Index into array, if applicable. Use 0 otherwise
            \return If the indices are valid, a shared pointer to the SRV. Otherwise returns nullptr
        */
        ShaderResourceView::SharedPtr getSrv(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const;

        /** Get a UAV object
            Please note that the register space and index are the global indices used in the program. Do not confuse those indices with ParameterBlock::BindLocation.
            \param[in] regSpace Register space the UAV is located in
            \param[in] baseRegIndex Register index the UAV is located at
            \param[in] arrayIndex Index into array, if applicable. Use 0 otherwise
            \return If the indices are valid, a shared pointer to the UAV. Otherwise returns nullptr
        */
        UnorderedAccessView::SharedPtr getUav(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const;

        /** Bind a sampler to the program in the global namespace.
            \param[in] name The name of the sampler object in the shader
            \param[in] pSampler The sampler object to bind
            \return false if the sampler was not found in the program, otherwise true
        */
        bool setSampler(const std::string& name, const Sampler::SharedPtr& pSampler);

        /** Bind a sampler to the program in the global namespace.
            Please note that the register space and index are the global indices used in the program. Do not confuse those indices with ParameterBlock::BindLocation.
            \param[in] regSpace Register space the sampler is located in
            \param[in] baseRegIndex Register index the sampler is located at
            \param[in] arrayIndex Index into sampler array, if applicable. Use 0 otherwise
            \param[in] The sampler object
            \return false if the sampler was not found in the program, otherwise true
        */
        bool setSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex, const Sampler::SharedPtr& pSampler);

        /** Gets a sampler object.
            \return If the index is valid, a shared pointer to the sampler. Otherwise returns nullptr
        */
        Sampler::SharedPtr getSampler(const std::string& name) const;

        /** Gets a sampler object.
            Please note that the register space and index are the global indices used in the program. Do not confuse those indices with ParameterBlock::BindLocation.
            \param[in] regSpace Register space the sampler is located in
            \param[in] baseRegIndex Register index the sampler is located at
            \param[in] arrayIndex Index into sampler array, if applicable. Use 0 otherwise
            \param[in] The sampler object
            \return If the index is valid, a shared pointer to the sampler. Otherwise returns nullptr
        */
        Sampler::SharedPtr getSampler(uint32_t regSpace, uint32_t baseRegIndex, uint32_t arrayIndex) const;

        /** Get the program reflection interface
        */
        ProgramReflection::SharedConstPtr getReflection() const { return mpReflector; }

        /** Get the root signature object
        */
        RootSignature::SharedPtr getRootSignature() const { return mpRootSignature; }

        /** Get the number of parameter-blocks
        */
        uint32_t getParameterBlockCount() const { return (uint32_t)mParameterBlocks.size(); }
        
        /** Get a list of indices translating a parameter-block's set index to the root-signature entry index
        */
        const std::vector<uint32_t>& getParameterBlockRootIndices(uint32_t blockIndex) const { return mParameterBlocks[blockIndex].rootIndex; }

        /** Get parameter-block by index. You can translate a name to an index using the ProgramReflection object
        */
        const ParameterBlock::SharedConstPtr getParameterBlock(uint32_t blockIndex) const;

        /** Get parameter-block by name
        */
        const ParameterBlock::SharedConstPtr getParameterBlock(const std::string& name) const;

        /** Set parameter-block by index. You can translate a name to an index using the ProgramReflection object
        */
        void setParameterBlock(uint32_t blockIndex, const ParameterBlock::SharedConstPtr& pBlock);

        /** Set parameter-block by name
        */
        void setParameterBlock(const std::string& name, const ParameterBlock::SharedConstPtr& pBlock);

        /** Get the default parameter-block
        */
        const ParameterBlock::SharedPtr& getDefaultBlock() const { return mDefaultBlock.pBlock; }

        // Delete some functions. If they are not deleted, the compiler will try to convert the uints to string, resulting in runtime error
        Sampler::SharedPtr getSampler(uint32_t) const = delete;
        bool setSampler(uint32_t, const Sampler::SharedPtr&) = delete;
        bool setConstantBuffer(uint32_t, const ConstantBuffer::SharedPtr&) = delete;
        ConstantBuffer::SharedPtr getConstantBuffer(uint32_t) const = delete;

        template<bool forGraphics>
        bool applyProgramVarsCommon(CopyContext* pContext, bool bindRootSig);

    protected:
        ProgramVars(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig);
        
        RootSignature::SharedPtr mpRootSignature;
        ProgramReflection::SharedConstPtr mpReflector;

        struct BlockData
        {
            ParameterBlock::SharedPtr pBlock;
            std::vector<uint32_t> rootIndex;        // Maps the block's set-index to the root-signature entry
            bool bind = true;
        };
        BlockData mDefaultBlock;
        std::vector<BlockData> mParameterBlocks; // First element is the global block
        ProgramVars::BlockData initParameterBlock(const ParameterBlockReflection::SharedConstPtr& pBlockReflection, bool createBuffers);
    };

    class GraphicsVars : public ProgramVars, public std::enable_shared_from_this<ProgramVars>
    {
    public:
        using SharedPtr = SharedPtrT<GraphicsVars>;
        using SharedConstPtr = std::shared_ptr<const GraphicsVars>;

        /** Create a new object
            \param[in] pReflector A program reflection object containing the requested declarations
            \param[in] createBuffers If true, will create the ConstantBuffer objects. Otherwise, the user will have to bind the CBs himself
            \param[in] pRootSignature A root-signature describing how to bind resources into the shader. If this parameter is nullptr, a root-signature object will be created from the program reflection object
        */
        static SharedPtr create(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers = true, const RootSignature::SharedPtr& pRootSig = nullptr);
        bool apply(RenderContext* pContext, bool bindRootSig);
    private:
        GraphicsVars(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig) :
            ProgramVars(pReflector, createBuffers, pRootSig) {}
    };

    class ComputeVars : public ProgramVars, public std::enable_shared_from_this<ProgramVars>
    {
    public:
        using SharedPtr = SharedPtrT<ComputeVars>;
        using SharedConstPtr = std::shared_ptr<const ComputeVars>;

        /** Create a new object
            \param[in] pReflector A program reflection object containing the requested declarations
            \param[in] createBuffers If true, will create the ConstantBuffer objects. Otherwise, the user will have to bind the CBs himself
            \param[in] pRootSignature A root-signature describing how to bind resources into the shader. If this parameter is nullptr, a root-signature object will be created from the program reflection object
        */
        static SharedPtr create(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers = true, const RootSignature::SharedPtr& pRootSig = nullptr);
        bool apply(ComputeContext* pContext, bool bindRootSig);
    private:
        ComputeVars(const ProgramReflection::SharedConstPtr& pReflector, bool createBuffers, const RootSignature::SharedPtr& pRootSig) :
            ProgramVars(pReflector, createBuffers, pRootSig) {}
    };
}