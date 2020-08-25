/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "DescriptorSet.h"

namespace Falcor
{
    class ProgramReflection;
    class EntryPointGroupReflection;
    class CopyContext;

    /** The root signature defines what resources are bound to the pipeline.

        The layout is defined by traversing the ParameterBlock hierarchy
        of a program to find all required root parameters. These are then
        arranged consecutively in the following order in the root signature:

        1. descriptor tables
        2. root descriptors
        3. root constants

        The get*BaseIndex() functions return the base index of the
        corresponding root parameter type in the root signature.
    */
    class dlldecl RootSignature
    {
    public:
        using SharedPtr = std::shared_ptr<RootSignature>;
        using SharedConstPtr = std::shared_ptr<const RootSignature>;
        using ApiHandle = RootSignatureHandle;

        using DescType = Falcor::DescriptorSet::Type;
        using DescriptorSetLayout = DescriptorSet::Layout;

        struct RootDescriptorDesc
        {
            DescType type;
            uint32_t regIndex;
            uint32_t spaceIndex;
            ShaderVisibility visibility;
        };

        struct RootConstantsDesc
        {
            uint32_t regIndex;
            uint32_t spaceIndex;
            uint32_t count;
        };

        class dlldecl Desc
        {
        public:
            Desc& addDescriptorSet(const DescriptorSetLayout& setLayout);
            Desc& addRootDescriptor(DescType type, uint32_t regIndex, uint32_t spaceIndex, ShaderVisibility visibility = ShaderVisibility::All);
            Desc& addRootConstants(uint32_t regIndex, uint32_t spaceIndex, uint32_t count); // #SHADER_VAR Make sure this works with the reflectors

#ifdef FALCOR_D3D12
            Desc& setLocal(bool isLocal) { mIsLocal = isLocal; return *this; }
#endif

            size_t getSetsCount() const { return mSets.size(); }
            const DescriptorSetLayout& getSet(size_t index) const { return mSets[index]; }

            size_t getRootDescriptorCount() const { return mRootDescriptors.size(); }
            const RootDescriptorDesc& getRootDescriptorDesc(size_t index) const { return mRootDescriptors[index]; }

            size_t getRootConstantCount() const { return mRootConstants.size(); }
            const RootConstantsDesc& getRootConstantDesc(size_t index) const { return mRootConstants[index]; }

        private:
            friend class RootSignature;

            std::vector<DescriptorSetLayout> mSets;
            std::vector<RootDescriptorDesc> mRootDescriptors;
            std::vector<RootConstantsDesc> mRootConstants;

#ifdef FALCOR_D3D12
            bool mIsLocal = false;
#endif
        };

        ~RootSignature();

        /** Get an empty root signature.
            \return Empty root signature, or throws an exception on error.
        */
        static SharedPtr getEmpty();

        /** Create a root signature.
            \param[in] desc Root signature description.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create(const Desc& desc);

        /** Create a root signature from program reflection.
            \param[in] pReflection Reflection object.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create(const ProgramReflection* pReflection);

        /** Create a local root signature for use with DXR.
            \param[in] pReflection Reflection object.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr createLocal(const EntryPointGroupReflection* pReflector);

        const ApiHandle& getApiHandle() const { return mApiHandle; }

        size_t getDescriptorSetCount() const { return mDesc.mSets.size(); }
        const DescriptorSetLayout& getDescriptorSet(size_t index) const { return mDesc.mSets[index]; }

        uint32_t getDescriptorSetBaseIndex() const { return 0; }
        uint32_t getRootDescriptorBaseIndex() const { return getDescriptorSetBaseIndex() + (uint32_t)mDesc.mSets.size(); }
        uint32_t getRootConstantBaseIndex() const { return getRootDescriptorBaseIndex() + (uint32_t)mDesc.mRootDescriptors.size(); }

        uint32_t getSizeInBytes() const { return mSizeInBytes; }
        uint32_t getElementByteOffset(uint32_t elementIndex) { return mElementByteOffset[elementIndex]; }

        void bindForGraphics(CopyContext* pCtx);
        void bindForCompute(CopyContext* pCtx);

        const Desc& getDesc() const { return mDesc; }

    protected:
        RootSignature(const Desc& desc);
        void apiInit();
#ifdef FALCOR_D3D12
        virtual void createApiHandle(ID3DBlobPtr pSigBlob);
#endif
        ApiHandle mApiHandle;
        Desc mDesc;
        static SharedPtr spEmptySig;
        static uint64_t sObjCount;

        uint32_t mSizeInBytes;
        std::vector<uint32_t> mElementByteOffset;
    };
}
