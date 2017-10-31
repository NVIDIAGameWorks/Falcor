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
#include "API/Sampler.h"
#include "API/DescriptorSet.h"

namespace Falcor
{
    class ProgramReflection;
    class CopyContext;

    class RootSignature
    {
    public:
        using SharedPtr = std::shared_ptr<RootSignature>;
        using SharedConstPtr = std::shared_ptr<const RootSignature>;
        using ApiHandle = RootSignatureHandle;
        
        using DescType = Falcor::DescriptorSet::Type;
        using DescriptorSetLayout = DescriptorSet::Layout;
        
        class Desc
        {
        public:
            Desc& addDescriptorSet(const DescriptorSetLayout& setLayout);
        private:
            friend class RootSignature;
            std::vector<DescriptorSetLayout> mSets;
        };

        ~RootSignature();
        static SharedPtr getEmpty();
        static SharedPtr create(const Desc& desc);
        static SharedPtr create(const ProgramReflection* pReflection);

        ApiHandle getApiHandle() const { return mApiHandle; }

        size_t getDescriptorSetCount() const { return mDesc.mSets.size(); }
        const DescriptorSetLayout& getDescriptorSet(size_t index) const { return mDesc.mSets[index]; }

        uint32_t getSizeInBytes() const { return mSizeInBytes; }
        uint32_t getElementByteOffset(uint32_t elementIndex) { return mElementByteOffset[elementIndex]; }

        void bindForGraphics(CopyContext* pCtx);
        void bindForCompute(CopyContext* pCtx);
    protected:
        RootSignature(const Desc& desc);
        bool apiInit();
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