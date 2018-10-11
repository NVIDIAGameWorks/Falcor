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
#include "RtProgram/RtProgramVersion.h"

namespace Falcor
{
    class RtStateObject : public std::enable_shared_from_this<RtStateObject>
    {
    public:
        using SharedPtr = std::shared_ptr<RtStateObject>;
        using SharedConstPtr = std::shared_ptr<const RtStateObject>;
        using ApiHandle = ID3D12StateObjectPtr;

        using ProgramList = std::vector<RtProgramVersion::SharedConstPtr>;

        class Desc
        {
        public:
            Desc& setProgramList(const ProgramList& list) { mProgList = list; return *this; }
            Desc& setMaxTraceRecursionDepth(uint32_t maxDepth) { mMaxTraceRecursionDepth = maxDepth; return *this; }
            Desc& setGlobalRootSignature(const std::shared_ptr<RootSignature>& pRootSig) { mpGlobalRootSignature = pRootSig; return *this; }
            bool operator==(const Desc& other) const;

        private:
            ProgramList mProgList;
            std::shared_ptr<RootSignature> mpGlobalRootSignature;
            uint32_t mMaxTraceRecursionDepth = 1;
            friend RtStateObject;
        };

        static SharedPtr create(const Desc& desc);
        const ApiHandle& getApiHandle() const { return mApiHandle; }

        const ProgramList& getProgramList() const { return mDesc.mProgList; }
        uint32_t getMaxTraceRecursionDepth() const { return mDesc.mMaxTraceRecursionDepth; }
        const std::shared_ptr<RootSignature>& getGlobalRootSignature() const { return mDesc.mpGlobalRootSignature; }
        const Desc& getDesc() const { return mDesc; }
    private:
        RtStateObject(const Desc& d) : mDesc(d) {}
        ApiHandle mApiHandle;
        Desc mDesc;
    };
}
