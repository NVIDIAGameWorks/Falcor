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
#include "Core/Program/ProgramVersion.h"
#include "Core/API/RootSignature.h"

namespace Falcor
{
    class dlldecl ComputeStateObject
    {
    public:
        using SharedPtr = std::shared_ptr<ComputeStateObject>;
        using SharedConstPtr = std::shared_ptr<const ComputeStateObject>;
        using ApiHandle = ComputeStateHandle;

        class dlldecl Desc
        {
        public:
            Desc& setRootSignature(RootSignature::SharedPtr pSignature) { mpRootSignature = pSignature; return *this; }
            Desc& setProgramKernels(const ProgramKernels::SharedConstPtr& pProgram) { mpProgram = pProgram; return *this; }
            const ProgramKernels::SharedConstPtr getProgramKernels() const { return mpProgram; }
            ProgramVersion::SharedConstPtr getProgramVersion() const { return mpProgram->getProgramVersion(); }
            bool operator==(const Desc& other) const;
        private:
            friend class ComputeStateObject;
            ProgramKernels::SharedConstPtr mpProgram;
            RootSignature::SharedPtr mpRootSignature;
        };

        ~ComputeStateObject();

        /** Create a compute state object.
            \param[in] desc State object description.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create(const Desc& desc);

        const ApiHandle& getApiHandle() { return mApiHandle; }
        const Desc& getDesc() const { return mDesc; }

    private:
        ComputeStateObject(const Desc& desc);
        void apiInit();

        Desc mDesc;
        ApiHandle mApiHandle;
    };
}
