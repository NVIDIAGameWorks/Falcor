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
#include "RtProgram/RtProgram.h"
#include "RtStateObject.h"
#include "Utils/Graph.h"

namespace Falcor
{
    class RtState : public std::enable_shared_from_this<RtState>
    {
    public:
        using SharedPtr = std::shared_ptr<RtState>;
        using SharedConstPtr = std::shared_ptr<const RtState>;
        
        static SharedPtr create();
        ~RtState();

        void setProgram(RtProgram::SharedPtr pProg) { mpProgram = pProg; }
        RtProgram::SharedPtr getProgram() const { return mpProgram; }

        void setMaxTraceRecursionDepth(uint32_t maxDepth);
        uint32_t getMaxTraceRecursionDepth() const { return mMaxTraceRecursionDepth; }

        void setProgramStackSize(uint32_t stackSize);

        RtStateObject::SharedPtr getRtso(RtProgramVars* pVars);
    private:
        RtState();
        RtProgram::SharedPtr mpProgram;
        uint32_t mMaxTraceRecursionDepth = 1;
        using StateGraph = Graph<RtStateObject::SharedPtr, void*>;
        StateGraph::SharedPtr mpRtsoGraph;

        RtStateObject::ProgramList createProgramList(RtProgramVars * pVars) const;
    };
}
