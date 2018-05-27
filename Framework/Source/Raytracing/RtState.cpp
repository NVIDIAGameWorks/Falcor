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
#include "Framework.h"
#include "RtState.h"
#include "RtProgramVars.h"

namespace Falcor
{
    RtState::SharedPtr RtState::create()
    {
        return SharedPtr(new RtState());
    }

    RtState::RtState()
    {
        mpRtsoGraph = StateGraph::create();
    }

    RtState::~RtState() = default;

    RtStateObject::ProgramList RtState::createProgramList(RtProgramVars * pVars) const
    {
        RtStateObject::ProgramList programs;
        assert(mpProgram->getRayGenProgram());
        
        programs.push_back(std::dynamic_pointer_cast<const RtProgramKernels>(mpProgram->getRayGenProgram()->getActiveVersion()->getKernels(pVars->getRayGenVars().get())));

        for (uint32_t i = 0; i < mpProgram->getHitProgramCount(); i++)
        {
            if(mpProgram->getHitProgram(i))
            {
                auto & vars = pVars->getHitVars(i);
                for (uint32_t j = 0; j < vars.size(); j++)
                    programs.push_back(std::dynamic_pointer_cast<const RtProgramKernels>(mpProgram->getHitProgram(i)->getActiveVersion()->getKernels(vars[j].get(), true)));
            }
        }

        for (uint32_t i = 0; i < mpProgram->getMissProgramCount(); i++)
        {
            if(mpProgram->getMissProgram(i))
            {
                auto & vars = pVars->getMissVars(i);
                programs.push_back(std::dynamic_pointer_cast<const RtProgramKernels>(mpProgram->getMissProgram(i)->getActiveVersion()->getKernels(vars.get())));
            }
        }

        return programs;
    }

    RtStateObject::SharedPtr RtState::getRtso(RtProgramVars* pVars)
    {
        RtStateObject::ProgramList programs = createProgramList(pVars);
        // Walk
        for (const auto& p : programs)
        {
            mpRtsoGraph->walk((void*)p.get());
        }

        RtStateObject::SharedPtr pRtso = mpRtsoGraph->getCurrentNode();

        if (pRtso == nullptr)
        {
            RtStateObject::Desc desc;
            desc.setProgramList(programs).setMaxTraceRecursionDepth(mMaxTraceRecursionDepth);
            desc.setGlobalRootSignature(mpProgram->getGlobalRootSignature());

            StateGraph::CompareFunc cmpFunc = [&desc](RtStateObject::SharedPtr pRtso) -> bool
            {
                return pRtso && (desc == pRtso->getDesc());
            };

            if (mpRtsoGraph->scanForMatchingNode(cmpFunc))
            {
                pRtso = mpRtsoGraph->getCurrentNode();
            }
            else
            {
                pRtso = RtStateObject::create(desc);
                mpRtsoGraph->setCurrentNodeData(pRtso);
            }
        }

        return pRtso;
    }

    void RtState::setMaxTraceRecursionDepth(uint32_t maxDepth)
    {
        if (mMaxTraceRecursionDepth != maxDepth)
        {
            uint64_t edge = (uint64_t)maxDepth;
            mpRtsoGraph->walk((void*)edge);
        }
        mMaxTraceRecursionDepth = maxDepth;
    }
}
