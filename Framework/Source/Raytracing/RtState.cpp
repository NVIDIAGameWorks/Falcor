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

    static RtProgramKernels::SharedConstPtr castToRtKernels(ProgramKernels::SharedConstPtr const& pKernels)
    {
        auto pRtKernels = dynamic_cast<const RtProgramKernels*>(pKernels.get());
        assert(pRtKernels);
        return RtProgramKernels::SharedConstPtr(pKernels, pRtKernels);
    }

    RtPipelineKernels::SharedPtr RtPipelineVersion::getKernels(RtProgramVars* pVars) const
    {
        assert(mpRayGenProgram);
        auto pRayGenKernels = castToRtKernels(mpRayGenProgram->getKernels(pVars->getRayGenVars().get()));

        std::vector<RtProgramKernels::SharedConstPtr> hitKernels;
        auto hitProgramCount = mHitPrograms.size();
        for (uint32_t i = 0; i < hitProgramCount; i++)
        {
            if(auto pHitProgram = mHitPrograms[i])
            {
                hitKernels.push_back(castToRtKernels(pHitProgram->getKernels(pVars->getHitVars(i)[0].get())));
            }
        }

        std::vector<RtProgramKernels::SharedConstPtr> missKernels;
        auto missProgramCount = mMissPrograms.size();
        for (uint32_t i = 0; i < missProgramCount; i++)
        {
            if(auto pMissProgram = mMissPrograms[i])
            {
                missKernels.push_back(castToRtKernels(pMissProgram->getKernels(pVars->getMissVars(i).get())));
            }
        }

        mpKernelGraph->walk((void*)pRayGenKernels.get());
        for( auto pHitKernels : hitKernels )
        {
            mpKernelGraph->walk((void*)pHitKernels.get());
        }
        for( auto pMissKernels : missKernels )
        {
            mpKernelGraph->walk((void*)pMissKernels.get());
        }

        RtPipelineKernels::SharedPtr pKernels = mpKernelGraph->getCurrentNode();
        if(pKernels)
            return pKernels;

        KernelGraph::CompareFunc cmpFunc = [&](RtPipelineKernels::SharedPtr pKernels) -> bool
        {
            if(!pKernels) return false;

            //
            if(pRayGenKernels != pKernels->getRayGenProgram()) return false;

            auto hitProgramCount = hitKernels.size();
            if(hitProgramCount != pKernels->getHitProgramCount()) return false;

            auto missProgramCount = missKernels.size();
            if(missProgramCount != pKernels->getMissProgramCount()) return false;

            for( size_t i = 0; i < hitProgramCount; ++i )
            {
                if(hitKernels[i] != pKernels->getHitProgram(i)) return false;
            }
            for( size_t i = 0; i < missProgramCount; ++i )
            {
                if(missKernels[i] != pKernels->getMissProgram(i)) return false;
            }

            return true;
        };

        if( mpKernelGraph->scanForMatchingNode(cmpFunc) )
        {
            pKernels = mpKernelGraph->getCurrentNode();
        }
        else
        {
            pKernels = RtPipelineKernels::create(
                mpGlobalKernels,
                pRayGenKernels,
                hitKernels,
                missKernels);
            mpKernelGraph->setCurrentNodeData(pKernels);
        }

        return pKernels;
    }

    RtStateObject::SharedPtr RtState::getRtso(RtProgramVars* pVars)
    {
        auto pKernels = mpProgram->getKernels(pVars);

        // Walk
        mpRtsoGraph->walk((void*)pKernels.get());

        RtStateObject::SharedPtr pRtso = mpRtsoGraph->getCurrentNode();

        if (pRtso == nullptr)
        {
            RtStateObject::Desc desc;
            desc.setKernels(pKernels).setMaxTraceRecursionDepth(mMaxTraceRecursionDepth);
            desc.setGlobalRootSignature(pKernels->getGlobalRootSignature());

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
