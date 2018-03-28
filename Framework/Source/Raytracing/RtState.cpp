/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
#include "Framework.h"
#include "RtState.h"

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

    RtStateObject::ProgramList RtState::createProgramList() const
    {
        RtStateObject::ProgramList programs;
        if (mpProgram->getRayGenProgram())
        {
            programs.push_back(mpProgram->getRayGenProgram()->getActiveVersion());
        }

        for (uint32_t i = 0; i < mpProgram->getHitProgramCount(); i++)
        {
            programs.push_back(mpProgram->getHitProgram(i)->getActiveVersion());

        }

        for (uint32_t i = 0; i < mpProgram->getMissProgramCount(); i++)
        {
            programs.push_back(mpProgram->getMissProgram(i)->getActiveVersion());
        }

        return programs;
    }

    RtStateObject::SharedPtr RtState::getRtso()
    {
        RtStateObject::ProgramList programs = createProgramList();
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
