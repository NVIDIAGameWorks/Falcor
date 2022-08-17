/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "ComputeState.h"
#include "Core/Program/ProgramVars.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    ComputeState::ComputeState()
    {
        mpCsoGraph = StateGraph::create();
    }

    ComputeStateObject::SharedPtr ComputeState::getCSO(const ComputeVars* pVars)
    {
        auto pProgramKernels = mpProgram ? mpProgram->getActiveVersion()->getKernels(pVars) : nullptr;
        bool newProgram = (pProgramKernels.get() != mCachedData.pProgramKernels);
        if (newProgram)
        {
            mCachedData.pProgramKernels = pProgramKernels.get();
            mpCsoGraph->walk((void*)mCachedData.pProgramKernels);
        }

        ComputeStateObject::SharedPtr pCso = mpCsoGraph->getCurrentNode();

        if(pCso == nullptr)
        {
            mDesc.setProgramKernels(pProgramKernels);

            StateGraph::CompareFunc cmpFunc = [&desc = mDesc](ComputeStateObject::SharedPtr pCso) -> bool
            {
                return pCso && (desc == pCso->getDesc());
            };

            if (mpCsoGraph->scanForMatchingNode(cmpFunc))
            {
                pCso = mpCsoGraph->getCurrentNode();
            }
            else
            {
                pCso = ComputeStateObject::create(mDesc);
                mpCsoGraph->setCurrentNodeData(pCso);
            }
        }

        return pCso;
    }

    FALCOR_SCRIPT_BINDING(ComputeState)
    {
        pybind11::class_<ComputeState, ComputeState::SharedPtr>(m, "ComputeState");
    }
}
