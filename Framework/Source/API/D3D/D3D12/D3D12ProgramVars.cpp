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
#include "Framework.h"
#include "API/CopyContext.h"
#include "Graphics/Program/ProgramVars.h"

namespace Falcor
{
    template<bool forGraphics>
    void bindConstantBuffers(CopyContext* pContext, const ProgramVars::ResourceMap<ConstantBuffer>& cbMap, const ProgramVars::RootSetVec& rootSets, bool forceBind)
    {
        for (auto& bufIt : cbMap)
        {
            const auto& rootData = bufIt.second[0].rootData;
            assert(rootData.rangeIndex == 0);
            ConstantBuffer* pCB = dynamic_cast<ConstantBuffer*>(bufIt.second[0].pResource.get());

            if (rootSets[rootData.rootIndex].dirty || forceBind)
            {
                auto& pList = pContext->getLowLevelData()->getCommandList();

                if (forGraphics)
                {
                    pList->SetGraphicsRootConstantBufferView(rootData.rootIndex, pCB->getGpuAddress());
                }
                else
                {
                    pList->SetComputeRootConstantBufferView(rootData.rootIndex, pCB->getGpuAddress());
                }
            }
        }
    }

    template void bindConstantBuffers<true>(CopyContext* pContext, const ProgramVars::ResourceMap<ConstantBuffer>& cbMap, const ProgramVars::RootSetVec& rootSets, bool forceBind);
    template void bindConstantBuffers<false>(CopyContext* pContext, const ProgramVars::ResourceMap<ConstantBuffer>& cbMap, const ProgramVars::RootSetVec& rootSets, bool forceBind);
}
