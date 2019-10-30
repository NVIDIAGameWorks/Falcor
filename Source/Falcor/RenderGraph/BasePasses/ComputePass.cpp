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
#include "stdafx.h"
#include "ComputePass.h"

namespace Falcor
{
    ComputePass::ComputePass(const Program::Desc& desc, const Program::DefineList& defines, bool createVars)
    {
        auto pProg = ComputeProgram::create(desc, defines);
        mpState = ComputeState::create();
        mpState->setProgram(pProg);
        if (createVars) mpVars = ComputeVars::create(pProg.get());
        if ((createVars && !mpVars) || !mpState || !pProg) throw std::exception();
    }

    ComputePass::SharedPtr ComputePass::create(const std::string& csFile, const std::string& csEntry, const Program::DefineList& defines, bool createVars)
    {
        return create(Program::Desc().addShaderLibrary(csFile).csEntry(csEntry), defines, createVars);
    }

    ComputePass::SharedPtr ComputePass::create(const Program::Desc& desc, const Program::DefineList& defines, bool createVars)
    {
        try
        {
            return SharedPtr(new ComputePass(desc, defines, createVars));
        }
        catch (std::exception) { return nullptr; }
    }

    void ComputePass::execute(ComputeContext* pContext, uint32_t nThreadX, uint32_t nThreadY, uint32_t nThreadZ)
    {
        assert(mpVars);
        uvec3 threadGroupSize = mpState->getProgram()->getReflector()->getThreadGroupSize();
        uvec3 groups = div_round_up(glm::uvec3(nThreadX, nThreadY, nThreadZ), threadGroupSize);
        pContext->dispatch(mpState.get(), mpVars.get(), groups);
    }

    void ComputePass::addDefine(const std::string& name, const std::string& value, bool updateVars)
    {
        mpState->getProgram()->addDefine(name, value);
        if (updateVars) mpVars = ComputeVars::create(mpState->getProgram().get());
    }

    void ComputePass::removeDefine(const std::string& name, bool updateVars)
    {
        mpState->getProgram()->removeDefine(name);
        if (updateVars) mpVars = ComputeVars::create(mpState->getProgram().get());
    }

    void ComputePass::setVars(const ComputeVars::SharedPtr& pVars)
    {
        mpVars = pVars ? pVars : ComputeVars::create(mpState->getProgram().get());
    }
}
