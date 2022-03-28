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
#include "stdafx.h"
#include "ComputeProgram.h"

namespace Falcor
{
    ComputeProgram::SharedPtr ComputeProgram::createFromFile(const std::filesystem::path& path, const std::string& csEntry, const DefineList& programDefines, Shader::CompilerFlags flags, const std::string& shaderModel)
    {
        Desc d(path);
        if (!shaderModel.empty()) d.setShaderModel(shaderModel);
        d.setCompilerFlags(flags);
        d.csEntry(csEntry);
        return create(d, programDefines);
    }

    ComputeProgram::SharedPtr ComputeProgram::create(const Desc& desc, const DefineList& programDefines)
    {
        auto pProg = SharedPtr(new ComputeProgram(desc, programDefines));
        registerProgramForReload(pProg);
        return pProg;
    }

    ComputeProgram::ComputeProgram(const Desc& desc, const DefineList& programDefines)
        : Program(desc, programDefines)
    {
    }

    void ComputeProgram::dispatchCompute(
        ComputeContext* pContext,
        ComputeVars* pVars,
        uint3 const& threadGroupCount)
    {
        auto pState = ComputeState::create();
        pState->setProgram(std::static_pointer_cast<ComputeProgram>(shared_from_this()));
        pContext->dispatch(pState.get(), pVars, threadGroupCount);
    }

    FALCOR_SCRIPT_BINDING(ComputeProgram)
    {
        pybind11::class_<ComputeProgram, ComputeProgram::SharedPtr>(m, "ComputeProgram");
    }
}
