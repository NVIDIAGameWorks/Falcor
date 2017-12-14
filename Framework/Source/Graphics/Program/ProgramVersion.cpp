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
#include "Graphics/Program/ProgramVersion.h"
#include "Graphics/Material/MaterialSystem.h"

namespace Falcor
{
    ProgramVersion::ProgramVersion(const Shader::SharedPtr& pVS, const Shader::SharedPtr& pPS, const Shader::SharedPtr& pGS, const Shader::SharedPtr& pHS, const Shader::SharedPtr& pDS, const Shader::SharedPtr& pCS, const std::string& name) : mName(name)
    {
        mpShaders[(uint32_t)ShaderType::Vertex] = pVS;
        mpShaders[(uint32_t)ShaderType::Pixel] = pPS;
        mpShaders[(uint32_t)ShaderType::Geometry] = pGS;
        mpShaders[(uint32_t)ShaderType::Domain] = pDS;
        mpShaders[(uint32_t)ShaderType::Hull] = pHS;
        mpShaders[(uint32_t)ShaderType::Compute] = pCS;
    }

    ProgramVersion::SharedPtr ProgramVersion::create(
        ProgramReflection::SharedPtr const& pReflector,
        const Shader::SharedPtr& pVS,
        const Shader::SharedPtr& pPS,
        const Shader::SharedPtr& pGS,
        const Shader::SharedPtr& pHS,
        const Shader::SharedPtr& pDS,
        std::string& log,
        const std::string& name)
    {
        // We must have at least a VS.
        if(pVS == nullptr)
        {
            log = "Program " + name + " doesn't contain a vertex-shader. This is illegal.";
            return nullptr;
        }
        SharedPtr pProgram = SharedPtr(new ProgramVersion(pVS, pPS, pGS, pHS, pDS, nullptr, name));

        if(pProgram->init(log) == false)
        {
            return nullptr;
        }

        pProgram->mpReflector = pReflector;
        if (pProgram->mpReflector == nullptr)
        {
            return nullptr;
        }
        return pProgram;
    }

    ProgramVersion::SharedPtr ProgramVersion::create(
        ProgramReflection::SharedPtr const& pReflector,
        const Shader::SharedPtr& pCS,
        std::string& log,
        const std::string& name)
    {
        // We must have at least a CS
        if (pCS == nullptr)
        {
            log = "Program " + name + " doesn't contain a compute-shader. This is illegal.";
            return nullptr;
        }
        SharedPtr pProgram = SharedPtr(new ProgramVersion(nullptr, nullptr, nullptr, nullptr, nullptr, pCS, name));

        if (pProgram->init(log) == false)
        {
            return nullptr;
        }
        pProgram->mpReflector = pReflector;
        if (pProgram->mpReflector == nullptr)
        {
            return nullptr;
        }
        return pProgram;
    }

    ProgramVersion::~ProgramVersion()
    {
        MaterialSystem::removeProgramVersion(this);
        deleteApiHandle();
    }
}
