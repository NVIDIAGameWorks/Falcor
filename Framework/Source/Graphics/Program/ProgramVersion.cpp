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
#include "Graphics/Program/Program.h"
#include "Graphics/Program/ProgramVersion.h"
#include "ProgramVars.h"

namespace Falcor
{
    ProgramKernels::ProgramKernels(const ProgramReflection::SharedPtr& pReflector,  const Shader::SharedPtr& pVS, const Shader::SharedPtr& pPS, const Shader::SharedPtr& pGS, const Shader::SharedPtr& pHS, const Shader::SharedPtr& pDS, const Shader::SharedPtr& pCS, const RootSignature::SharedPtr& pRootSignature, const std::string& name) 
        : mName(name), mpReflector(pReflector)
    {
        mpShaders[(uint32_t)ShaderType::Vertex] = pVS;
        mpShaders[(uint32_t)ShaderType::Pixel] = pPS;
        mpShaders[(uint32_t)ShaderType::Geometry] = pGS;
        mpShaders[(uint32_t)ShaderType::Domain] = pDS;
        mpShaders[(uint32_t)ShaderType::Hull] = pHS;
        mpShaders[(uint32_t)ShaderType::Compute] = pCS;
        mpRootSignature = pRootSignature;
    }

    ProgramKernels::SharedPtr ProgramKernels::create(
        ProgramReflection::SharedPtr const& pReflector,
        const Shader::SharedPtr& pVS,
        const Shader::SharedPtr& pPS,
        const Shader::SharedPtr& pGS,
        const Shader::SharedPtr& pHS,
        const Shader::SharedPtr& pDS,
        const RootSignature::SharedPtr& pRootSignature,
        std::string& log,
        const std::string& name)
    {
        // We must have at least a VS.
        if(pVS == nullptr)
        {
            log = "Program " + name + " doesn't contain a vertex-shader. This is illegal.";
            return nullptr;
        }
        SharedPtr pProgram = SharedPtr(new ProgramKernels(pReflector, pVS, pPS, pGS, pHS, pDS, nullptr, pRootSignature, name));

        if(pProgram->init(log) == false)
        {
            return nullptr;
        }

        return pProgram;
    }

    ProgramKernels::SharedPtr ProgramKernels::create(
        const ProgramReflection::SharedPtr& pReflector,
        const Shader::SharedPtr& pCS,
        const RootSignature::SharedPtr& pRootSignature,
        std::string& log,
        const std::string& name)
    {
        // We must have at least a CS
        if (pCS == nullptr)
        {
            log = "Program " + name + " doesn't contain a compute-shader. This is illegal.";
            return nullptr;
        }
        SharedPtr pProgram = SharedPtr(new ProgramKernels(pReflector, nullptr, nullptr, nullptr, nullptr, nullptr, pCS, pRootSignature, name));

        if (pProgram->init(log) == false)
        {
            return nullptr;
        }
        return pProgram;
    }

    ProgramKernels::~ProgramKernels()
    {
        deleteApiHandle();
    }

    ProgramVersion::SharedPtr ProgramVersion::create(
        std::shared_ptr<Program>     const& pProgram,
        DefineList                   const& defines,
        ProgramReflection::SharedPtr const& pReflector,
        std::string                  const& name,
        SlangCompileRequest * compileReq)
    {
        return SharedPtr(new ProgramVersion(pProgram, defines, pReflector, name, compileReq));
    }

    ProgramVersion::ProgramVersion(
        std::shared_ptr<Program>     const& pProgram,
        DefineList                   const& defines,
        ProgramReflection::SharedPtr const& pReflector,
        std::string                  const& name,
        SlangCompileRequest*         compileReq)
        : mpProgram(pProgram)
        , mDefines(defines)
        , mpReflector(pReflector)
        , mName(name)
        , slangRequest(compileReq)
    {}

    ProgramKernels::SharedConstPtr ProgramVersion::getKernels(ProgramVars const* pVars, bool renameEntrypoint) const
    {
        // TODO: need a caching layer here, which takes into account:
        //
        // - The types of any shader components bound to `pVars`
        // - Any active `#define`s set on `pVars`
        //
        // For now we just cache one copy of things, since specialization
        // based on `ProgramVars` isn't implemented yet.
        //
        int programKey = 0;
        for (uint32_t i = 0; i < pVars->getParameterBlockCount(); i++)
        {
            programKey ^= pVars->getParameterBlock(i)->getTypeId();
            programKey <<= 8;
        }
        auto findRs = mpKernels.find(programKey);
        if( findRs != mpKernels.end() )
        {
            return findRs->second;
        }

        // Loop so that user can trigger recompilation on error
        for(;;)
        {
            std::string log;
            std::vector<std::string> newKernelNames;
            if (renameEntrypoint)
            {
                newKernelNames.resize((size_t)ShaderType::Count);
                for (size_t i = 0u; i < (size_t)ShaderType::Count; i++)
                {
                    auto entryPointName = mpProgram->mDesc.getShaderEntryPoint((ShaderType)i);
                    if (entryPointName.length())
                    {
                        newKernelNames[i] = entryPointName + std::to_string(programKey);
                    }
                }
            }
            
            auto kernels = mpProgram->preprocessAndCreateProgramKernels(this, pVars, newKernelNames, log);
            if( kernels )
            {
                // Success.
                mpKernels[programKey] = kernels;
                return kernels;
            }
            else
            {
                // Failure
                std::string error = std::string("Program Linkage failed.\n\n");
                error += getName() + "\n";
                error += log;
                 
                if(msgBox(error, MsgBoxType::RetryCancel) == MsgBoxButton::Cancel)
                {
                    // User has chosen not to retry
                    logError(error);
                    return nullptr;
                }

                // Continue loop to keep trying...
            }
        }
    }
}
