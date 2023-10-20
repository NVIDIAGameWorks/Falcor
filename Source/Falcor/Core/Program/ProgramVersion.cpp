/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "ProgramVersion.h"
#include "Program.h"
#include "ProgramManager.h"
#include "ProgramVars.h"
#include "Core/Error.h"
#include "Core/API/Device.h"
#include "Core/API/ParameterBlock.h"
#include "Utils/Logger.h"

#include <slang.h>

#include <set>

namespace Falcor
{

//
// EntryPointGroupKernels
//

ref<const EntryPointGroupKernels> EntryPointGroupKernels::create(
    EntryPointGroupKernels::Type type,
    const std::vector<ref<EntryPointKernel>>& kernels,
    const std::string& exportName
)
{
    return ref<EntryPointGroupKernels>(new EntryPointGroupKernels(type, kernels, exportName));
}

EntryPointGroupKernels::EntryPointGroupKernels(Type type, const std::vector<ref<EntryPointKernel>>& kernels, const std::string& exportName)
    : mType(type), mKernels(kernels), mExportName(exportName)
{}

const EntryPointKernel* EntryPointGroupKernels::getKernel(ShaderType type) const
{
    for (auto& pKernel : mKernels)
    {
        if (pKernel->getType() == type)
            return pKernel.get();
    }
    return nullptr;
}

//
// ProgramKernels
//

ProgramKernels::ProgramKernels(
    const ProgramVersion* pVersion,
    const ref<const ProgramReflection>& pReflector,
    const ProgramKernels::UniqueEntryPointGroups& uniqueEntryPointGroups,
    const std::string& name
)
    : mName(name), mUniqueEntryPointGroups(uniqueEntryPointGroups), mpReflector(pReflector), mpVersion(pVersion)
{}

ref<ProgramKernels> ProgramKernels::create(
    Device* pDevice,
    const ProgramVersion* pVersion,
    slang::IComponentType* pSpecializedSlangGlobalScope,
    const std::vector<slang::IComponentType*>& pTypeConformanceSpecializedEntryPoints,
    const ref<const ProgramReflection>& pReflector,
    const ProgramKernels::UniqueEntryPointGroups& uniqueEntryPointGroups,
    std::string& log,
    const std::string& name
)
{
    ref<ProgramKernels> pProgram = ref<ProgramKernels>(new ProgramKernels(pVersion, pReflector, uniqueEntryPointGroups, name));

    gfx::IShaderProgram::Desc programDesc = {};
    programDesc.linkingStyle = gfx::IShaderProgram::LinkingStyle::SeparateEntryPointCompilation;
    programDesc.slangGlobalScope = pSpecializedSlangGlobalScope;

    // Check if we are creating program kernels for ray tracing pipeline.
    bool isRayTracingProgram = false;
    if (pTypeConformanceSpecializedEntryPoints.size())
    {
        auto stage = pTypeConformanceSpecializedEntryPoints[0]->getLayout()->getEntryPointByIndex(0)->getStage();
        switch (stage)
        {
        case SLANG_STAGE_ANY_HIT:
        case SLANG_STAGE_RAY_GENERATION:
        case SLANG_STAGE_CLOSEST_HIT:
        case SLANG_STAGE_CALLABLE:
        case SLANG_STAGE_INTERSECTION:
        case SLANG_STAGE_MISS:
            isRayTracingProgram = true;
            break;
        default:
            break;
        }
    }
    // Deduplicate entry points by name for ray tracing program.
    std::vector<slang::IComponentType*> deduplicatedEntryPoints;
    if (isRayTracingProgram)
    {
        std::set<std::string> entryPointNames;
        for (auto entryPoint : pTypeConformanceSpecializedEntryPoints)
        {
            auto compiledEntryPointName = std::string(entryPoint->getLayout()->getEntryPointByIndex(0)->getNameOverride());
            if (entryPointNames.find(compiledEntryPointName) == entryPointNames.end())
            {
                entryPointNames.insert(compiledEntryPointName);
                deduplicatedEntryPoints.push_back(entryPoint);
            }
        }
        programDesc.entryPointCount = (uint32_t)deduplicatedEntryPoints.size();
        programDesc.slangEntryPoints = (slang::IComponentType**)deduplicatedEntryPoints.data();
    }
    else
    {
        programDesc.entryPointCount = (uint32_t)pTypeConformanceSpecializedEntryPoints.size();
        programDesc.slangEntryPoints = (slang::IComponentType**)pTypeConformanceSpecializedEntryPoints.data();
    }

    Slang::ComPtr<ISlangBlob> diagnostics;
    if (SLANG_FAILED(pDevice->getGfxDevice()->createProgram(programDesc, pProgram->mGfxProgram.writeRef(), diagnostics.writeRef())))
    {
        pProgram = nullptr;
    }
    if (diagnostics)
    {
        log = (const char*)diagnostics->getBufferPointer();
    }

    return pProgram;
}

const EntryPointKernel* ProgramKernels::getKernel(ShaderType type) const
{
    for (auto& pEntryPointGroup : mUniqueEntryPointGroups)
    {
        if (auto pShader = pEntryPointGroup->getKernel(type))
            return pShader;
    }
    return nullptr;
}

ProgramVersion::ProgramVersion(Program* pProgram, slang::IComponentType* pSlangGlobalScope)
    : mpProgram(pProgram), mpSlangGlobalScope(pSlangGlobalScope)
{
    FALCOR_ASSERT(pProgram);
}

void ProgramVersion::init(
    const DefineList& defineList,
    const ref<const ProgramReflection>& pReflector,
    const std::string& name,
    const std::vector<Slang::ComPtr<slang::IComponentType>>& pSlangEntryPoints
)
{
    FALCOR_ASSERT(pReflector);
    mDefines = defineList;
    mpReflector = pReflector;
    mName = name;
    mpSlangEntryPoints = pSlangEntryPoints;
}

ref<ProgramVersion> ProgramVersion::createEmpty(Program* pProgram, slang::IComponentType* pSlangGlobalScope)
{
    return ref<ProgramVersion>(new ProgramVersion(pProgram, pSlangGlobalScope));
}

ref<const ProgramKernels> ProgramVersion::getKernels(Device* pDevice, ProgramVars const* pVars) const
{
    // We need are going to look up or create specialized kernels
    // based on how parameters are bound in `pVars`.
    //
    // To do this we need to identify those parameters that are relevant
    // to specialization, and what argument type/value is bound to
    // those parameters.
    //
    std::string specializationKey;

    ParameterBlock::SpecializationArgs specializationArgs;
    if (pVars)
    {
        pVars->collectSpecializationArgs(specializationArgs);
    }

    bool first = true;
    for (auto specializationArg : specializationArgs)
    {
        if (!first)
            specializationKey += ",";
        specializationKey += std::string(specializationArg.type->getName());
        first = false;
    }

    auto foundKernels = mpKernels.find(specializationKey);
    if (foundKernels != mpKernels.end())
    {
        return foundKernels->second;
    }

    FALCOR_ASSERT(mpProgram);

    // Loop so that user can trigger recompilation on error
    for (;;)
    {
        std::string log;
        auto pKernels = pDevice->getProgramManager()->createProgramKernels(*mpProgram, *this, *pVars, log);
        if (pKernels)
        {
            // Success

            if (!log.empty())
            {
                logWarning("Warnings in program:\n{}\n{}", getName(), log);
            }

            mpKernels[specializationKey] = pKernels;
            return pKernels;
        }
        else
        {
            // Failure
            std::string msg = fmt::format("Failed to link program:\n{}\n\n{}", getName(), log);
            bool showMessageBox = is_set(getErrorDiagnosticFlags(), ErrorDiagnosticFlags::ShowMessageBoxOnError);
            if (showMessageBox && reportErrorAndAllowRetry(msg))
                continue;
            FALCOR_THROW(msg);
        }
    }
}

slang::ISession* ProgramVersion::getSlangSession() const
{
    return getSlangGlobalScope()->getSession();
}

slang::IComponentType* ProgramVersion::getSlangGlobalScope() const
{
    return mpSlangGlobalScope;
}

slang::IComponentType* ProgramVersion::getSlangEntryPoint(uint32_t index) const
{
    return mpSlangEntryPoints[index];
}
} // namespace Falcor
