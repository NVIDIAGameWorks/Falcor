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
#include "ProgramManager.h"
#include "Core/API/Device.h"
#include "Core/Platform/OS.h"
#include "Utils/Logger.h"
#include "Utils/Timing/CpuTimer.h"

#include <slang.h>

namespace Falcor
{

inline SlangStage getSlangStage(ShaderType type)
{
    switch (type)
    {
    case ShaderType::Vertex:
        return SLANG_STAGE_VERTEX;
    case ShaderType::Pixel:
        return SLANG_STAGE_PIXEL;
    case ShaderType::Geometry:
        return SLANG_STAGE_GEOMETRY;
    case ShaderType::Hull:
        return SLANG_STAGE_HULL;
    case ShaderType::Domain:
        return SLANG_STAGE_DOMAIN;
    case ShaderType::Compute:
        return SLANG_STAGE_COMPUTE;
    case ShaderType::RayGeneration:
        return SLANG_STAGE_RAY_GENERATION;
    case ShaderType::Intersection:
        return SLANG_STAGE_INTERSECTION;
    case ShaderType::AnyHit:
        return SLANG_STAGE_ANY_HIT;
    case ShaderType::ClosestHit:
        return SLANG_STAGE_CLOSEST_HIT;
    case ShaderType::Miss:
        return SLANG_STAGE_MISS;
    case ShaderType::Callable:
        return SLANG_STAGE_CALLABLE;
    default:
        FALCOR_UNREACHABLE();
        return SLANG_STAGE_NONE;
    }
}

inline std::string getSlangProfileString(ShaderModel shaderModel)
{
    return fmt::format("sm_{}_{}", getShaderModelMajorVersion(shaderModel), getShaderModelMinorVersion(shaderModel));
}

inline bool doSlangReflection(
    const ProgramVersion& programVersion,
    slang::IComponentType* pSlangGlobalScope,
    std::vector<Slang::ComPtr<slang::IComponentType>> pSlangLinkedEntryPoints,
    ref<const ProgramReflection>& pReflector,
    std::string& log
)
{
    auto pSlangGlobalScopeLayout = pSlangGlobalScope->getLayout();

    // TODO: actually need to reflect the entry point groups!

    std::vector<slang::EntryPointLayout*> pSlangEntryPointReflectors;

    for (auto pSlangLinkedEntryPoint : pSlangLinkedEntryPoints)
    {
        auto pSlangEntryPointLayout = pSlangLinkedEntryPoint->getLayout()->getEntryPointByIndex(0);
        pSlangEntryPointReflectors.push_back(pSlangEntryPointLayout);
    }

    pReflector = ProgramReflection::create(&programVersion, pSlangGlobalScopeLayout, pSlangEntryPointReflectors, log);

    return true;
}

ProgramManager::ProgramManager(Device* pDevice) : mpDevice(pDevice)
{
    // Set global shader defines
    DefineList globalDefines = {
        {"FALCOR_NVAPI_AVAILABLE", (FALCOR_NVAPI_AVAILABLE && mpDevice->getType() == Device::Type::D3D12) ? "1" : "0"},
#if FALCOR_NVAPI_AVAILABLE
        {"NV_SHADER_EXTN_SLOT", "u999"},
        {"__SHADER_TARGET_MAJOR", std::to_string(getShaderModelMajorVersion(mpDevice->getSupportedShaderModel()))},
        {"__SHADER_TARGET_MINOR", std::to_string(getShaderModelMinorVersion(mpDevice->getSupportedShaderModel()))},
#endif
    };

    addGlobalDefines(globalDefines);

}

ref<const ProgramVersion> ProgramManager::createProgramVersion(const Program& program, std::string& log) const
{
    CpuTimer timer;
    timer.update();

    auto pSlangRequest = createSlangCompileRequest(program);
    if (pSlangRequest == nullptr)
        return nullptr;

    SlangResult slangResult = spCompile(pSlangRequest);
    log += spGetDiagnosticOutput(pSlangRequest);
    if (SLANG_FAILED(slangResult))
    {
        spDestroyCompileRequest(pSlangRequest);
        return nullptr;
    }

    Slang::ComPtr<slang::IComponentType> pSlangGlobalScope;
    spCompileRequest_getProgram(pSlangRequest, pSlangGlobalScope.writeRef());

    Slang::ComPtr<slang::ISession> pSlangSession(pSlangGlobalScope->getSession());

    // Prepare entry points.
    std::vector<Slang::ComPtr<slang::IComponentType>> pSlangEntryPoints;
    for (const auto& entryPointGroup : program.mDesc.entryPointGroups)
    {
        for (const auto& entryPoint : entryPointGroup.entryPoints)
        {
            Slang::ComPtr<slang::IComponentType> pSlangEntryPoint;
            spCompileRequest_getEntryPoint(pSlangRequest, entryPoint.globalIndex, pSlangEntryPoint.writeRef());

            // Rename entry point in the generated code if the exported name differs from the source name.
            // This makes it possible to generate different specializations of the same source entry point,
            // for example by setting different type conformances.
            if (entryPoint.exportName != entryPoint.name)
            {
                Slang::ComPtr<slang::IComponentType> pRenamedEntryPoint;
                pSlangEntryPoint->renameEntryPoint(entryPoint.exportName.c_str(), pRenamedEntryPoint.writeRef());
                pSlangEntryPoints.push_back(pRenamedEntryPoint);
            }
            else
            {
                pSlangEntryPoints.push_back(pSlangEntryPoint);
            }
        }
    }

    // Extract list of files referenced, for dependency-tracking purposes.
    int depFileCount = spGetDependencyFileCount(pSlangRequest);
    for (int ii = 0; ii < depFileCount; ++ii)
    {
        std::string depFilePath = spGetDependencyFilePath(pSlangRequest, ii);
        if (std::filesystem::exists(depFilePath))
            program.mFileTimeMap[depFilePath] = getFileModifiedTime(depFilePath);
    }

    // Note: the `ProgramReflection` needs to be able to refer back to the
    // `ProgramVersion`, but the `ProgramVersion` can't be initialized
    // until we have its reflection. We cut that dependency knot by
    // creating an "empty" program first, and then initializing it
    // after the reflection is created.
    //
    // TODO: There is no meaningful semantic difference between `ProgramVersion`
    // and `ProgramReflection`: they are one-to-one. Ideally in a future version
    // of Falcor they could be the same object.
    //
    // TODO @skallweit remove const cast
    ref<ProgramVersion> pVersion = ProgramVersion::createEmpty(const_cast<Program*>(&program), pSlangGlobalScope);

    // Note: Because of interactions between how `SV_Target` outputs
    // and `u` register bindings work in Slang today (as a compatibility
    // feature for Shader Model 5.0 and below), we need to make sure
    // that the entry points are included in the component type we use
    // for reflection.
    //
    // TODO: Once Slang drops that behavior for SM 5.1+, we should be able
    // to just use `pSlangGlobalScope` for the reflection step instead
    // of `pSlangProgram`.
    //
    Slang::ComPtr<slang::IComponentType> pSlangProgram;
    spCompileRequest_getProgram(pSlangRequest, pSlangProgram.writeRef());

    ref<const ProgramReflection> pReflector;
    if (!doSlangReflection(*pVersion, pSlangGlobalScope, pSlangEntryPoints, pReflector, log))
    {
        return nullptr;
    }

    auto descStr = program.getProgramDescString();
    pVersion->init(program.getDefineList(), pReflector, descStr, pSlangEntryPoints);

    timer.update();
    double time = timer.delta();
    mCompilationStats.programVersionCount++;
    mCompilationStats.programVersionTotalTime += time;
    mCompilationStats.programVersionMaxTime = std::max(mCompilationStats.programVersionMaxTime, time);
    logDebug("Created program version in {:.3f} s: {}", timer.delta(), descStr);

    return pVersion;
}

ref<const ProgramKernels> ProgramManager::createProgramKernels(
    const Program& program,
    const ProgramVersion& programVersion,
    const ProgramVars& programVars,
    std::string& log
) const
{
    CpuTimer timer;
    timer.update();

    auto pSlangGlobalScope = programVersion.getSlangGlobalScope();
    auto pSlangSession = pSlangGlobalScope->getSession();

    slang::IComponentType* pSpecializedSlangGlobalScope = pSlangGlobalScope;

    // Create a composite component type that represents all type conformances
    // linked into the `ProgramVersion`.
    auto createTypeConformanceComponentList = [&](const TypeConformanceList& typeConformances
                                              ) -> std::optional<Slang::ComPtr<slang::IComponentType>>
    {
        Slang::ComPtr<slang::IComponentType> pTypeConformancesCompositeComponent;
        std::vector<Slang::ComPtr<slang::ITypeConformance>> typeConformanceComponentList;
        std::vector<slang::IComponentType*> typeConformanceComponentRawPtrList;

        for (auto& typeConformance : typeConformances)
        {
            Slang::ComPtr<slang::IBlob> pSlangDiagnostics;
            Slang::ComPtr<slang::ITypeConformance> pTypeConformanceComponent;

            // Look for the type and interface type specified by the type conformance.
            // If not found we'll log an error and return.
            auto slangType = pSlangGlobalScope->getLayout()->findTypeByName(typeConformance.first.typeName.c_str());
            auto slangInterfaceType = pSlangGlobalScope->getLayout()->findTypeByName(typeConformance.first.interfaceName.c_str());
            if (!slangType)
            {
                log += fmt::format("Type '{}' in type conformance was not found.\n", typeConformance.first.typeName.c_str());
                return {};
            }
            if (!slangInterfaceType)
            {
                log += fmt::format("Interface type '{}' in type conformance was not found.\n", typeConformance.first.interfaceName.c_str());
                return {};
            }

            auto res = pSlangSession->createTypeConformanceComponentType(
                slangType,
                slangInterfaceType,
                pTypeConformanceComponent.writeRef(),
                (SlangInt)typeConformance.second,
                pSlangDiagnostics.writeRef()
            );
            if (SLANG_FAILED(res))
            {
                log += "Slang call createTypeConformanceComponentType() failed.\n";
                return {};
            }
            if (pSlangDiagnostics && pSlangDiagnostics->getBufferSize() > 0)
            {
                log += (char const*)pSlangDiagnostics->getBufferPointer();
            }
            if (pTypeConformanceComponent)
            {
                typeConformanceComponentList.push_back(pTypeConformanceComponent);
                typeConformanceComponentRawPtrList.push_back(pTypeConformanceComponent.get());
            }
        }
        if (!typeConformanceComponentList.empty())
        {
            Slang::ComPtr<slang::IBlob> pSlangDiagnostics;
            auto res = pSlangSession->createCompositeComponentType(
                &typeConformanceComponentRawPtrList[0],
                (SlangInt)typeConformanceComponentRawPtrList.size(),
                pTypeConformancesCompositeComponent.writeRef(),
                pSlangDiagnostics.writeRef()
            );
            if (SLANG_FAILED(res))
            {
                log += "Slang call createCompositeComponentType() failed.\n";
                return {};
            }
        }
        return pTypeConformancesCompositeComponent;
    };

    // Create one composite component type for the type conformances of each entry point group.
    // The type conformances for each group is the combination of the global and group type conformances.
    std::vector<Slang::ComPtr<slang::IComponentType>> typeConformancesCompositeComponents;
    typeConformancesCompositeComponents.reserve(program.mDesc.entryPointGroups.size());
    for (const auto& group : program.mDesc.entryPointGroups)
    {
        TypeConformanceList typeConformances = program.mTypeConformanceList;
        typeConformances.add(group.typeConformances);
        if (auto typeConformanceComponentList = createTypeConformanceComponentList(typeConformances))
            typeConformancesCompositeComponents.emplace_back(*typeConformanceComponentList);
        else
            return nullptr;
    }

    std::vector<Slang::ComPtr<slang::IComponentType>> pTypeConformanceSpecializedEntryPoints;
    std::vector<slang::IComponentType*> pTypeConformanceSpecializedEntryPointsRawPtr;
    std::vector<Slang::ComPtr<slang::IComponentType>> pLinkedEntryPoints;

    // Create a `IComponentType` for each entry point.
    for (size_t groupIndex = 0; groupIndex < program.mDesc.entryPointGroups.size(); ++groupIndex)
    {
        const auto& entryPointGroup = program.mDesc.entryPointGroups[groupIndex];

        for (const auto& entryPoint : entryPointGroup.entryPoints)
        {
            auto pSlangEntryPoint = programVersion.getSlangEntryPoint(entryPoint.globalIndex);

            Slang::ComPtr<slang::IBlob> pSlangDiagnostics;

            Slang::ComPtr<slang::IComponentType> pTypeComformanceSpecializedEntryPoint;
            if (typeConformancesCompositeComponents[groupIndex])
            {
                slang::IComponentType* componentTypes[] = {pSlangEntryPoint, typeConformancesCompositeComponents[groupIndex]};
                auto res = pSlangSession->createCompositeComponentType(
                    componentTypes, 2, pTypeComformanceSpecializedEntryPoint.writeRef(), pSlangDiagnostics.writeRef()
                );
                if (SLANG_FAILED(res))
                {
                    log += "Slang call createCompositeComponentType() failed.\n";
                    return nullptr;
                }
            }
            else
            {
                pTypeComformanceSpecializedEntryPoint = pSlangEntryPoint;
            }
            pTypeConformanceSpecializedEntryPoints.push_back(pTypeComformanceSpecializedEntryPoint);
            pTypeConformanceSpecializedEntryPointsRawPtr.push_back(pTypeComformanceSpecializedEntryPoint.get());

            Slang::ComPtr<slang::IComponentType> pLinkedSlangEntryPoint;
            {
                slang::IComponentType* componentTypes[] = {pSpecializedSlangGlobalScope, pTypeComformanceSpecializedEntryPoint};

                auto res = pSlangSession->createCompositeComponentType(
                    componentTypes, 2, pLinkedSlangEntryPoint.writeRef(), pSlangDiagnostics.writeRef()
                );
                if (SLANG_FAILED(res))
                {
                    log += "Slang call createCompositeComponentType() failed.\n";
                    return nullptr;
                }
            }
            pLinkedEntryPoints.push_back(pLinkedSlangEntryPoint);
        }
    }

    // Once specialization and linking are completed we need to
    // re-run the reflection step.
    //
    // A key guarantee we get from Slang is that the relative
    // ordering of parameters at the global scope or within a
    // given entry-point group will not change, so that when
    // `ParameterBlock`s and their descriptor tables/sets are allocated
    // using the unspecialized `ProgramReflection`, they will still
    // be valid to bind to the specialized program.
    //
    // Still, the specialized reflector may differ from the
    // unspecialized reflector in a few key ways:
    //
    // * There may be additional registers/bindings allocated for
    //   the global scope to account for the data required by
    //   specialized shader parameters (e.g., now that we know
    //   an `IFoo` parameter should actually be a `Bar`, we need
    //   to allocate those `Bar` resources somewhere).
    //
    // * As a result of specialized global-scope parameters taking
    //   up additional bindings/registers, the bindings/registers
    //   allocated to entry points and entry-point groups may be
    //   shifted.
    //
    // Note: Because of interactions between how `SV_Target` outputs
    // and `u` register bindings work in Slang today (as a compatibility
    // feature for Shader Model 5.0 and below), we need to make sure
    // that the entry points are included in the component type we use
    // for reflection.
    //
    // TODO: Once the behavior is fixed in Slang for SM 5.1+, we can
    // eliminate this step and use `pSpecializedSlangGlobalScope` instead
    // of `pSpecializedSlangProgram`, so long as we are okay with dropping
    // support for SM5.0 and below.
    //
    Slang::ComPtr<slang::IComponentType> pSpecializedSlangProgram;
    {
        // We are going to compose the global scope (specialized) with
        // all the entry points. Note that we do *not* use the "linked"
        // versions of the entry points because those already incorporate
        // the global scope, and we'd end up with multiple copies of
        // the global scope in that case.
        //
        std::vector<slang::IComponentType*> componentTypesForProgram;
        componentTypesForProgram.push_back(pSpecializedSlangGlobalScope);

        // TODO: Eventually this would need to use the specialized
        // (but not linked) version of each entry point.
        //
        componentTypesForProgram.insert(
            componentTypesForProgram.end(), programVersion.getSlangEntryPoints().begin(), programVersion.getSlangEntryPoints().end()
        );

        // Add type conformances for all entry point groups.
        // TODO: Is it correct to put all these in the global scope?
        for (auto pTypeConformancesComposite : typeConformancesCompositeComponents)
        {
            if (pTypeConformancesComposite)
            {
                componentTypesForProgram.push_back(pTypeConformancesComposite);
            }
        }

        auto res = pSlangSession->createCompositeComponentType(
            componentTypesForProgram.data(), componentTypesForProgram.size(), pSpecializedSlangProgram.writeRef()
        );
        if (SLANG_FAILED(res))
        {
            log += "Slang call createCompositeComponentType() failed.\n";
            return nullptr;
        }
    }

    ref<const ProgramReflection> pReflector;
    doSlangReflection(programVersion, pSpecializedSlangProgram, pLinkedEntryPoints, pReflector, log);

    // Create kernel objects for each entry point and cache them here.
    std::vector<ref<EntryPointKernel>> allKernels;
    for (const auto& entryPointGroup : program.mDesc.entryPointGroups)
    {
        for (const auto& entryPoint : entryPointGroup.entryPoints)
        {
            auto pLinkedEntryPoint = pLinkedEntryPoints[entryPoint.globalIndex];
            ref<EntryPointKernel> kernel = EntryPointKernel::create(pLinkedEntryPoint, entryPoint.type, entryPoint.exportName);
            if (!kernel)
                return nullptr;

            allKernels.push_back(std::move(kernel));
        }
    }

    // In order to construct the `ProgramKernels` we need to extract
    // the kernels for each entry-point group.
    //
    std::vector<ref<const EntryPointGroupKernels>> entryPointGroups;

    // TODO: Because we aren't actually specializing entry-point groups,
    // we will again loop over the original unspecialized entry point
    // groups from the `ProgramDesc`, and assume that they line up
    // one-to-one with the entries in `pLinkedEntryPointGroups`.
    //
    for (size_t groupIndex = 0; groupIndex < program.mDesc.entryPointGroups.size(); ++groupIndex)
    {
        const auto& entryPointGroup = program.mDesc.entryPointGroups[groupIndex];

        // For each entry-point group we will collect the compiled kernel
        // code for its constituent entry points, using the "linked"
        // version of the entry-point group.
        //
        std::vector<ref<EntryPointKernel>> kernels;
        for (const auto& entryPoint : entryPointGroup.entryPoints)
        {
            kernels.push_back(allKernels[entryPoint.globalIndex]);
        }
        auto pGroupReflector = pReflector->getEntryPointGroup(groupIndex);
        auto pEntryPointGroupKernels = createEntryPointGroupKernels(kernels, pGroupReflector);
        entryPointGroups.push_back(pEntryPointGroupKernels);
    }

    auto descStr = program.getProgramDescString();
    ref<const ProgramKernels> pProgramKernels = ProgramKernels::create(
        mpDevice,
        &programVersion,
        pSpecializedSlangGlobalScope,
        pTypeConformanceSpecializedEntryPointsRawPtr,
        pReflector,
        entryPointGroups,
        log,
        descStr
    );

    timer.update();
    double time = timer.delta();
    mCompilationStats.programKernelsCount++;
    mCompilationStats.programKernelsTotalTime += time;
    mCompilationStats.programKernelsMaxTime = std::max(mCompilationStats.programKernelsMaxTime, time);
    logDebug("Created program kernels in {:.3f} s: {}", time, descStr);

    return pProgramKernels;
}

ref<const EntryPointGroupKernels> ProgramManager::createEntryPointGroupKernels(
    const std::vector<ref<EntryPointKernel>>& kernels,
    const ref<EntryPointBaseReflection>& pReflector
) const
{
    FALCOR_ASSERT(kernels.size() != 0);

    switch (kernels[0]->getType())
    {
    case ShaderType::Vertex:
    case ShaderType::Pixel:
    case ShaderType::Geometry:
    case ShaderType::Hull:
    case ShaderType::Domain:
        return EntryPointGroupKernels::create(EntryPointGroupKernels::Type::Rasterization, kernels, kernels[0]->getEntryPointName());
    case ShaderType::Compute:
        return EntryPointGroupKernels::create(EntryPointGroupKernels::Type::Compute, kernels, kernels[0]->getEntryPointName());
    case ShaderType::AnyHit:
    case ShaderType::ClosestHit:
    case ShaderType::Intersection:
    {
        if (pReflector->getResourceRangeCount() > 0 || pReflector->getRootDescriptorRangeCount() > 0 ||
            pReflector->getParameterBlockSubObjectRangeCount() > 0)
        {
            FALCOR_THROW("Local root signatures are not supported for raytracing entry points.");
        }
        std::string exportName = fmt::format("HitGroup{}", mHitGroupID++);
        return EntryPointGroupKernels::create(EntryPointGroupKernels::Type::RtHitGroup, kernels, exportName);
    }
    case ShaderType::RayGeneration:
    case ShaderType::Miss:
    case ShaderType::Callable:
        return EntryPointGroupKernels::create(EntryPointGroupKernels::Type::RtSingleShader, kernels, kernels[0]->getEntryPointName());
    }

    FALCOR_UNREACHABLE();

    return nullptr;
}

std::string ProgramManager::getHlslLanguagePrelude() const
{
    Slang::ComPtr<ISlangBlob> prelude;
    mpDevice->getSlangGlobalSession()->getLanguagePrelude(SLANG_SOURCE_LANGUAGE_HLSL, prelude.writeRef());
    return std::string(reinterpret_cast<const char*>(prelude->getBufferPointer()), prelude->getBufferSize());
}

void ProgramManager::setHlslLanguagePrelude(const std::string& prelude)
{
    mpDevice->getSlangGlobalSession()->setLanguagePrelude(SLANG_SOURCE_LANGUAGE_HLSL, prelude.c_str());
}

void ProgramManager::registerProgramForReload(Program* program)
{
    mLoadedPrograms.push_back(program);
}

void ProgramManager::unregisterProgramForReload(Program* program)
{
    mLoadedPrograms.erase(std::remove(mLoadedPrograms.begin(), mLoadedPrograms.end(), program), mLoadedPrograms.end());
}

bool ProgramManager::reloadAllPrograms(bool forceReload)
{
    bool hasReloaded = false;

    for (auto program : mLoadedPrograms)
    {
        if (program->checkIfFilesChanged() || forceReload)
        {
            program->reset();
            hasReloaded = true;
        }
    }

    return hasReloaded;
}

void ProgramManager::addGlobalDefines(const DefineList& defineList)
{
    mGlobalDefineList.add(defineList);
    reloadAllPrograms(true);
}

void ProgramManager::removeGlobalDefines(const DefineList& defineList)
{
    mGlobalDefineList.remove(defineList);
    reloadAllPrograms(true);
}

void ProgramManager::setGenerateDebugInfoEnabled(bool enabled)
{
    mGenerateDebugInfo = enabled;
}

bool ProgramManager::isGenerateDebugInfoEnabled()
{
    return mGenerateDebugInfo;
}

void ProgramManager::setForcedCompilerFlags(ForcedCompilerFlags forcedCompilerFlags)
{
    mForcedCompilerFlags = forcedCompilerFlags;
    reloadAllPrograms(true);
}

ProgramManager::ForcedCompilerFlags ProgramManager::getForcedCompilerFlags()
{
    return mForcedCompilerFlags;
}

SlangCompileRequest* ProgramManager::createSlangCompileRequest(const Program& program) const
{
    slang::IGlobalSession* pSlangGlobalSession = mpDevice->getSlangGlobalSession();
    FALCOR_ASSERT(pSlangGlobalSession);

    slang::SessionDesc sessionDesc;

    // Add our shader search paths as `#include` search paths for Slang.
    //
    // Note: Slang allows application to plug in a callback API to
    // implement file I/O, and this could be used instead of specifying
    // the data directories to Slang.
    //
    std::vector<std::string> searchPaths;
    std::vector<const char*> slangSearchPaths;
    for (auto& path : getShaderDirectoriesList())
    {
        searchPaths.push_back(path.string());
        slangSearchPaths.push_back(searchPaths.back().data());
    }
    sessionDesc.searchPaths = slangSearchPaths.data();
    sessionDesc.searchPathCount = (SlangInt)slangSearchPaths.size();

    slang::TargetDesc targetDesc;
    targetDesc.format = SLANG_TARGET_UNKNOWN;
    targetDesc.profile = pSlangGlobalSession->findProfile(getSlangProfileString(program.mDesc.shaderModel).c_str());

    if (targetDesc.profile == SLANG_PROFILE_UNKNOWN)
        FALCOR_THROW("Can't find Slang profile for shader model {}", program.mDesc.shaderModel);

    // Get compiler flags and adjust with forced flags.
    SlangCompilerFlags compilerFlags = program.mDesc.compilerFlags;
    compilerFlags &= ~mForcedCompilerFlags.disabled;
    compilerFlags |= mForcedCompilerFlags.enabled;

    // Set floating point mode. If no shader compiler flags for this were set, we use Slang's default mode.
    bool flagFast = is_set(compilerFlags, SlangCompilerFlags::FloatingPointModeFast);
    bool flagPrecise = is_set(compilerFlags, SlangCompilerFlags::FloatingPointModePrecise);
    if (flagFast && flagPrecise)
    {
        logWarning(
            "Shader compiler flags 'FloatingPointModeFast' and 'FloatingPointModePrecise' can't be used simultaneously. Ignoring "
            "'FloatingPointModeFast'."
        );
        flagFast = false;
    }

    SlangFloatingPointMode slangFpMode = SLANG_FLOATING_POINT_MODE_DEFAULT;
    if (flagFast)
        slangFpMode = SLANG_FLOATING_POINT_MODE_FAST;
    else if (flagPrecise)
        slangFpMode = SLANG_FLOATING_POINT_MODE_PRECISE;

    targetDesc.floatingPointMode = slangFpMode;

    targetDesc.forceGLSLScalarBufferLayout = true;

    if (getEnvironmentVariable("FALCOR_USE_SLANG_SPIRV_BACKEND") == "1")
    {
        targetDesc.flags |= SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;
    }

    const char* targetMacroName;

    // Pick the right target based on the current graphics API
    switch (mpDevice->getType())
    {
    case Device::Type::D3D12:
        targetDesc.format = SLANG_DXIL;
        targetMacroName = "FALCOR_D3D12";
        break;
    case Device::Type::Vulkan:
        targetDesc.format = SLANG_SPIRV;
        targetMacroName = "FALCOR_VULKAN";
        break;
    default:
        FALCOR_UNREACHABLE();
    }

    // Pass any `#define` flags along to Slang, since we aren't doing our
    // own preprocessing any more.
    //
    std::vector<slang::PreprocessorMacroDesc> slangDefines;
    const auto addSlangDefine = [&slangDefines](const char* name, const char* value) { slangDefines.push_back({name, value}); };

    // Add global followed by program specific defines.
    for (const auto& shaderDefine : mGlobalDefineList)
        addSlangDefine(shaderDefine.first.c_str(), shaderDefine.second.c_str());
    for (const auto& shaderDefine : program.getDefineList())
        addSlangDefine(shaderDefine.first.c_str(), shaderDefine.second.c_str());

    // Add a `#define`s based on the target and shader model.
    addSlangDefine(targetMacroName, "1");

    // Add a `#define` based on the shader model.
    std::string sm = fmt::format(
        "__SM_{}_{}__", getShaderModelMajorVersion(program.mDesc.shaderModel), getShaderModelMinorVersion(program.mDesc.shaderModel)
    );
    addSlangDefine(sm.c_str(), "1");

    sessionDesc.preprocessorMacros = slangDefines.data();
    sessionDesc.preprocessorMacroCount = (SlangInt)slangDefines.size();

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    // We always use row-major matrix layout in Falcor so by default that's what we pass to Slang
    // to allow it to compute correct reflection information. Slang then invokes the downstream compiler.
    // Column major option can be useful when compiling external shader sources that don't depend
    // on anything Falcor.
    bool useColumnMajor = is_set(compilerFlags, SlangCompilerFlags::MatrixLayoutColumnMajor);
    sessionDesc.defaultMatrixLayoutMode = useColumnMajor ? SLANG_MATRIX_LAYOUT_COLUMN_MAJOR : SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    Slang::ComPtr<slang::ISession> pSlangSession;
    pSlangGlobalSession->createSession(sessionDesc, pSlangSession.writeRef());
    FALCOR_ASSERT(pSlangSession);

    program.mFileTimeMap.clear(); // TODO @skallweit

    SlangCompileRequest* pSlangRequest = nullptr;
    pSlangSession->createCompileRequest(&pSlangRequest);
    FALCOR_ASSERT(pSlangRequest);

    // Disable noisy warnings enabled in newer slang versions.
    spOverrideDiagnosticSeverity(pSlangRequest, 15602, SLANG_SEVERITY_DISABLED); // #pragma once in modules
    spOverrideDiagnosticSeverity(pSlangRequest, 30081, SLANG_SEVERITY_DISABLED); // implicit conversion

    // Enable/disable intermediates dump
    bool dumpIR = is_set(program.mDesc.compilerFlags, SlangCompilerFlags::DumpIntermediates);
    spSetDumpIntermediates(pSlangRequest, dumpIR);

    // Set debug level
    if (mGenerateDebugInfo || is_set(program.mDesc.compilerFlags, SlangCompilerFlags::GenerateDebugInfo))
        spSetDebugInfoLevel(pSlangRequest, SLANG_DEBUG_INFO_LEVEL_STANDARD);

    // Configure any flags for the Slang compilation step
    SlangCompileFlags slangFlags = 0;

    // When we invoke the Slang compiler front-end, skip code generation step
    // so that the compiler does not complain about missing arguments for
    // specialization parameters.
    //
    slangFlags |= SLANG_COMPILE_FLAG_NO_CODEGEN;

    spSetCompileFlags(pSlangRequest, slangFlags);

    // Set additional command line arguments.
    {
        std::vector<const char*> args;
        for (const auto& arg : mGlobalCompilerArguments)
            args.push_back(arg.c_str());
        for (const auto& arg : program.mDesc.compilerArguments)
            args.push_back(arg.c_str());
#if FALCOR_NVAPI_AVAILABLE
        // If NVAPI is available, we need to inform slang/dxc where to find it.
        std::string nvapiInclude = "-I" + (getRuntimeDirectory() / "shaders/nvapi").string();
        args.push_back("-Xdxc");
        args.push_back(nvapiInclude.c_str());
#endif
        if (!args.empty())
            spProcessCommandLineArguments(pSlangRequest, args.data(), (int)args.size());
    }

    for (size_t moduleIndex = 0; moduleIndex < program.mDesc.shaderModules.size(); ++moduleIndex)
    {
        const auto& module = program.mDesc.shaderModules[moduleIndex];
        // If module name is empty, pass in nullptr to let Slang generate a name internally.
        const char* name = !module.name.empty() ? module.name.c_str() : nullptr;
        int translationUnitIndex = spAddTranslationUnit(pSlangRequest, SLANG_SOURCE_LANGUAGE_SLANG, name);
        FALCOR_ASSERT(translationUnitIndex == moduleIndex);

        for (const auto& source : module.sources)
        {
            // Add source code to the translation unit
            if (source.type == ProgramDesc::ShaderSource::Type::File)
            {
                // If this is not an HLSL or a SLANG file, display a warning
                const auto& path = source.path;
                if (!(hasExtension(path, "hlsl") || hasExtension(path, "slang")))
                {
                    logWarning(
                        "Compiling a shader file which is not a SLANG file or an HLSL file. This is not an error, but make sure that the "
                        "file contains valid shaders"
                    );
                }
                std::filesystem::path fullPath;
                if (!findFileInShaderDirectories(path, fullPath))
                {
                    spDestroyCompileRequest(pSlangRequest);
                    FALCOR_THROW("Can't find shader file {}", path);
                }
                spAddTranslationUnitSourceFile(pSlangRequest, translationUnitIndex, fullPath.string().c_str());
            }
            else
            {
                FALCOR_ASSERT(source.type == ProgramDesc::ShaderSource::Type::String);
                spAddTranslationUnitSourceString(pSlangRequest, translationUnitIndex, source.path.string().c_str(), source.string.c_str());
            }
        }
    }

    // Now we make a separate pass and add the entry points.
    // Each entry point references the index of the source
    // it uses, and luckily, the Slang API can use these
    // indices directly.
    for (const auto& entryPointGroup : program.mDesc.entryPointGroups)
    {
        for (const auto& entryPoint : entryPointGroup.entryPoints)
        {
            spAddEntryPoint(pSlangRequest, entryPointGroup.shaderModuleIndex, entryPoint.name.c_str(), getSlangStage(entryPoint.type));
        }
    }

    return pSlangRequest;
}

} // namespace Falcor
