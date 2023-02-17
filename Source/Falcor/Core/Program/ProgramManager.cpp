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

inline std::string getSlangProfileString(const std::string& shaderModel)
{
    return "sm_" + shaderModel;
}

inline bool doSlangReflection(
    const ProgramVersion& programVersion,
    slang::IComponentType* pSlangGlobalScope,
    std::vector<ComPtr<slang::IComponentType>> pSlangLinkedEntryPoints,
    ProgramReflection::SharedPtr& pReflector,
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

ProgramManager::ProgramManager(std::weak_ptr<Device> pDevice) : mpDevice(pDevice) {}

ProgramVersion::SharedPtr ProgramManager::createProgramVersion(const Program& program, std::string& log) const
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

    ComPtr<slang::IComponentType> pSlangGlobalScope;
    spCompileRequest_getProgram(pSlangRequest, pSlangGlobalScope.writeRef());

    ComPtr<slang::ISession> pSlangSession(pSlangGlobalScope->getSession());

    // Prepare entry points.
    std::vector<ComPtr<slang::IComponentType>> pSlangEntryPoints;
    uint32_t entryPointCount = (uint32_t)program.mDesc.mEntryPoints.size();
    for (uint32_t ee = 0; ee < entryPointCount; ++ee)
    {
        ComPtr<slang::IComponentType> pSlangEntryPoint;
        spCompileRequest_getEntryPoint(pSlangRequest, ee, pSlangEntryPoint.writeRef());

        // Rename entry point in the generated code if the exported name differs from the source name.
        // This makes it possible to generate different specializations of the same source entry point,
        // for example by setting different type conformances.
        const auto& entryPointDesc = program.mDesc.mEntryPoints[ee];
        if (entryPointDesc.exportName != entryPointDesc.name)
        {
            ComPtr<slang::IComponentType> pRenamedEntryPoint;
            pSlangEntryPoint->renameEntryPoint(entryPointDesc.exportName.c_str(), pRenamedEntryPoint.writeRef());
            pSlangEntryPoints.push_back(pRenamedEntryPoint);
        }
        else
        {
            pSlangEntryPoints.push_back(pSlangEntryPoint);
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
    ProgramVersion::SharedPtr pVersion = ProgramVersion::createEmpty(const_cast<Program*>(&program), pSlangGlobalScope);

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
    ComPtr<slang::IComponentType> pSlangProgram;
    spCompileRequest_getProgram(pSlangRequest, pSlangProgram.writeRef());

    ProgramReflection::SharedPtr pReflector;
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

ProgramKernels::SharedPtr ProgramManager::createProgramKernels(
    const Program& program,
    const ProgramVersion& programVersion,
    const ProgramVars& programVars,
    std::string& log
) const
{
    Device* pDevice = mpDevice.lock().get();
    FALCOR_ASSERT(pDevice);

    CpuTimer timer;
    timer.update();

    auto pSlangGlobalScope = programVersion.getSlangGlobalScope();
    auto pSlangSession = pSlangGlobalScope->getSession();

    slang::IComponentType* pSpecializedSlangGlobalScope = pSlangGlobalScope;

    // Create a composite component type that represents all type conformances
    // linked into the `ProgramVersion`.
    auto createTypeConformanceComponentList = [&](const Program::TypeConformanceList& typeConformances
                                              ) -> std::optional<ComPtr<slang::IComponentType>>
    {
        ComPtr<slang::IComponentType> pTypeConformancesCompositeComponent;
        std::vector<ComPtr<slang::ITypeConformance>> typeConformanceComponentList;
        std::vector<slang::IComponentType*> typeConformanceComponentRawPtrList;

        for (auto& typeConformance : typeConformances)
        {
            ComPtr<slang::IBlob> pSlangDiagnostics;
            ComPtr<slang::ITypeConformance> pTypeConformanceComponent;

            // Look for the type and interface type specified by the type conformance.
            // If not found we'll log an error and return.
            auto slangType = pSlangGlobalScope->getLayout()->findTypeByName(typeConformance.first.mTypeName.c_str());
            auto slangInterfaceType = pSlangGlobalScope->getLayout()->findTypeByName(typeConformance.first.mInterfaceName.c_str());
            if (!slangType)
            {
                log += fmt::format("Type '{}' in type conformance was not found.\n", typeConformance.first.mTypeName.c_str());
                return {};
            }
            if (!slangInterfaceType)
            {
                log +=
                    fmt::format("Interface type '{}' in type conformance was not found.\n", typeConformance.first.mInterfaceName.c_str());
                return {};
            }

            auto res = pSlangSession->createTypeConformanceComponentType(
                slangType, slangInterfaceType, pTypeConformanceComponent.writeRef(), (SlangInt)typeConformance.second,
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
            ComPtr<slang::IBlob> pSlangDiagnostics;
            auto res = pSlangSession->createCompositeComponentType(
                &typeConformanceComponentRawPtrList[0], (SlangInt)typeConformanceComponentRawPtrList.size(),
                pTypeConformancesCompositeComponent.writeRef(), pSlangDiagnostics.writeRef()
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
    std::vector<ComPtr<slang::IComponentType>> typeConformancesCompositeComponents;
    typeConformancesCompositeComponents.reserve(program.getEntryPointGroupCount());
    for (const auto& group : program.mDesc.mGroups)
    {
        Program::TypeConformanceList typeConformances = program.mTypeConformanceList;
        typeConformances.add(group.typeConformances);
        if (auto typeConformanceComponentList = createTypeConformanceComponentList(typeConformances))
            typeConformancesCompositeComponents.emplace_back(*typeConformanceComponentList);
        else
            return nullptr;
    }

    // Create a `IComponentType` for each entry point.
    uint32_t allEntryPointCount = uint32_t(program.mDesc.mEntryPoints.size());

    std::vector<ComPtr<slang::IComponentType>> pTypeConformanceSpecializedEntryPoints;
    std::vector<slang::IComponentType*> pTypeConformanceSpecializedEntryPointsRawPtr;
    std::vector<ComPtr<slang::IComponentType>> pLinkedEntryPoints;

    for (uint32_t ee = 0; ee < allEntryPointCount; ++ee)
    {
        auto pSlangEntryPoint = programVersion.getSlangEntryPoint(ee);

        int32_t groupIndex = program.mDesc.mEntryPoints[ee].groupIndex;
        FALCOR_ASSERT(groupIndex >= 0 && groupIndex < typeConformancesCompositeComponents.size());

        ComPtr<slang::IBlob> pSlangDiagnostics;

        ComPtr<slang::IComponentType> pTypeComformanceSpecializedEntryPoint;
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

        ComPtr<slang::IComponentType> pLinkedSlangEntryPoint;
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
    ComPtr<slang::IComponentType> pSpecializedSlangProgram;
    {
        // We are going to compose the global scope (specialized) with
        // all the entry points. Note that we do *not* use the "linked"
        // versions of the entry points because those already incorporate
        // the global scope, and we'd end up with multiple copies of
        // the global scope in that case.
        //
        std::vector<slang::IComponentType*> componentTypesForProgram;
        componentTypesForProgram.push_back(pSpecializedSlangGlobalScope);

        for (uint32_t ee = 0; ee < allEntryPointCount; ++ee)
        {
            // TODO: Eventually this would need to use the specialized
            // (but not linked) version of each entry point.
            //
            auto pSlangEntryPoint = programVersion.getSlangEntryPoint(ee);
            componentTypesForProgram.push_back(pSlangEntryPoint);
        }

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

    ProgramReflection::SharedPtr pReflector;
    doSlangReflection(programVersion, pSpecializedSlangProgram, pLinkedEntryPoints, pReflector, log);

    // Create Shader objects for each entry point and cache them here.
    std::vector<Shader::SharedPtr> allShaders;
    for (uint32_t i = 0; i < allEntryPointCount; i++)
    {
        auto pLinkedEntryPoint = pLinkedEntryPoints[i];
        auto entryPointDesc = program.mDesc.mEntryPoints[i];

        Shader::SharedPtr shader =
            Shader::create(pLinkedEntryPoint, entryPointDesc.stage, entryPointDesc.exportName, program.mDesc.getCompilerFlags(), log);
        if (!shader)
            return nullptr;

        allShaders.push_back(std::move(shader));
    }

    // In order to construct the `ProgramKernels` we need to extract
    // the kernels for each entry-point group.
    //
    std::vector<EntryPointGroupKernels::SharedPtr> entryPointGroups;

    // TODO: Because we aren't actually specializing entry-point groups,
    // we will again loop over the original unspecialized entry point
    // groups from the `Program::Desc`, and assume that they line up
    // one-to-one with the entries in `pLinkedEntryPointGroups`.
    //
    uint32_t entryPointGroupCount = uint32_t(program.mDesc.mGroups.size());
    for (uint32_t gg = 0; gg < entryPointGroupCount; ++gg)
    {
        auto entryPointGroupDesc = program.mDesc.mGroups[gg];
        // For each entry-point group we will collect the compiled kernel
        // code for its constituent entry points, using the "linked"
        // version of the entry-point group.
        //
        std::vector<Shader::SharedPtr> shaders;
        for (auto entryPointIndex : entryPointGroupDesc.entryPoints)
        {
            shaders.push_back(allShaders[entryPointIndex]);
        }
        auto pGroupReflector = pReflector->getEntryPointGroup(gg);
        auto pEntryPointGroupKernels = createEntryPointGroupKernels(shaders, pGroupReflector);
        entryPointGroups.push_back(pEntryPointGroupKernels);
    }

    auto descStr = program.getProgramDescString();
    ProgramKernels::SharedPtr pProgramKernels = ProgramKernels::create(
        pDevice, &programVersion, pSpecializedSlangGlobalScope, pTypeConformanceSpecializedEntryPointsRawPtr, pReflector, entryPointGroups,
        log, descStr
    );

    timer.update();
    double time = timer.delta();
    mCompilationStats.programKernelsCount++;
    mCompilationStats.programKernelsTotalTime += time;
    mCompilationStats.programKernelsMaxTime = std::max(mCompilationStats.programKernelsMaxTime, time);
    logDebug("Created program kernels in {:.3f} s: {}", time, descStr);

    return pProgramKernels;
}

EntryPointGroupKernels::SharedPtr ProgramManager::createEntryPointGroupKernels(
    const std::vector<Shader::SharedPtr>& shaders,
    EntryPointBaseReflection::SharedPtr const& pReflector
) const
{
    FALCOR_ASSERT(shaders.size() != 0);

    switch (shaders[0]->getType())
    {
    case ShaderType::Vertex:
    case ShaderType::Pixel:
    case ShaderType::Geometry:
    case ShaderType::Hull:
    case ShaderType::Domain:
        return EntryPointGroupKernels::create(EntryPointGroupKernels::Type::Rasterization, shaders, shaders[0]->getEntryPoint());
    case ShaderType::Compute:
        return EntryPointGroupKernels::create(EntryPointGroupKernels::Type::Compute, shaders, shaders[0]->getEntryPoint());
    case ShaderType::AnyHit:
    case ShaderType::ClosestHit:
    case ShaderType::Intersection:
    {
        if (pReflector->getResourceRangeCount() > 0 || pReflector->getRootDescriptorRangeCount() > 0 ||
            pReflector->getParameterBlockSubObjectRangeCount() > 0)
        {
            throw RuntimeError("Local root signatures are not supported for raytracing entry points.");
        }
        std::string exportName = fmt::format("HitGroup{}", mHitGroupID++);
        return EntryPointGroupKernels::create(EntryPointGroupKernels::Type::RtHitGroup, shaders, exportName);
    }
    case ShaderType::RayGeneration:
    case ShaderType::Miss:
    case ShaderType::Callable:
        return EntryPointGroupKernels::create(EntryPointGroupKernels::Type::RtSingleShader, shaders, shaders[0]->getEntryPoint());
    }

    FALCOR_UNREACHABLE();

    return nullptr;
}

void ProgramManager::registerProgramForReload(const Program::SharedPtr& pProg)
{
    mLoadedPrograms.push_back(pProg);
}

bool ProgramManager::reloadAllPrograms(bool forceReload)
{
    bool hasReloaded = false;

    // The `mLoadedPrograms` array stores weak pointers, and we will
    // use this step as a chance to clean up the contents of
    // the array that might have changed to `nullptr` because
    // the `Program` has been deleted.
    //
    // We will do this cleanup in a single pass without creating
    // a copy of the array by tracking two iterators: one for
    // reading and one for writing. The write iterator will
    // be explicit:
    //
    auto writeIter = mLoadedPrograms.begin();
    //
    // The read iterator will be implicit in our loop over the
    // entire array of programs:
    //
    for (auto& pWeakProgram : mLoadedPrograms)
    {
        // We will skip any programs where the weak pointer
        // has changed to `nullptr` because the object was
        // already deleted.
        //
        auto pProgram = pWeakProgram.lock();
        if (!pProgram)
            continue;

        // Now we know that we have a valid (non-null) `Program`,
        // so we wnat to keep it in the array for next time.
        //
        *writeIter++ = pProgram;

        // Next we check if any of the files that affected the
        // compilation of `pProgram` has been changed. If not,
        // we can skip further processing of this program
        // (unless forceReload flag is set).
        //
        if (!(pProgram->checkIfFilesChanged() || forceReload))
            continue;

        // If any files have changed, then we need to reset
        // the caches of compiled information for the program.
        //
        pProgram->reset();

        hasReloaded = true;
    }

    // Once we are done, we will have written a compacted
    // version of `mLoadedPrograms` (skipping the null elements)
    // to the first N elements of the vector. To make the
    // vector only contain those first N elements, we
    // then need to erase everything past the last point
    // we wrote to.
    //
    mLoadedPrograms.erase(writeIter, mLoadedPrograms.end());

    return hasReloaded;
}

void ProgramManager::addGlobalDefines(const Program::DefineList& defineList)
{
    mGlobalDefineList.add(defineList);
    reloadAllPrograms(true);
}

void ProgramManager::removeGlobalDefines(const Program::DefineList& defineList)
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
    auto pDevice = mpDevice.lock();
    FALCOR_ASSERT(pDevice);

    slang::IGlobalSession* pSlangGlobalSession = pDevice->getSlangGlobalSession();
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
    targetDesc.profile = pSlangGlobalSession->findProfile(getSlangProfileString(program.mDesc.mShaderModel).c_str());

    if (targetDesc.profile == SLANG_PROFILE_UNKNOWN)
    {
        reportError("Can't find Slang profile for shader model " + program.mDesc.mShaderModel);
        return nullptr;
    }

    // Get compiler flags and adjust with forced flags.
    Shader::CompilerFlags compilerFlags = program.mDesc.getCompilerFlags();
    compilerFlags &= ~mForcedCompilerFlags.disabled;
    compilerFlags |= mForcedCompilerFlags.enabled;

    // Set floating point mode. If no shader compiler flags for this were set, we use Slang's default mode.
    bool flagFast = is_set(compilerFlags, Shader::CompilerFlags::FloatingPointModeFast);
    bool flagPrecise = is_set(compilerFlags, Shader::CompilerFlags::FloatingPointModePrecise);
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

    const char* targetMacroName;

    // Pick the right target based on the current graphics API
    switch (pDevice->getType())
    {
    case Device::Type::D3D12:
        targetDesc.format = SLANG_DXIL;
        targetMacroName = "FALCOR_D3D";
        break;
    case Device::Type::Vulkan:
        targetDesc.format = SLANG_SPIRV;
        targetMacroName = "FALCOR_VK";
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

    std::string sm = "__SM_" + program.mDesc.mShaderModel + "__";
    addSlangDefine(sm.c_str(), "1");

    sessionDesc.preprocessorMacros = slangDefines.data();
    sessionDesc.preprocessorMacroCount = (SlangInt)slangDefines.size();

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    // We always use row-major matrix layout in Falcor so by default that's what we pass to Slang
    // to allow it to compute correct reflection information. Slang then invokes the downstream compiler.
    // Column major option can be useful when compiling external shader sources that don't depend
    // on anything Falcor.
    bool useColumnMajor = is_set(compilerFlags, Shader::CompilerFlags::MatrixLayoutColumnMajor);
    sessionDesc.defaultMatrixLayoutMode = useColumnMajor ? SLANG_MATRIX_LAYOUT_COLUMN_MAJOR : SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    ComPtr<slang::ISession> pSlangSession;
    pSlangGlobalSession->createSession(sessionDesc, pSlangSession.writeRef());
    FALCOR_ASSERT(pSlangSession);

    program.mFileTimeMap.clear(); // TODO @skallweit

    if (!program.mDesc.mLanguagePrelude.empty())
    {
        if (targetDesc.format == SLANG_DXIL)
        {
            pSlangGlobalSession->setLanguagePrelude(SLANG_SOURCE_LANGUAGE_HLSL, program.mDesc.mLanguagePrelude.c_str());
        }
        else
        {
            reportError("Language prelude set for unsupported target " + std::string(targetMacroName));
            return nullptr;
        }
    }

    SlangCompileRequest* pSlangRequest = nullptr;
    pSlangSession->createCompileRequest(&pSlangRequest);
    FALCOR_ASSERT(pSlangRequest);

    // Disable noisy warnings enabled in newer slang versions.
    spOverrideDiagnosticSeverity(pSlangRequest, 30081, SLANG_SEVERITY_DISABLED); // implicit conversion

    // Enable/disable intermediates dump
    bool dumpIR = is_set(program.mDesc.getCompilerFlags(), Shader::CompilerFlags::DumpIntermediates);
    spSetDumpIntermediates(pSlangRequest, dumpIR);

    // Set debug level
    if (mGenerateDebugInfo || is_set(program.mDesc.getCompilerFlags(), Shader::CompilerFlags::GenerateDebugInfo))
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
        for (const auto& arg : program.mDesc.mCompilerArguments)
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

    // Now lets add all our input shader code, one-by-one
    int translationUnitsAdded = 0;
    int translationUnitIndex = -1;

    for (const auto& src : program.mDesc.mSources)
    {
        // Register new translation unit with Slang if needed.
        if (translationUnitIndex < 0 || src.source.createTranslationUnit)
        {
            // If module name is empty, pass in nullptr to let Slang generate a name internally.
            const char* name = !src.source.moduleName.empty() ? src.source.moduleName.c_str() : nullptr;
            translationUnitIndex = spAddTranslationUnit(pSlangRequest, SLANG_SOURCE_LANGUAGE_SLANG, name);
            FALCOR_ASSERT(translationUnitIndex == translationUnitsAdded);
            translationUnitsAdded++;
        }
        FALCOR_ASSERT(translationUnitIndex >= 0);

        // Add source code to the translation unit
        if (src.getType() == Program::ShaderModule::Type::File)
        {
            // If this is not an HLSL or a SLANG file, display a warning
            const auto& path = src.source.filePath;
            if (!(hasExtension(path, "hlsl") || hasExtension(path, "slang")))
            {
                logWarning(
                    "Compiling a shader file which is not a SLANG file or an HLSL file. This is not an error, but make sure that the file "
                    "contains valid shaders"
                );
            }
            std::filesystem::path fullPath;
            if (!findFileInShaderDirectories(path, fullPath))
            {
                reportError("Can't find file " + path.string());
                spDestroyCompileRequest(pSlangRequest);
                return nullptr;
            }
            spAddTranslationUnitSourceFile(pSlangRequest, translationUnitIndex, fullPath.string().c_str());
        }
        else
        {
            FALCOR_ASSERT(src.getType() == Program::ShaderModule::Type::String);
            spAddTranslationUnitSourceString(pSlangRequest, translationUnitIndex, src.source.modulePath.c_str(), src.source.str.c_str());
        }
    }

    // Now we make a separate pass and add the entry points.
    // Each entry point references the index of the source
    // it uses, and luckily, the Slang API can use these
    // indices directly.
    for (auto& entryPoint : program.mDesc.mEntryPoints)
    {
        spAddEntryPoint(pSlangRequest, entryPoint.sourceIndex, entryPoint.name.c_str(), getSlangStage(entryPoint.stage));
    }

    return pSlangRequest;
}

} // namespace Falcor
