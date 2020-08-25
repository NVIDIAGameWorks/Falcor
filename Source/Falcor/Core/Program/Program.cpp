/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Program.h"
#include "Slang/slang.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
#ifdef FALCOR_VK
    const std::string kSupportedShaderModels[] = { "400", "410", "420", "430", "440", "450" };
#elif defined FALCOR_D3D12
    const std::string kSupportedShaderModels[] = { "4_0", "4_1", "5_0", "5_1", "6_0", "6_1", "6_2", "6_3" };
#endif

    static Program::DefineList sGlobalDefineList;

    static Shader::SharedPtr createShaderFromBlob(const Shader::Blob& shaderBlob, ShaderType shaderType, const std::string& entryPointName, Shader::CompilerFlags flags, std::string& log)
    {
        std::string errorMsg;
        auto pShader = Shader::create(shaderBlob, shaderType, entryPointName, flags, log);
        return pShader;
    }

    Program::Desc::Desc() = default;

    Program::Desc::Desc(std::string const& path)
    {
        addShaderLibrary(path);
    }

    Program::Desc& Program::Desc::addShaderLibrary(std::string const& path)
    {
        Source source(ShaderLibrary::create(path));
        source.firstEntryPoint = uint32_t(mEntryPoints.size());

        mActiveSource = (int32_t) mSources.size();
        mSources.emplace_back(std::move(source));
        return *this;
    }

    Program::Desc& Program::Desc::addShaderString(const std::string& shader)
    {
        mActiveSource = (int32_t) mSources.size();
        mSources.emplace_back(shader);

        return *this;
    }

    Program::Desc& Program::Desc::beginEntryPointGroup()
    {
        EntryPointGroup group;
        group.firstEntryPoint = uint32_t(mEntryPoints.size());
        group.entryPointCount = 0;

        mActiveGroup = (int32_t) mGroups.size();
        mGroups.push_back(group);

        return *this;
    }

    Program::Desc& Program::Desc::entryPoint(ShaderType shaderType, std::string const& name)
    {
        if(name.size() == 0)
            return *this;

        if(mActiveSource < 0)
        {
            throw std::exception("Cannot add an entry point without first adding a source file/library");
        }

        if(mActiveGroup < 0)
        {
            beginEntryPointGroup();
        }

        EntryPoint entryPoint;
        entryPoint.stage = shaderType;
        entryPoint.name = name;

        entryPoint.sourceIndex = mActiveSource;
        entryPoint.groupIndex = mActiveGroup;

        mGroups[mActiveGroup].entryPointCount++;
        mSources[mActiveSource].entryPointCount++;

        mEntryPoints.push_back(entryPoint);

        return *this;
    }

    Program::Desc& Program::Desc::addDefaultVertexShaderIfNeeded()
    {
        // Don't set default vertex shader if one was set already.
        if(hasEntryPoint(ShaderType::Vertex))
        {
            return *this;
        }
        return addShaderLibrary("Scene/Raster.slang").entryPoint(ShaderType::Vertex, "defaultVS");
    }

    Program::Desc& Program::Desc::setShaderModel(const std::string& sm)
    {
        // Check that the model is supported
        bool b = false;
        for (size_t i = 0; i < arraysize(kSupportedShaderModels); i++)
        {
            if (kSupportedShaderModels[i] == sm)
            {
                b = true;
                break;
            }
        }

        if (b == false)
        {
            std::string warn = "Unsupported shader-model '" + sm + "' requested. Supported shader-models are ";
            for (size_t i = 0; i < arraysize(kSupportedShaderModels); i++)
            {
                warn += kSupportedShaderModels[i];
                warn += (i == kSupportedShaderModels->size() - 1) ? "." : ", ";
            }
            warn += "\nThis is not an error, but if something goes wrong try using one of the supported models.";
            logWarning(warn);
        }

        mShaderModel = sm;
        return *this;
    }

    bool Program::Desc::hasEntryPoint(ShaderType stage) const
    {
        for(auto& entryPoint : mEntryPoints)
        {
            if(entryPoint.stage == stage)
            {
                return true;
            }
        }
        return false;
    }

    // Program
    std::vector<std::weak_ptr<Program>> Program::sPrograms;

    void Program::init(Desc const& desc, DefineList const& defineList)
    {
        mDesc = desc;
        mDefineList = defineList;

        sPrograms.push_back(shared_from_this());
    }

    Program::~Program()
    {
    }

    std::string Program::getProgramDescString() const
    {
        std::string desc;

        int32_t groupCount = (int32_t) mDesc.mGroups.size();

        for(auto& src : mDesc.mSources)
        {
            switch (src.type)
            {
            case Desc::Source::Type::File:
                desc += src.pLibrary->getFilename();
                break;
            case Desc::Source::Type::String:
                desc += "Created from string";
                break;
            default:
                should_not_get_here();
            }

            uint32_t entryPointCount = src.entryPointCount;
            desc += "(";
            for( uint32_t ee = 0; ee < entryPointCount; ++ee )
            {
                auto& entryPoint = mDesc.mEntryPoints[src.firstEntryPoint + ee];

                if(ee != 0) desc += ", ";
                desc += entryPoint.name;
            }
            desc += ")";
        }

        return desc;
    }

    bool Program::addDefine(const std::string& name, const std::string& value)
    {
        // Make sure that it doesn't exist already
        if(mDefineList.find(name) != mDefineList.end())
        {
            if(mDefineList[name] == value)
            {
                // Same define
                return false;
            }
        }
        markDirty();
        mDefineList[name] = value;
        return true;
    }

    bool Program::addDefines(const DefineList& dl)
    {
        bool dirty = false;
        for (auto it : dl)
        {
            if (addDefine(it.first, it.second))
            {
                dirty = true;
            }
        }
        return dirty;
    }

    bool Program::removeDefine(const std::string& name)
    {
        if(mDefineList.find(name) != mDefineList.end())
        {
            markDirty();
            mDefineList.erase(name);
            return true;
        }
        return false;
    }

    bool Program::removeDefines(const DefineList& dl)
    {
        bool dirty = false;
        for (auto it : dl)
        {
            if (removeDefine(it.first))
            {
                dirty = true;
            }
        }
        return dirty;
    }

    bool Program::removeDefines(size_t pos, size_t len, const std::string& str)
    {
        bool dirty = false;
        for (auto it = mDefineList.cbegin(); it != mDefineList.cend();)
        {
            if (pos < it->first.length() && it->first.compare(pos, len, str) == 0)
            {
                markDirty();
                it = mDefineList.erase(it);
                dirty = true;
            }
            else
            {
                ++it;
            }
        }
        return dirty;
    }

    bool Program::setDefines(const DefineList& dl)
    {
        if (dl != mDefineList)
        {
            markDirty();
            mDefineList = dl;
            return true;
        }
        return false;
    }

    bool Program::checkIfFilesChanged()
    {
        if(mpActiveVersion == nullptr)
        {
            // We never linked, so nothing really changed
            return false;
        }

        // Have any of the files we depend on changed?
        for(auto& entry : mFileTimeMap)
        {
            auto& path = entry.first;
            auto& modifiedTime = entry.second;

            if( modifiedTime != getFileModifiedTime(path) )
            {
                return true;
            }
        }
        return false;
    }

    const ProgramVersion::SharedConstPtr& Program::getActiveVersion() const
    {
        if (mLinkRequired)
        {
            const auto& it = mProgramVersions.find(mDefineList);
            if (it == mProgramVersions.end())
            {
                // Note that link() updates mActiveProgram only if the operation was successful.
                // On error we get false, and mActiveProgram points to the last successfully compiled version.
                if (link() == false)
                {
                    throw std::exception("Program linkage failed");
                }
                else
                {
                    mProgramVersions[mDefineList] = mpActiveVersion;
                }
            }
            else
            {
                mpActiveVersion = it->second;
            }
            mLinkRequired = false;
        }
        assert(mpActiveVersion);
        return mpActiveVersion;
    }

    slang::IGlobalSession* createSlangGlobalSession()
    {
        slang::IGlobalSession* result = nullptr;
        slang::createGlobalSession(&result);
        return result;
    }

    slang::IGlobalSession* getSlangGlobalSession()
    {
        static slang::IGlobalSession* pSlangGlobalSession = createSlangGlobalSession();
        return pSlangGlobalSession;
    }

    // Translation a Falcor `ShaderType` to the corresponding `SlangStage`
    SlangStage getSlangStage(ShaderType type)
    {
        switch(type)
        {
        case ShaderType::Vertex:        return SLANG_STAGE_VERTEX;
        case ShaderType::Pixel:         return SLANG_STAGE_PIXEL;
        case ShaderType::Geometry:      return SLANG_STAGE_GEOMETRY;
        case ShaderType::Hull:          return SLANG_STAGE_HULL;
        case ShaderType::Domain:        return SLANG_STAGE_DOMAIN;
        case ShaderType::Compute:       return SLANG_STAGE_COMPUTE;
#ifdef FALCOR_D3D12
        case ShaderType::RayGeneration: return SLANG_STAGE_RAY_GENERATION;
        case ShaderType::Intersection:  return SLANG_STAGE_INTERSECTION;
        case ShaderType::AnyHit:        return SLANG_STAGE_ANY_HIT;
        case ShaderType::ClosestHit:    return SLANG_STAGE_CLOSEST_HIT;
        case ShaderType::Miss:          return SLANG_STAGE_MISS;
        case ShaderType::Callable:      return SLANG_STAGE_CALLABLE;
#endif
        default:
            should_not_get_here();
            return SLANG_STAGE_NONE;
        }
    }

    static std::string getSlangProfileString(const std::string& shaderModel)
    {
#if defined FALCOR_VK
        return "glsl_" + shaderModel;
#elif defined FALCOR_D3D12
        return "sm_" + shaderModel;
#else
#error unknown shader compilation target
#endif
    }

    SlangCompileRequest* Program::createSlangCompileRequest(
        const DefineList&   defineList) const
    {
        slang::IGlobalSession* pSlangGlobalSession = getSlangGlobalSession();
        assert(pSlangGlobalSession);

        slang::SessionDesc sessionDesc;

        // Add our media search paths as `#include` search paths for Slang.
        //
        // Note: Slang allows application to plug in a callback API to
        // implement file I/O, and this could be used instead of specifying
        // the data directories to Slang.
        //
        std::vector<const char*> slangSearchPaths;
        for (auto& path : getShaderDirectoriesList())
        {
            slangSearchPaths.push_back(path.c_str());
        }
        sessionDesc.searchPaths = slangSearchPaths.data();
        sessionDesc.searchPathCount = (SlangInt) slangSearchPaths.size();

        slang::TargetDesc targetDesc;
        targetDesc.format = SLANG_TARGET_UNKNOWN;
        targetDesc.profile = pSlangGlobalSession->findProfile(getSlangProfileString(mDesc.mShaderModel).c_str());

        if (targetDesc.profile == SLANG_PROFILE_UNKNOWN)
        {
            logError("Can't find Slang profile for shader model " + mDesc.mShaderModel);
            return nullptr;
        }

        // Set floating point mode. If no shader compiler flags for this were set, we use Slang's default mode.
        bool flagFast = is_set(mDesc.getCompilerFlags(), Shader::CompilerFlags::FloatingPointModeFast);
        bool flagPrecise = is_set(mDesc.getCompilerFlags(), Shader::CompilerFlags::FloatingPointModePrecise);
        if (flagFast && flagPrecise)
        {
            logWarning("Shader compiler flags 'FloatingPointModeFast' and 'FloatingPointModePrecise' can't be used simultaneously. Ignoring 'FloatingPointModeFast'.");
            flagFast = false;
        }

        SlangFloatingPointMode slangFpMode = SLANG_FLOATING_POINT_MODE_DEFAULT;
        if (flagFast) slangFpMode = SLANG_FLOATING_POINT_MODE_FAST;
        else if (flagPrecise) slangFpMode = SLANG_FLOATING_POINT_MODE_PRECISE;

        targetDesc.floatingPointMode = slangFpMode;

        const char* targetMacroName;

        // Pick the right target based on the current graphics API
#ifdef FALCOR_VK
        targetMacroName = "FALCOR_VK";
        targetDesc.format = SLANG_SPIRV;
#elif defined FALCOR_D3D12
        targetMacroName = "FALCOR_D3D";

        // If the profile string starts with a `4_` or a `5_`, use DXBC. Otherwise, use DXIL
        if (hasPrefix(mDesc.mShaderModel, "4_") || hasPrefix(mDesc.mShaderModel, "5_")) targetDesc.format = SLANG_DXBC;
        else                                                                            targetDesc.format = SLANG_DXIL;
#else
#error unknown shader compilation target
#endif


        // Pass any `#define` flags along to Slang, since we aren't doing our
        // own preprocessing any more.
        //
        std::vector<slang::PreprocessorMacroDesc> slangDefines;
        const auto addSlangDefine = [&slangDefines] (const char* name, const char* value)
        {
            slangDefines.push_back({ name, value });
        };

        // Add global defines.
        for (const auto& shaderDefine : sGlobalDefineList)
        {
            addSlangDefine(shaderDefine.first.c_str(), shaderDefine.second.c_str());
        }

        // Add program specific defines.
        for (const auto& shaderDefine : getDefineList())
        {
            addSlangDefine(shaderDefine.first.c_str(), shaderDefine.second.c_str());
        }

        // Add a `#define`s based on the target and shader model.
        addSlangDefine(targetMacroName, "1");

        std::string sm = "__SM_" + mDesc.mShaderModel + "__";
        addSlangDefine(sm.c_str(), "1");

        sessionDesc.preprocessorMacros = slangDefines.data();
        sessionDesc.preprocessorMacroCount = (SlangInt) slangDefines.size();

        sessionDesc.targets = &targetDesc;
        sessionDesc.targetCount = 1;

        // We always use row-major matrix layout (and when we invoke fxc/dxc we pass in the
        // appropriate flags to request this behavior), so we need to inform Slang that
        // this is what we want/expect so that it can compute correct reflection information.
        //
        sessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

        ComPtr<slang::ISession> pSlangSession;
        pSlangGlobalSession->createSession(
            sessionDesc,
            pSlangSession.writeRef());
        assert(pSlangSession);

        mFileTimeMap.clear();

        SlangCompileRequest* pSlangRequest = nullptr;
        pSlangSession->createCompileRequest(
            &pSlangRequest);
        assert(pSlangRequest);

        // Enable/disable intermediates dump
        bool dumpIR = is_set(mDesc.getCompilerFlags(), Shader::CompilerFlags::DumpIntermediates);
        spSetDumpIntermediates(pSlangRequest, dumpIR);

        if (is_set(mDesc.getCompilerFlags(), Shader::CompilerFlags::GenerateDebugInfo))
        {
            spSetDebugInfoLevel(pSlangRequest, SLANG_DEBUG_INFO_LEVEL_STANDARD);
        }

        // Configure any flags for the Slang compilation step
        SlangCompileFlags slangFlags = 0;

        // When we invoke the Slang compiler front-end, skip code generation step
        // so that the compiler does not complain about missing arguments for
        // specialization parameters.
        //
        slangFlags |= SLANG_COMPILE_FLAG_NO_CODEGEN;

        spSetCompileFlags(pSlangRequest, slangFlags);

        // Now lets add all our input shader code, one-by-one
        int translationUnitsAdded = 0;

        // TODO: All of the sources in a program (or at least all of those
        // in an entry point group) should be considered as a single
        // translation unit for Slang (so that they can see and resolve
        // definitions).
        //
        for(auto src : mDesc.mSources)
        {
            // Register the translation unit with Slang
            int translationUnitIndex = spAddTranslationUnit(pSlangRequest, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
            assert(translationUnitIndex == translationUnitsAdded);
            translationUnitsAdded++;

            // Add source code to the translation unit
            if (src.type == Desc::Source::Type::File)
            {
                // If this is not an HLSL or a SLANG file, display a warning
                if (!hasSuffix(src.pLibrary->getFilename(), ".hlsl", false) && !hasSuffix(src.pLibrary->getFilename(), ".slang", false))
                {
                    logWarning("Compiling a shader file which is not a SLANG file or an HLSL file. This is not an error, but make sure that the file contains valid shaders");
                }
                std::string fullpath;
                if (!findFileInShaderDirectories(src.pLibrary->getFilename(), fullpath))
                {
                    logError("Can't find file " + src.pLibrary->getFilename());
                    spDestroyCompileRequest(pSlangRequest);
                    return nullptr;
                }
                spAddTranslationUnitSourceFile(pSlangRequest, translationUnitIndex, fullpath.c_str());
            }
            else
            {
                assert(src.type == Desc::Source::Type::String);
                spAddTranslationUnitSourceString(pSlangRequest, translationUnitIndex, "", src.str.c_str());
            }
        }

        // Now we make a separate pass and add the entry points.
        // Each entry point references the index of the source
        // it uses, and luckily, the Slang API can use these
        // indices directly.
        for(auto& entryPoint : mDesc.mEntryPoints)
        {
            auto& group = mDesc.mGroups[entryPoint.groupIndex];

            spAddEntryPoint(
                pSlangRequest,
                entryPoint.sourceIndex,
                entryPoint.name.c_str(),
                getSlangStage(entryPoint.stage));
        }

        return pSlangRequest;
    }

    bool Program::doSlangReflection(
        ProgramVersion const*                       pVersion,
        slang::IComponentType*                      pSlangGlobalScope,
        std::vector<ComPtr<slang::IComponentType>>  pSlangLinkedEntryPoints,
        ProgramReflection::SharedPtr&               pReflector,
        std::string&                                log) const
    {
        auto pSlangGlobalScopeLayout = pSlangGlobalScope->getLayout();

        // TODO: actually need to reflect the entry point groups!

        std::vector<slang::EntryPointLayout*> pSlangEntryPointReflectors;

        for( auto pSlangLinkedEntryPoint : pSlangLinkedEntryPoints )
        {
            auto pSlangEntryPointLayout = pSlangLinkedEntryPoint->getLayout()->getEntryPointByIndex(0);
            pSlangEntryPointReflectors.push_back(pSlangEntryPointLayout);
        }

        pReflector = ProgramReflection::create(
            pVersion,
            pSlangGlobalScopeLayout,
            pSlangEntryPointReflectors,
            log);

        return true;
    }

    static ComPtr<slang::IComponentType> doSlangSpecialization(
        slang::IComponentType*                      pSlangProgram,
        ParameterBlock::SpecializationArgs const&   specializationArgs,
        std::string&                                log)
    {
        ComPtr<slang::IBlob> pSlangDiagnostics;
        ComPtr<slang::IComponentType> pSpecializedSlangProgram;
        bool failed = SLANG_FAILED(pSlangProgram->specialize(
            specializationArgs.data(),
            specializationArgs.size(),
            pSpecializedSlangProgram.writeRef(),
            pSlangDiagnostics.writeRef()));

        if (pSlangDiagnostics && pSlangDiagnostics->getBufferSize() > 0)
        {
            log += (char const*) pSlangDiagnostics->getBufferPointer();
        }

        return failed ? nullptr : pSpecializedSlangProgram;
    }

    ProgramKernels::SharedPtr Program::preprocessAndCreateProgramKernels(
        ProgramVersion const* pVersion,
        ProgramVars    const* pVars,
        std::string         & log) const
    {
        auto pSlangGlobalScope = pVersion->getSlangGlobalScope();
        auto pSlangSession = pSlangGlobalScope->getSession();

        // Global-scope specialization parameters apply to all the entry points
        // in a `Program`. We will collect the arguments for global specialization
        // parameters here, using the global `ProgramVars`.
        //
        ParameterBlock::SpecializationArgs specializationArgs;
        pVars->collectSpecializationArgs(specializationArgs);

        // Next we instruct Slang to specialize the global scope based on
        // the global specialization arguments.
        //
        ComPtr<slang::IComponentType> pSpecializedSlangGlobalScope = doSlangSpecialization(
            pSlangGlobalScope,
            specializationArgs,
            log);
        if (!pSpecializedSlangGlobalScope)
        {
            return nullptr;
        }

        uint32_t allEntryPointCount = uint32_t(mDesc.mEntryPoints.size());
        std::vector<ComPtr<slang::IComponentType>> pLinkedEntryPoints;

        for( uint32_t ee = 0; ee < allEntryPointCount; ++ee )
        {
            auto pSlangEntryPoint = pVersion->getSlangEntryPoint(ee);

            slang::IComponentType* componentTypes[] = {pSpecializedSlangGlobalScope, pSlangEntryPoint};

            ComPtr<slang::IComponentType> pLinkedSlangEntryPoint;
            ComPtr<slang::IBlob> pSlangDiagnostics;
            pSlangSession->createCompositeComponentType(
                componentTypes,
                2,
                pLinkedSlangEntryPoint.writeRef(),
                pSlangDiagnostics.writeRef());

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
            for( uint32_t ee = 0; ee < allEntryPointCount; ++ee )
            {
                // TODO: Eventually this would need to use the specialized
                // (but not linked) version of each entry point.
                //
                auto pSlangEntryPoint = pVersion->getSlangEntryPoint(ee);
                componentTypesForProgram.push_back(pSlangEntryPoint);
            }
            pSlangSession->createCompositeComponentType(
                componentTypesForProgram.data(),
                componentTypesForProgram.size(),
                pSpecializedSlangProgram.writeRef());
        }

        ProgramReflection::SharedPtr pReflector;
        doSlangReflection(pVersion, pSpecializedSlangProgram, pLinkedEntryPoints, pReflector, log);

        // In order to construct the `ProgramKernels` we need to extract
        // the kernels for each entry-point group.
        //
        std::vector<EntryPointGroupKernels::SharedPtr> entryPointGroups;

        // TODO: Because we aren't actually specializing entry-point groups,
        // we will again loop over the original unspecialized entry point
        // groups from the `Program::Desc`, and assume that they line up
        // one-to-one with the entries in `pLinkedEntryPointGroups`.
        //
        uint32_t entryPointGroupCount = uint32_t(mDesc.mGroups.size());
        for( uint32_t gg = 0; gg < entryPointGroupCount; ++gg )
        {
            auto entryPointGroupDesc = mDesc.mGroups[gg];

            // For each entry-point group we will collect the compiled kernel
            // code for its constituent entry points, using the "linked"
            // version of the entry-point group.
            //
            auto groupEntryPointCount = entryPointGroupDesc.entryPointCount;
            std::vector<Shader::SharedPtr> shaders;
            for(uint32_t ee = 0; ee < groupEntryPointCount; ++ee)
            {
                auto entryPointIndex = entryPointGroupDesc.firstEntryPoint + ee;

                auto pLinkedEntryPoint = pLinkedEntryPoints[entryPointIndex];
                auto entryPointDesc = mDesc.mEntryPoints[entryPointIndex];

                Shader::Blob blob;
                ComPtr<slang::IBlob> pSlangDiagnostics;
                bool failed = SLANG_FAILED(pLinkedEntryPoint->getEntryPointCode(
                    /* entryPointIndex: */ 0,
                    /* targetIndex: */ 0,
                    blob.writeRef(),
                    pSlangDiagnostics.writeRef()));

                if (pSlangDiagnostics && pSlangDiagnostics->getBufferSize() > 0)
                {
                    log += (char const*) pSlangDiagnostics->getBufferPointer();
                }

                if (failed) return nullptr;

                Shader::SharedPtr shader = createShaderFromBlob(blob, entryPointDesc.stage, entryPointDesc.name, mDesc.getCompilerFlags(), log);
                if (!shader) return nullptr;

                shaders.emplace_back(std::move(shader));
            }

            auto pGroupReflector = pReflector->getEntryPointGroup(gg);

            auto pEntryPointGroupKernels = createEntryPointGroupKernels(shaders, pGroupReflector);
            entryPointGroups.push_back(pEntryPointGroupKernels);
        }

        return ProgramKernels::create(
                pVersion,
                pReflector,
                entryPointGroups,
                log,
                getProgramDescString());
    }

    ProgramVersion::SharedPtr Program::preprocessAndCreateProgramVersion(
        std::string& log) const
    {
        auto pSlangRequest = createSlangCompileRequest(mDefineList);
        if (pSlangRequest == nullptr) return nullptr;

        SlangResult slangResult = spCompile(pSlangRequest);
        log += spGetDiagnosticOutput(pSlangRequest);
        if(SLANG_FAILED(slangResult))
        {
            spDestroyCompileRequest(pSlangRequest);
            return nullptr;
        }

        ComPtr<slang::IComponentType> pSlangGlobalScope;
        spCompileRequest_getProgram(
            pSlangRequest,
            pSlangGlobalScope.writeRef());

        ComPtr<slang::ISession> pSlangSession(pSlangGlobalScope->getSession());

        std::vector<ComPtr<slang::IComponentType>> pSlangEntryPoints;
        uint32_t entryPointCount = (uint32_t) mDesc.mEntryPoints.size();
        for( uint32_t ee = 0; ee < entryPointCount; ++ee )
        {
            auto entryPointDesc = mDesc.mEntryPoints[ee];

            ComPtr<slang::IComponentType> pSlangEntryPoint;
            spCompileRequest_getEntryPoint(
                pSlangRequest,
                ee,
                pSlangEntryPoint.writeRef());

            pSlangEntryPoints.push_back(pSlangEntryPoint);
        }

        // Extract list of files referenced, for dependency-tracking purposes
        int depFileCount = spGetDependencyFileCount(pSlangRequest);
        for(int ii = 0; ii < depFileCount; ++ii)
        {
            std::string depFilePath = spGetDependencyFilePath(pSlangRequest, ii);
            mFileTimeMap[depFilePath] = getFileModifiedTime(depFilePath);
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
        ProgramVersion::SharedPtr pVersion = ProgramVersion::createEmpty(const_cast<Program*>(this), pSlangGlobalScope);

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
        spCompileRequest_getProgram(
            pSlangRequest,
            pSlangProgram.writeRef());

        ProgramReflection::SharedPtr pReflector;
        if( !doSlangReflection(pVersion.get(), pSlangGlobalScope, pSlangEntryPoints, pReflector, log) )
        {
            return nullptr;
        }

        pVersion->init(
            mDefineList,
            pReflector,
            getProgramDescString(),
            pSlangEntryPoints);

        return pVersion;
    }

    EntryPointGroupKernels::SharedPtr Program::createEntryPointGroupKernels(
        const std::vector<Shader::SharedPtr>& shaders,
        EntryPointBaseReflection::SharedPtr const& pReflector) const
    {
        return EntryPointGroupKernels::create(EntryPointGroupKernels::Type::Rasterization, shaders);
    }

    bool Program::link() const
    {
        while(1)
        {
            // Create the program
            std::string log;
            auto pVersion = preprocessAndCreateProgramVersion(log);

            if (pVersion == nullptr)
            {
                std::string error = "Failed to link program:\n" + getProgramDescString() + "\n\n" + log;
                logError(error, Logger::MsgBox::RetryAbort);

                // Continue loop to keep trying...
            }
            else
            {
                if (!log.empty())
                {
                    std::string warn = "Warnings in program:\n" + getProgramDescString() + "\n" + log;
                    logWarning(warn);
                }

                mpActiveVersion = pVersion;
                return true;
            }
        }
    }

    void Program::reset()
    {
        mpActiveVersion = nullptr;
        mProgramVersions.clear();
        mFileTimeMap.clear();
        mLinkRequired = true;
    }

    bool Program::reloadAllPrograms(bool forceReload)
    {
        bool hasReloaded = false;

        // The `sPrograms` array stores weak pointers, and we will
        // use this step as a chance to clean up the contents of
        // the array that might have changed to `nullptr` because
        // the `Program` has been deleted.
        //
        // We will do this cleanup in a single pass without creating
        // a copy of the array by tracking two iterators: one for
        // reading and one for writing. The write iterator will
        // be explicit:
        //
        auto writeIter = sPrograms.begin();
        //
        // The read iterator will be implicit in our loop over the
        // entire array of programs:
        //
        for(auto& pWeakProgram : sPrograms)
        {
            // We will skip any programs where the weak pointer
            // has changed to `nullptr` because the object was
            // already deleted.
            //
            auto pProgram = pWeakProgram.lock();
            if(!pProgram)
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
            if(!(pProgram->checkIfFilesChanged() || forceReload))
                continue;

            // If any files have changed, then we need to reset
            // the caches of compiled information for the program.
            //
            pProgram->reset();

            hasReloaded = true;
        }

        // Once we are done, we will have written a compacted
        // version of `sPrograms` (skipping the null elements)
        // to the first N elements of the vector. To make the
        // vector only contain those first N elements, we
        // then need to erase everything past the last point
        // we wrote to.
        //
        sPrograms.erase(writeIter, sPrograms.end());

        return hasReloaded;
    }

    void Program::addGlobalDefines(const DefineList& defineList)
    {
        sGlobalDefineList.add(defineList);
        reloadAllPrograms(true);
    }

    void Program::removeGlobalDefines(const DefineList& defineList)
    {
        sGlobalDefineList.remove(defineList);
        reloadAllPrograms(true);
    }

    SCRIPT_BINDING(Program)
    {
        pybind11::class_<Program, Program::SharedPtr>(m, "Program");
    }
}
