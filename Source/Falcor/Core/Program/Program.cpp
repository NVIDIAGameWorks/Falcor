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
#include "Program.h"
#include "ProgramVars.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include "Core/API/ParameterBlock.h"
#include "Utils/StringUtils.h"
#include "Utils/Logger.h"
#include "Utils/Timing/CpuTimer.h"
#include "Utils/Scripting/ScriptBindings.h"

#include <slang.h>

#include <set>

namespace Falcor
{
    const std::string kSupportedShaderModels[] = { "6_0", "6_1", "6_2", "6_3", "6_4", "6_5"
#if FALCOR_HAS_D3D12_AGILITY_SDK
        , "6_6"
#endif
    };

    static Program::DefineList sGlobalDefineList;
    static bool sGenerateDebugInfo;
    static Program::ForcedCompilerFlags sForcedCompilerFlags;

    Program::Desc applyForcedCompilerFlags(Program::Desc desc)
    {
        Shader::CompilerFlags flags = desc.getCompilerFlags();
        flags &= ~sForcedCompilerFlags.disabled;
        flags |= sForcedCompilerFlags.enabled;
        desc.setCompilerFlags(flags);
        return desc;
    }

    Program::Desc::Desc() = default;

    Program::Desc::Desc(const std::filesystem::path& path)
    {
        addShaderLibrary(path);
    }

    Program::Desc& Program::Desc::addShaderModule(const ShaderModule& src)
    {
        if (src.type == ShaderModule::Type::String && src.createTranslationUnit && src.moduleName.empty())
        {
            // Warn if module name is left empty when creating a new translation unit from string.
            // This is valid, but unexpected so issue a warning.
            logWarning("addShaderModule() - Creating a new translation unit, but missing module name. Is this intended?");
        }

        SourceEntryPoints source(src);
        mActiveSource = (int32_t)mSources.size();
        mSources.emplace_back(std::move(source));

        return *this;
    }

    Program::Desc& Program::Desc::addShaderModules(const ShaderModuleList& modules)
    {
        for (const auto& module : modules)
        {
            addShaderModule(module);
        }
        return *this;
    }

    Program::Desc& Program::Desc::beginEntryPointGroup(const std::string& entryPointNameSuffix)
    {
        mActiveGroup = (int32_t)mGroups.size();
        mGroups.push_back(EntryPointGroup());
        mGroups[mActiveGroup].nameSuffix = entryPointNameSuffix;

        return *this;
    }

    Program::Desc& Program::Desc::entryPoint(ShaderType shaderType, std::string const& name)
    {
        checkArgument(!name.empty(), "Missing entry point name.");

        if (mActiveGroup < 0)
        {
            beginEntryPointGroup();
        }

        uint32_t entryPointIndex = declareEntryPoint(shaderType, name);
        mGroups[mActiveGroup].entryPoints.push_back(entryPointIndex);
        return *this;
    }

    bool Program::Desc::hasEntryPoint(ShaderType stage) const
    {
        for (auto& entryPoint : mEntryPoints)
        {
            if (entryPoint.stage == stage)
            {
                return true;
            }
        }
        return false;
    }

    Program::Desc& Program::Desc::addTypeConformancesToGroup(const TypeConformanceList& typeConformances)
    {
        FALCOR_ASSERT(mActiveGroup >= 0);
        mGroups[mActiveGroup].typeConformances.add(typeConformances);
        return *this;
    }

    uint32_t Program::Desc::declareEntryPoint(ShaderType type, const std::string& name)
    {
        FALCOR_ASSERT(!name.empty());
        FALCOR_ASSERT(mActiveGroup >= 0 && mActiveGroup < mGroups.size());

        if (mActiveSource < 0)
        {
            throw RuntimeError("Cannot declare an entry point without first adding a source file/library");
        }

        EntryPoint entryPoint;
        entryPoint.stage = type;
        entryPoint.name = name;
        entryPoint.exportName = name + mGroups[mActiveGroup].nameSuffix;
        entryPoint.sourceIndex = mActiveSource;
        entryPoint.groupIndex = mActiveGroup;

        uint32_t index = (uint32_t)mEntryPoints.size();
        mEntryPoints.push_back(entryPoint);
        mSources[mActiveSource].entryPoints.push_back(index);

        return index;
    }

    Program::Desc& Program::Desc::setShaderModel(const std::string& sm)
    {
        // Check that the model is supported
        bool b = false;
        for (size_t i = 0; i < std::size(kSupportedShaderModels); i++)
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
            for (size_t i = 0; i < std::size(kSupportedShaderModels); i++)
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

    // Program
    std::vector<std::weak_ptr<Program>> Program::sProgramsForReload;
    Program::CompilationStats Program::sCompilationStats;

    void Program::registerProgramForReload(const SharedPtr& pProg)
    {
        sProgramsForReload.push_back(pProg);
    }

    Program::Program(Desc const& desc, DefineList const& defineList)
        : mDesc(applyForcedCompilerFlags(desc))
        , mDefineList(defineList)
        , mTypeConformanceList(desc.mTypeConformances)
    {
        validateEntryPoints();
    }

    void Program::validateEntryPoints() const
    {
        // Check that all exported entry point names are unique for each shader type.
        // They don't necessarily have to be, but it could be an indication of the program not created correctly.
        using NameTypePair = std::pair<std::string, ShaderType>;
        std::set<NameTypePair> entryPointNamesAndTypes;
        for (const auto& e : mDesc.mEntryPoints)
        {
            if (!entryPointNamesAndTypes.insert(NameTypePair(e.exportName, e.stage)).second)
            {
                logWarning("Duplicate program entry points '{}' of type '{}'.", e.exportName, to_string(e.stage));
            }
        }
    }

    Program::~Program()
    {
    }

    std::string Program::getProgramDescString() const
    {
        std::string desc;

        int32_t groupCount = (int32_t)mDesc.mGroups.size();

        for (size_t i = 0; i < mDesc.mSources.size(); i++)
        {
            const auto& src = mDesc.mSources[i];
            if (i != 0) desc += " ";
            switch (src.getType())
            {
            case ShaderModule::Type::File:
                desc += src.source.filePath.string();
                break;
            case ShaderModule::Type::String:
                desc += "Created from string";
                break;
            default:
                FALCOR_UNREACHABLE();
            }

            desc += "(";
            for (size_t ee = 0; ee < src.entryPoints.size(); ++ee)
            {
                auto& entryPoint = mDesc.mEntryPoints[src.entryPoints[ee]];

                if (ee != 0) desc += ", ";
                desc += entryPoint.exportName;
            }
            desc += ")";
        }

        return desc;
    }

    bool Program::addDefine(const std::string& name, const std::string& value)
    {
        // Make sure that it doesn't exist already
        if (mDefineList.find(name) != mDefineList.end())
        {
            if (mDefineList[name] == value)
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
        if (mDefineList.find(name) != mDefineList.end())
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

    bool Program::addTypeConformance(const std::string& typeName, const std::string interfaceType, uint32_t id)
    {
        Shader::TypeConformance conformance = Shader::TypeConformance(typeName, interfaceType);
        if (mTypeConformanceList.find(conformance) == mTypeConformanceList.end())
        {
            markDirty();
            mTypeConformanceList.add(typeName, interfaceType, id);
            return true;
        }
        return false;
    }

    bool Program::removeTypeConformance(const std::string& typeName, const std::string interfaceType)
    {
        Shader::TypeConformance conformance = Shader::TypeConformance(typeName, interfaceType);
        if (mTypeConformanceList.find(conformance) != mTypeConformanceList.end())
        {
            markDirty();
            mTypeConformanceList.remove(typeName, interfaceType);
            return true;
        }
        return false;
    }

    bool Program::setTypeConformances(const TypeConformanceList& conformances)
    {
        if (conformances != mTypeConformanceList)
        {
            markDirty();
            mTypeConformanceList = conformances;
            return true;
        }
        return false;
    }

    bool Program::checkIfFilesChanged()
    {
        if (mpActiveVersion == nullptr)
        {
            // We never linked, so nothing really changed
            return false;
        }

        // Have any of the files we depend on changed?
        for (auto& entry : mFileTimeMap)
        {
            auto& path = entry.first;
            auto& modifiedTime = entry.second;

            if (modifiedTime != getFileModifiedTime(path))
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
                    throw RuntimeError("Program linkage failed");
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
        FALCOR_ASSERT(mpActiveVersion);
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
        switch (type)
        {
        case ShaderType::Vertex:        return SLANG_STAGE_VERTEX;
        case ShaderType::Pixel:         return SLANG_STAGE_PIXEL;
        case ShaderType::Geometry:      return SLANG_STAGE_GEOMETRY;
        case ShaderType::Hull:          return SLANG_STAGE_HULL;
        case ShaderType::Domain:        return SLANG_STAGE_DOMAIN;
        case ShaderType::Compute:       return SLANG_STAGE_COMPUTE;
        case ShaderType::RayGeneration: return SLANG_STAGE_RAY_GENERATION;
        case ShaderType::Intersection:  return SLANG_STAGE_INTERSECTION;
        case ShaderType::AnyHit:        return SLANG_STAGE_ANY_HIT;
        case ShaderType::ClosestHit:    return SLANG_STAGE_CLOSEST_HIT;
        case ShaderType::Miss:          return SLANG_STAGE_MISS;
        case ShaderType::Callable:      return SLANG_STAGE_CALLABLE;
        default:
            FALCOR_UNREACHABLE();
            return SLANG_STAGE_NONE;
        }
    }

    static std::string getSlangProfileString(const std::string& shaderModel)
    {
        return "sm_" + shaderModel;
    }

    void Program::setUpSlangCompilationTarget(
        slang::TargetDesc&  ioTargetDesc,
        char const*&        ioTargetMacroName) const
    {
#ifdef FALCOR_D3D12
        ioTargetDesc.format = SLANG_DXIL;
        ioTargetMacroName = "FALCOR_D3D";
#else
        switch (gpDevice->getApiHandle()->getDeviceInfo().deviceType)
        {
        case gfx::DeviceType::DirectX12:
            ioTargetDesc.format = SLANG_DXIL;
            ioTargetMacroName = "FALCOR_D3D";
            break;
        case gfx::DeviceType::Vulkan:
            ioTargetDesc.format = SLANG_SPIRV;
            ioTargetMacroName = "FALCOR_VK";
            break;
        default:
            FALCOR_UNREACHABLE();
        }
#endif
    }

    SlangCompileRequest* Program::createSlangCompileRequest(
        const DefineList& defineList) const
    {
        slang::IGlobalSession* pSlangGlobalSession = getSlangGlobalSession();
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
        targetDesc.profile = pSlangGlobalSession->findProfile(getSlangProfileString(mDesc.mShaderModel).c_str());

        if (targetDesc.profile == SLANG_PROFILE_UNKNOWN)
        {
            reportError("Can't find Slang profile for shader model " + mDesc.mShaderModel);
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

        targetDesc.forceGLSLScalarBufferLayout = true;

        const char* targetMacroName;

        // Pick the right target based on the current graphics API
        setUpSlangCompilationTarget(targetDesc, targetMacroName);

        // Pass any `#define` flags along to Slang, since we aren't doing our
        // own preprocessing any more.
        //
        std::vector<slang::PreprocessorMacroDesc> slangDefines;
        const auto addSlangDefine = [&slangDefines](const char* name, const char* value)
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
        sessionDesc.preprocessorMacroCount = (SlangInt)slangDefines.size();

        sessionDesc.targets = &targetDesc;
        sessionDesc.targetCount = 1;

        // We always use row-major matrix layout in Falcor so by default that's what we pass to Slang
        // to allow it to compute correct reflection information. Slang then invokes the downstream compiler.
        // Column major option can be useful when compiling external shader sources that don't depend
        // on anything Falcor.
        bool useColumnMajor = is_set(mDesc.getCompilerFlags(), Shader::CompilerFlags::MatrixLayoutColumnMajor);
        sessionDesc.defaultMatrixLayoutMode = useColumnMajor ? SLANG_MATRIX_LAYOUT_COLUMN_MAJOR : SLANG_MATRIX_LAYOUT_ROW_MAJOR;

        ComPtr<slang::ISession> pSlangSession;
        pSlangGlobalSession->createSession(
            sessionDesc,
            pSlangSession.writeRef());
        FALCOR_ASSERT(pSlangSession);

        mFileTimeMap.clear();

        SlangCompileRequest* pSlangRequest = nullptr;
        pSlangSession->createCompileRequest(
            &pSlangRequest);
        FALCOR_ASSERT(pSlangRequest);

        // Enable/disable intermediates dump
        bool dumpIR = is_set(mDesc.getCompilerFlags(), Shader::CompilerFlags::DumpIntermediates);
        spSetDumpIntermediates(pSlangRequest, dumpIR);

        if (sGenerateDebugInfo || is_set(mDesc.getCompilerFlags(), Shader::CompilerFlags::GenerateDebugInfo))
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

        // Set additional command line arguments.
        if (!mDesc.mCompilerArguments.empty())
        {
            std::vector<const char*> args;
            for (const auto& arg : mDesc.mCompilerArguments) args.push_back(arg.c_str());
            spProcessCommandLineArguments(pSlangRequest, args.data(), (int)args.size());
        }

        // Now lets add all our input shader code, one-by-one
        int translationUnitsAdded = 0;
        int translationUnitIndex = -1;

        for (auto src : mDesc.mSources)
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
            if (src.getType() == ShaderModule::Type::File)
            {
                // If this is not an HLSL or a SLANG file, display a warning
                const auto& path = src.source.filePath;
                if (!(hasExtension(path, "hlsl") || hasExtension(path, "slang")))
                {
                    logWarning("Compiling a shader file which is not a SLANG file or an HLSL file. This is not an error, but make sure that the file contains valid shaders");
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
                FALCOR_ASSERT(src.getType() == ShaderModule::Type::String);
                spAddTranslationUnitSourceString(pSlangRequest, translationUnitIndex, src.source.modulePath.c_str(), src.source.str.c_str());
            }
        }

        // Now we make a separate pass and add the entry points.
        // Each entry point references the index of the source
        // it uses, and luckily, the Slang API can use these
        // indices directly.
        for (auto& entryPoint : mDesc.mEntryPoints)
        {
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
            log += (char const*)pSlangDiagnostics->getBufferPointer();
        }

        return failed ? nullptr : pSpecializedSlangProgram;
    }

    ProgramKernels::SharedPtr Program::preprocessAndCreateProgramKernels(
        ProgramVersion const* pVersion,
        ProgramVars    const* pVars,
        std::string         & log) const
    {
        CpuTimer timer;
        timer.update();

        auto pSlangGlobalScope = pVersion->getSlangGlobalScope();
        auto pSlangSession = pSlangGlobalScope->getSession();

#ifdef FALCOR_D3D12
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
#else
        slang::IComponentType* pSpecializedSlangGlobalScope = pSlangGlobalScope;
#endif
        // Create a composite component type that represents all type conformances
        // linked into the `ProgramVersion`.
        auto createTypeConformanceComponentList = [&](const TypeConformanceList& typeConformances) -> std::optional<ComPtr<slang::IComponentType>>
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
                    log += fmt::format("Interface type '{}' in type conformance was not found.\n", typeConformance.first.mInterfaceName.c_str());
                    return {};
                }

                auto res = pSlangSession->createTypeConformanceComponentType(
                    slangType,
                    slangInterfaceType,
                    pTypeConformanceComponent.writeRef(),
                    (SlangInt)typeConformance.second,
                    pSlangDiagnostics.writeRef());
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
                    &typeConformanceComponentRawPtrList[0],
                    (SlangInt)typeConformanceComponentRawPtrList.size(),
                    pTypeConformancesCompositeComponent.writeRef(),
                    pSlangDiagnostics.writeRef());
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
        typeConformancesCompositeComponents.reserve(getEntryPointGroupCount());
        for (const auto& group : mDesc.mGroups)
        {
            TypeConformanceList typeConformances = mTypeConformanceList;
            typeConformances.add(group.typeConformances);
            if (auto typeConformanceComponentList = createTypeConformanceComponentList(typeConformances))
                typeConformancesCompositeComponents.emplace_back(*typeConformanceComponentList);
            else
                return nullptr;
        }

        // Create a `IComponentType` for each entry point.
        uint32_t allEntryPointCount = uint32_t(mDesc.mEntryPoints.size());

        std::vector<ComPtr<slang::IComponentType>> pTypeConformanceSpecializedEntryPoints;
        std::vector<slang::IComponentType*> pTypeConformanceSpecializedEntryPointsRawPtr;
        std::vector<ComPtr<slang::IComponentType>> pLinkedEntryPoints;

        for (uint32_t ee = 0; ee < allEntryPointCount; ++ee)
        {
            auto pSlangEntryPoint = pVersion->getSlangEntryPoint(ee);

            int32_t groupIndex = mDesc.mEntryPoints[ee].groupIndex;
            FALCOR_ASSERT(groupIndex >= 0 && groupIndex < typeConformancesCompositeComponents.size());

            ComPtr<slang::IBlob> pSlangDiagnostics;

            ComPtr<slang::IComponentType> pTypeComformanceSpecializedEntryPoint;
            if (typeConformancesCompositeComponents[groupIndex])
            {
                slang::IComponentType* componentTypes[] = { pSlangEntryPoint, typeConformancesCompositeComponents[groupIndex] };
                auto res = pSlangSession->createCompositeComponentType(
                    componentTypes,
                    2,
                    pTypeComformanceSpecializedEntryPoint.writeRef(),
                    pSlangDiagnostics.writeRef());
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
                slang::IComponentType* componentTypes[] = { pSpecializedSlangGlobalScope, pTypeComformanceSpecializedEntryPoint };

                auto res = pSlangSession->createCompositeComponentType(
                    componentTypes,
                    2,
                    pLinkedSlangEntryPoint.writeRef(),
                    pSlangDiagnostics.writeRef());
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
                auto pSlangEntryPoint = pVersion->getSlangEntryPoint(ee);
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
                componentTypesForProgram.data(),
                componentTypesForProgram.size(),
                pSpecializedSlangProgram.writeRef());
            if (SLANG_FAILED(res))
            {
                log += "Slang call createCompositeComponentType() failed.\n";
                return nullptr;
            }
        }

        ProgramReflection::SharedPtr pReflector;
        doSlangReflection(pVersion, pSpecializedSlangProgram, pLinkedEntryPoints, pReflector, log);

        // Create Shader objects for each entry point and cache them here.
        std::vector<Shader::SharedPtr> allShaders;
        for (uint32_t i = 0; i < allEntryPointCount; i++)
        {
            auto pLinkedEntryPoint = pLinkedEntryPoints[i];
            auto entryPointDesc = mDesc.mEntryPoints[i];

            Shader::SharedPtr shader = Shader::create(pLinkedEntryPoint, entryPointDesc.stage, entryPointDesc.exportName, mDesc.getCompilerFlags(), log);
            if (!shader) return nullptr;

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
        uint32_t entryPointGroupCount = uint32_t(mDesc.mGroups.size());
        for (uint32_t gg = 0; gg < entryPointGroupCount; ++gg)
        {
            auto entryPointGroupDesc = mDesc.mGroups[gg];
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

        auto descStr = getProgramDescString();
        ProgramKernels::SharedPtr pProgramKernels = createProgramKernels(
            pVersion,
            pSpecializedSlangGlobalScope,
            pTypeConformanceSpecializedEntryPointsRawPtr,
            pReflector,
            entryPointGroups,
            log,
            descStr);

        timer.update();
        double time = timer.delta();
        sCompilationStats.programKernelsCount++;
        sCompilationStats.programKernelsTotalTime += time;
        sCompilationStats.programKernelsMaxTime = std::max(sCompilationStats.programKernelsMaxTime, time);
        logDebug("Created program kernels in {:.3f} s: {}", time, descStr);

        return pProgramKernels;
    }

    ProgramKernels::SharedPtr Program::createProgramKernels(
        const ProgramVersion* pVersion,
        slang::IComponentType* pSpecializedSlangGlobalScope,
        const std::vector<slang::IComponentType*>& pTypeConformanceSpecializedEntryPoints,
        const ProgramReflection::SharedPtr& pReflector,
        const ProgramKernels::UniqueEntryPointGroups& uniqueEntryPointGroups,
        std::string& log,
        const std::string& name) const
    {
        return ProgramKernels::create(
            pVersion,
            pSpecializedSlangGlobalScope,
            pTypeConformanceSpecializedEntryPoints,
            pReflector,
            uniqueEntryPointGroups,
            log,
            name);
    }

    ProgramVersion::SharedPtr Program::preprocessAndCreateProgramVersion(
        std::string& log) const
    {
        CpuTimer timer;
        timer.update();

        auto pSlangRequest = createSlangCompileRequest(mDefineList);
        if (pSlangRequest == nullptr) return nullptr;

        SlangResult slangResult = spCompile(pSlangRequest);
        log += spGetDiagnosticOutput(pSlangRequest);
        if (SLANG_FAILED(slangResult))
        {
            spDestroyCompileRequest(pSlangRequest);
            return nullptr;
        }

        ComPtr<slang::IComponentType> pSlangGlobalScope;
        spCompileRequest_getProgram(
            pSlangRequest,
            pSlangGlobalScope.writeRef());

        ComPtr<slang::ISession> pSlangSession(pSlangGlobalScope->getSession());

        // Prepare entry points.
        std::vector<ComPtr<slang::IComponentType>> pSlangEntryPoints;
        uint32_t entryPointCount = (uint32_t)mDesc.mEntryPoints.size();
        for (uint32_t ee = 0; ee < entryPointCount; ++ee)
        {
            ComPtr<slang::IComponentType> pSlangEntryPoint;
            spCompileRequest_getEntryPoint(
                pSlangRequest,
                ee,
                pSlangEntryPoint.writeRef());

            // Rename entry point in the generated code if the exported name differs from the source name.
            // This makes it possible to generate different specializations of the same source entry point,
            // for example by setting different type conformances.
            const auto& entryPointDesc = mDesc.mEntryPoints[ee];
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
        if (!doSlangReflection(pVersion.get(), pSlangGlobalScope, pSlangEntryPoints, pReflector, log))
        {
            return nullptr;
        }

        auto descStr = getProgramDescString();
        pVersion->init(
            mDefineList,
            pReflector,
            descStr,
            pSlangEntryPoints);

        timer.update();
        double time = timer.delta();
        sCompilationStats.programVersionCount++;
        sCompilationStats.programVersionTotalTime += time;
        sCompilationStats.programVersionMaxTime = std::max(sCompilationStats.programVersionMaxTime, time);
        logDebug("Created program version in {:.3f} s: {}", timer.delta(), descStr);

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
        while (1)
        {
            // Create the program
            std::string log;
            auto pVersion = preprocessAndCreateProgramVersion(log);

            if (pVersion == nullptr)
            {
                std::string error = "Failed to link program:\n" + getProgramDescString() + "\n\n" + log;
                reportErrorAndAllowRetry(error);

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

        // The `sProgramsForReload` array stores weak pointers, and we will
        // use this step as a chance to clean up the contents of
        // the array that might have changed to `nullptr` because
        // the `Program` has been deleted.
        //
        // We will do this cleanup in a single pass without creating
        // a copy of the array by tracking two iterators: one for
        // reading and one for writing. The write iterator will
        // be explicit:
        //
        auto writeIter = sProgramsForReload.begin();
        //
        // The read iterator will be implicit in our loop over the
        // entire array of programs:
        //
        for (auto& pWeakProgram : sProgramsForReload)
        {
            // We will skip any programs where the weak pointer
            // has changed to `nullptr` because the object was
            // already deleted.
            //
            auto pProgram = pWeakProgram.lock();
            if (!pProgram) continue;

            // Now we know that we have a valid (non-null) `Program`,
            // so we wnat to keep it in the array for next time.
            //
            *writeIter++ = pProgram;

            // Next we check if any of the files that affected the
            // compilation of `pProgram` has been changed. If not,
            // we can skip further processing of this program
            // (unless forceReload flag is set).
            //
            if (!(pProgram->checkIfFilesChanged() || forceReload)) continue;

            // If any files have changed, then we need to reset
            // the caches of compiled information for the program.
            //
            pProgram->reset();

            hasReloaded = true;
        }

        // Once we are done, we will have written a compacted
        // version of `sProgramsForReload` (skipping the null elements)
        // to the first N elements of the vector. To make the
        // vector only contain those first N elements, we
        // then need to erase everything past the last point
        // we wrote to.
        //
        sProgramsForReload.erase(writeIter, sProgramsForReload.end());

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

    void Program::setGenerateDebugInfoEnabled(bool enabled)
    {
        sGenerateDebugInfo = enabled;
    }

    bool Program::isGenerateDebugInfoEnabled()
    {
        return sGenerateDebugInfo;
    }

    void Program::setForcedCompilerFlags(ForcedCompilerFlags forcedCompilerFlags)
    {
        sForcedCompilerFlags = forcedCompilerFlags;
    }

    Program::ForcedCompilerFlags Program::getForcedCompilerFlags() { return sForcedCompilerFlags; }

    FALCOR_SCRIPT_BINDING(Program)
    {
        pybind11::class_<Program, Program::SharedPtr>(m, "Program");
    }
}
