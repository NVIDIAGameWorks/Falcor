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
#include "Program.h"
#include <vector>
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "Graphics/TextureHelper.h"
#include "Utils/Platform/OS.h"
#include "API/Shader.h"
#include "Graphics/Program/ProgramVersion.h"
#include "API/Texture.h"
#include "API/Sampler.h"
#include "API/RenderContext.h"
#include "Utils/StringUtils.h"
#include "ShaderLibrary.h"
#include "Graphics/Material/Material.h"

namespace Falcor
{
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
        mActiveLibraryIndex = (int)mShaderLibraries.size();
        mShaderLibraries.emplace_back(ShaderLibrary::create(path));

        return *this;
    }

    Program::Desc& Program::Desc::entryPoint(ShaderType shaderType, std::string const& name)
    {
        auto& entryPoint = mEntryPoints[int(shaderType)];
        entryPoint.name = name;
        if (name.size() == 0)
        {
            entryPoint.libraryIndex = -1;
        }
        else
        {
            assert(mActiveLibraryIndex >= 0);
            if (entryPoint.libraryIndex != -1)
            {
                logWarning("Trying to set a " + to_string(shaderType) + " entry-point when one already exists. Overriding previous entry-point");
            }
            entryPoint.libraryIndex = mActiveLibraryIndex;
            entryPoint.name = name;
        }

        return *this;
    }

    Program::Desc& Program::Desc::addDefaultVertexShaderIfNeeded()
    {
        // Don't set default vertex shader if one was set already.
        if (mEntryPoints[int(ShaderType::Vertex)].isValid()) return *this;
        return addShaderLibrary("DefaultVS.slang").entryPoint(ShaderType::Vertex, "defaultVS");        
    }

    const std::string& Program::Desc::getShaderEntryPoint(ShaderType shaderType) const
    {
        static std::string s;
        return mEntryPoints[(uint32_t)shaderType].isValid() ? mEntryPoints[(uint32_t)shaderType].name : s;
    }

    const ShaderLibrary::SharedPtr& Program::Desc::getShaderLibrary(ShaderType shaderType) const
    {
        static ShaderLibrary::SharedPtr pM;;
        const auto& e = mEntryPoints[(uint32_t)shaderType];

        return e.isValid() ? mShaderLibraries[e.libraryIndex] : pM;
    }

    // Program
    std::vector<Program*> Program::sPrograms;

    Program::Program()
    {
        sPrograms.push_back(this);
    }

    void Program::init(Desc const& desc, DefineList const& programDefines)
    {
        mDesc = desc;
        mDefineList = programDefines;
    }

    Program::~Program()
    {
        // Remove the current program from the program vector
        for(auto it = sPrograms.begin() ; it != sPrograms.end() ; it++)
        {
            if(*it == this)
            {
                sPrograms.erase(it);
                break;;
            }
        }
    }

    std::string Program::getProgramDescString() const
    {
        std::string desc = "Program with Shaders:\n";

        int sourceCounter = 0;
        for(auto pModule : mDesc.mShaderLibraries)
        {
            int sourceIndex = sourceCounter++;

            desc += pModule->getFilename();

            for( auto entryPoint : mDesc.mEntryPoints )
            {
                if(entryPoint.libraryIndex != sourceIndex) continue;
                desc += "/*" + entryPoint.name + "*/";
            }
            desc += "\n";
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
        mLinkRequired = true;
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
            mLinkRequired = true;
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
                mLinkRequired = true;
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

    bool Program::clearDefines()
    {
        if (!mDefineList.empty())
        {
            mLinkRequired = true;
            mDefineList.clear();
            return true;
        }
        return false;
    }

    bool Program::replaceAllDefines(const DefineList& dl)
    {
        // TODO: re-link only if new macros differ from existing
        if (!mDefineList.empty() || !dl.empty())
        {
            mLinkRequired = true;
            mDefineList = dl;
            return true;
        }
        return false;
    }

    bool Program::checkIfFilesChanged()
    {
        if(mActiveProgram.pVersion == nullptr)
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

    ProgramVersion::SharedConstPtr Program::getActiveVersion() const
    {
        if(mLinkRequired)
        {
            const auto& it = mProgramVersions.find(mDefineList);
            if(it == mProgramVersions.end())
            {
                if(link() == false)
                {
                    return nullptr;
                }
                else
                {
                    mProgramVersions[mDefineList] = mActiveProgram;
                }
            }
            else
            {
                mActiveProgram = mProgramVersions[mDefineList];
            }
        }

        return mActiveProgram.pVersion;
    }

    SlangSession* getSlangSession()
    {
        // TODO: figure out a strategy for finalizing the Slang session, if desired

        static SlangSession* slangSession = spCreateSession(NULL);
        return slangSession;
    }

    void loadSlangBuiltins(char const* name, char const* text)
    {
        spAddBuiltins(getSlangSession(), name, text);
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
#ifdef FALCOR_DXR
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

    static const char* getSlangProfileString()
    {
#if defined FALCOR_VK
        return "glsl_450";
#elif defined FALCOR_D3D12
        return "sm_5_1";
#else
#error unknown shader compilation target
#endif
    }

    // createSlangCompileRequest now takes global type arguments for specialization
    SlangCompileRequest* Program::createSlangCompileRequest(DefineList const& defines,
        CompilePurpose purpose,
        const std::vector<std::string> & typeArgs) const
    {
        mFileTimeMap.clear();

        // Run all of the shaders through Slang, so that we can get final code,
        // reflection data, etc.
        //
        // Note that we provide all the shaders at once, so that automatically
        // generated bindings can be made consistent across the stages.

        SlangSession* slangSession = getSlangSession();

        // Start building a request for compilation
        SlangCompileRequest* slangRequest = spCreateCompileRequest(slangSession);

        // Add our media search paths as `#include` search paths for Slang.
        //
        // TODO: Slang should probably support a callback API for all file I/O,
        // rather than having us specify data directories to it...
        for (auto path : getDataDirectoriesList())
        {
            spAddSearchPath(slangRequest, path.c_str());
        }

        // Enable/disable intermediates dump
        bool dumpIR = is_set(mDesc.getCompilerFlags(), Shader::CompilerFlags::DumpIntermediates);
        spSetDumpIntermediates(slangRequest, dumpIR);

        // Pass any `#define` flags along to Slang, since we aren't doing our
        // own preprocessing any more.
        for (auto shaderDefine : defines)
        {
            spAddPreprocessorDefine(slangRequest, shaderDefine.first.c_str(), shaderDefine.second.c_str());
        }

        // Pick the right target based on the current graphics API
#ifdef FALCOR_VK
        spSetCodeGenTarget(slangRequest, SLANG_SPIRV);
        spAddPreprocessorDefine(slangRequest, "FALCOR_VK", "1");
#elif defined FALCOR_D3D12
        spAddPreprocessorDefine(slangRequest, "FALCOR_D3D", "1");
        // Note: we could compile Slang directly to DXBC (by having Slang invoke the MS compiler for us,
        // but that path seems to have more issues at present, so let's just go to HLSL instead...)
        spSetCodeGenTarget(slangRequest, SLANG_HLSL);
#else
#error unknown shader compilation target
#endif

        spSetTargetProfile(slangRequest, 0, spFindProfile(slangSession, getSlangProfileString()));

        // Configure any flags for the Slang compilation step
        SlangCompileFlags slangFlags = 0;

        // Don't actually perform semantic checking: just pass through functions bodies to downstream compiler
        slangFlags |= SLANG_COMPILE_FLAG_NO_CHECKING | SLANG_COMPILE_FLAG_SPLIT_MIXED_TYPES;
        if (purpose == CompilePurpose::ReflectionOnly)
            slangFlags |= SLANG_COMPILE_FLAG_NO_CODEGEN;
        spSetCompileFlags(slangRequest, slangFlags);

        // Now lets add all our input shader code, one-by-one
        int translationUnitsAdded = 0;
        for(auto pLibrary : mDesc.mShaderLibraries)
        {
            // If this is not an HLSL or a SLANG file, display a warning
            if (!hasSuffix(pLibrary->getFilename(), ".hlsl", false) && !hasSuffix(pLibrary->getFilename(), ".slang", false))
            {
                logWarning("Compiling a shader file which is not a SLANG file or an HLSL file. This is not an error, but make sure that the file contains valid shaders");
            }

            // Register the translation unit with Slang
            int translationUnitIndex = spAddTranslationUnit(slangRequest, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
            assert(translationUnitIndex == translationUnitsAdded);
            translationUnitsAdded++;

            // Add source code to the translation unit
            std::string fullpath;
            findFileInDataDirectories(pLibrary->getFilename(), fullpath);
            spAddTranslationUnitSourceFile(slangRequest, translationUnitIndex, fullpath.c_str());
        }

        std::vector<char const*> typeArgNames;
        for (auto & name : typeArgs)
            typeArgNames.push_back(name.c_str());
        char const ** typeArgNamesPtr = nullptr;
        if (typeArgNames.size())
            typeArgNamesPtr = &typeArgNames[0];

        // Now we make a separate pass and add the entry points.
        // Each entry point references the index of the source
        // it uses, and luckily, the Slang API can use these
        // indices directly.
        if (purpose != CompilePurpose::ReflectionOnly)
        {
            for (uint32_t i = 0; i < kShaderCount; ++i)
            {
                auto& entryPoint = mDesc.mEntryPoints[i];

                // Skip unused entry points
                if (entryPoint.libraryIndex < 0)
                    continue;

                spAddEntryPointEx(
                    slangRequest,
                    entryPoint.libraryIndex,
                    entryPoint.name.c_str(),
                    getSlangStage(ShaderType(i)),
                    (int)typeArgs.size(),
                    typeArgNamesPtr);
            }
        }
        return slangRequest;
    }

    int Program::doSlangCompilation(SlangCompileRequest* slangRequest, std::string& log) const
    {
        int anySlangErrors = spCompile(slangRequest);
        log += spGetDiagnosticOutput(slangRequest);
        if (anySlangErrors)
        {
            spDestroyCompileRequest(slangRequest);
            return 1;
        }
        return 0;
    }

    Program::VersionData Program::preprocessAndCreateProgramVersion(std::string& log) const
    {
        SlangCompileRequest* slangRequest = createSlangCompileRequest(mDefineList,
            CompilePurpose::ReflectionOnly,
            std::vector<std::string>());
        int anyErrors = doSlangCompilation(slangRequest, log);
        if (anyErrors)
            return VersionData();

        VersionData programVersion;

        // Extract the reflection data
        programVersion.reflectors.pReflector = ProgramReflection::create(slang::ShaderReflection::get(slangRequest), ProgramReflection::ResourceScope::All, log);
        programVersion.reflectors.pLocalReflector = ProgramReflection::create(slang::ShaderReflection::get(slangRequest), ProgramReflection::ResourceScope::Local, log);
        programVersion.reflectors.pGlobalReflector = ProgramReflection::create(slang::ShaderReflection::get(slangRequest), ProgramReflection::ResourceScope::Global, log);

        // Extract list of files referenced, for dependency-tracking purposes
        int depFileCount = spGetDependencyFileCount(slangRequest);
        for(int ii = 0; ii < depFileCount; ++ii)
        {
            std::string depFilePath = spGetDependencyFilePath(slangRequest, ii);
            mFileTimeMap[depFilePath] = getFileModifiedTime(depFilePath);
        }

        // Now that we've preprocessed things, dispatch to the actual program creation logic,
        // which may vary in subclasses of `Program`
        programVersion.pVersion = ProgramVersion::create(const_cast<Program*>(this)->shared_from_this(), mDefineList, programVersion.reflectors.pReflector, getProgramDescString(), slangRequest);
        return programVersion;
    }

    ProgramKernels::SharedPtr Program::preprocessAndCreateProgramKernels(
        ProgramVersion const* pVersion,
        ProgramVars    const* pVars,
        const std::vector<std::string>  &newEntryPointNames,
        std::string         & log) const
    {
        // TODO: bind type parameters as needed based on `pVars`
        auto originalReflector = pVersion->getReflector();
        auto paramBlockCount = originalReflector->getParameterBlockCount();
        std::vector<std::string> typeArguments;
        for (uint32_t i = 0; i < paramBlockCount; i++)
        {
            auto paramBlock = originalReflector->getParameterBlock(i);
            auto newParamBlock = pVars->getParameterBlock(i);
            std::string typeParamName = newParamBlock->genericTypeParamName;
            if (newParamBlock->getTypeName() == "TMaterial")
                typeParamName = "TMaterial";
            if (typeParamName.length() )
            {
                auto index = originalReflector->getTypeParameterIndexByName(typeParamName);
                if (typeArguments.size() <= index)
                    typeArguments.resize(index + 1);
                typeArguments[index] = newParamBlock->genericTypeArgumentName;
                if (typeParamName == "TMaterial" && newParamBlock->genericTypeArgumentName.length() == 0)
                    typeArguments[index] = Material::kDefaultMaterialType;
            }
        }
        SlangCompileRequest* slangRequest = createSlangCompileRequest(pVersion->getDefines(),
            CompilePurpose::CodeGen,
            typeArguments);
        
        int anyErrors = doSlangCompilation(slangRequest, log);
        if (anyErrors)
            return nullptr;
        
        // Extract the generated code for each stage
        int entryPointCounter = 0;
        Shader::Blob shaderBlob[kShaderCount];

        for (uint32_t i = 0; i < kShaderCount; i++)
        {
            auto& entryPoint = mDesc.mEntryPoints[i];

            // Skip unused entry points
            if(entryPoint.libraryIndex < 0)
                continue;
            int entryPointIndex = entryPointCounter++;

            size_t size = 0;
#ifdef FALCOR_VK
            const uint8_t* data = (uint8_t*)spGetEntryPointCode(slangRequest, entryPointIndex, &size);
            shaderBlob[i].data.assign(data, data + size);
            shaderBlob[i].type = Shader::Blob::Type::Bytecode;
#else
            auto find_and_replace = [](std::string& source, std::string const& find, std::string const& replace)
            {
                for (std::string::size_type i = 0; (i = source.find(find, i)) != std::string::npos;)
                {
                    source.replace(i, find.length(), replace);
                    i += replace.length();
                }
            };
            auto srcStr = std::string(spGetEntryPointSource(slangRequest, entryPointIndex));
            if (newEntryPointNames.size() > entryPointIndex)
            {
                find_and_replace(srcStr, entryPoint.name, newEntryPointNames[i]);
            }
            shaderBlob[i].data.assign(srcStr.c_str(), srcStr.c_str() + srcStr.length());
            shaderBlob[i].type = Shader::Blob::Type::String;
#endif
        }
        // Extract the reflection data
        auto kernelReflector = ProgramReflection::create(slang::ShaderReflection::get(slangRequest), ProgramReflection::ResourceScope::All, log);
        return createProgramKernels(log, shaderBlob, kernelReflector, newEntryPointNames);
    }

    ProgramKernels::SharedPtr Program::createProgramKernels(std::string& log, const Shader::Blob shaderBlob[kShaderCount], ProgramReflection::SharedPtr reflector,
        const std::vector<std::string> & entryPointNames) const
    {
        // create the shaders
        Shader::SharedPtr shaders[kShaderCount] = {};
        for (uint32_t i = 0; i < kShaderCount; i++)
        {
            if (shaderBlob[i].data.size())
            {
                shaders[i] = createShaderFromBlob(shaderBlob[i], ShaderType(i), 
                    (entryPointNames.size() > i ? entryPointNames[i] : mDesc.mEntryPoints[i].name), 
                    mDesc.getCompilerFlags(), log);
                if (!shaders[i]) return nullptr;
            }
        }

        auto rootSignature = RootSignature::create(reflector.get());

        if (shaders[(uint32_t)ShaderType::Compute])
        {
            return ProgramKernels::create(
                reflector,
                shaders[(uint32_t)ShaderType::Compute], rootSignature, log, getProgramDescString());
        }
        else
        {
            return ProgramKernels::create(
                reflector,
                shaders[(uint32_t)ShaderType::Vertex],
                shaders[(uint32_t)ShaderType::Pixel],
                shaders[(uint32_t)ShaderType::Geometry],
                shaders[(uint32_t)ShaderType::Hull],
                shaders[(uint32_t)ShaderType::Domain],
                rootSignature,
                log,
                getProgramDescString());
        }
    }


    bool Program::link() const
    {
        while(1)
        {
            // create the program
            std::string log;
            VersionData programVersion = preprocessAndCreateProgramVersion(log);

            if(programVersion.pVersion == nullptr)
            {
                std::string error = std::string("Program Linkage failed.\n\n");
                error += getProgramDescString() + "\n";
                error += log;

                MsgBoxButton button = msgBox(error, MsgBoxType::AbortRetryIgnore);
                if (button == MsgBoxButton::Abort)
                {
                    logErrorAndExit(error);
                }
                else if (button == MsgBoxButton::Ignore)
                {
                    logError(error);
                    return false;
                }
            }
            else
            {
                mActiveProgram = programVersion;
                return true;
            }
        }
    }

    void Program::reset()
    {
        mActiveProgram = VersionData();
        mProgramVersions.clear();
        mFileTimeMap.clear();
        mLinkRequired = true;
    }

    void Program::reloadAllPrograms()
    {
        for(auto& pProgram : sPrograms)
        {
            if(pProgram->checkIfFilesChanged())
            {
                pProgram->reset();
            }
        }
    }

}
