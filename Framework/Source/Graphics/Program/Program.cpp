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

namespace Falcor
{
#ifdef FALCOR_VK
    const std::string kSupportedShaderModels[] = { "400", "410", "420", "430", "440", "450", "460" };
#elif defined FALCOR_D3D12
    const std::string kSupportedShaderModels[] = { "4_0", "4_1", "5_0", "5_1", "6_0", "6_1", "6_2", "6_3" };
#endif

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
        mActiveSource = (int)mSources.size();
        mSources.emplace_back(ShaderLibrary::create(path));
        return *this;
    }

    Program::Desc& Program::Desc::addShaderString(const std::string& shader)
    {
        mActiveSource = (int)mSources.size();
        mSources.emplace_back(shader);

        return *this;
    }

    Program::Desc& Program::Desc::entryPoint(ShaderType shaderType, std::string const& name)
    {
        auto& entryPoint = mEntryPoints[int(shaderType)];
        entryPoint.name = name;
        if (name.size() == 0)
        {
            entryPoint.index = -1;
        }
        else
        {
            assert(mActiveSource >= 0);
            if (entryPoint.index != -1)
            {
                logWarning("Trying to set a " + to_string(shaderType) + " entry-point when one already exists. Overriding previous entry-point");
            }
            entryPoint.index = mActiveSource;
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
            std::string warn = "Unsupported shader-model `" + sm + "` requested. Supported shader-models are ";
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

    const ShaderLibrary::SharedPtr& Program::Desc::getShaderLibrary(ShaderType shaderType) const
    {
        static ShaderLibrary::SharedPtr pM;
        const auto& e = mEntryPoints[(uint32_t)shaderType];

        return e.isValid() ? mSources[e.index].pLibrary : pM;
    }

    const std::string& Program::Desc::getShaderString(ShaderType shaderType) const
    {
        static std::string s;
        const auto& e = mEntryPoints[(uint32_t)shaderType];

        return e.isValid() ? mSources[e.index].str : s;
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
        for(auto src : mDesc.mSources)
        {
            int sourceIndex = sourceCounter++;

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

            for( auto entryPoint : mDesc.mEntryPoints )
            {
                if(entryPoint.index != sourceIndex) continue;
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
    
    bool Program::setDefines(const DefineList& dl)
    {
        if (dl != mDefineList)
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
        case ShaderType::RayGeneration: return SLANG_STAGE_RAY_GENERATION;
        case ShaderType::Intersection:  return SLANG_STAGE_INTERSECTION;
        case ShaderType::AnyHit:        return SLANG_STAGE_ANY_HIT;
        case ShaderType::ClosestHit:    return SLANG_STAGE_CLOSEST_HIT;
        case ShaderType::Miss:          return SLANG_STAGE_MISS;
        case ShaderType::Callable:      return SLANG_STAGE_CALLABLE;
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

    Program::VersionData Program::preprocessAndCreateProgramVersion(std::string& log) const
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
        for(auto shaderDefine : mDefineList)
        {
            spAddPreprocessorDefine(slangRequest, shaderDefine.first.c_str(), shaderDefine.second.c_str());
        }

        SlangCompileTarget slangTarget = SLANG_TARGET_UNKNOWN;
        const char* preprocessorDefine;
        // Pick the right target based on the current graphics API
#ifdef FALCOR_VK
        slangTarget = SLANG_SPIRV;
        preprocessorDefine = "FALCOR_VK";
#elif defined FALCOR_D3D12
        preprocessorDefine = "FALCOR_D3D";
        // If the profile string starts with a `4_` or a `5_`, use DXBC. Otherwise, use DXIL
        if (hasPrefix(mDesc.mShaderModel, "4_") || hasPrefix(mDesc.mShaderModel, "5_")) slangTarget = SLANG_DXBC;
        else                                                                            slangTarget = SLANG_DXIL;
#else
#error unknown shader compilation target
#endif
        spSetCodeGenTarget(slangRequest, slangTarget);
        spAddPreprocessorDefine(slangRequest, preprocessorDefine, "1");
        std::string sm = "__SM_" + mDesc.mShaderModel + "__";
        spAddPreprocessorDefine(slangRequest, sm.c_str(), "1");

        spSetTargetProfile(slangRequest, 0, spFindProfile(slangSession, getSlangProfileString(mDesc.mShaderModel).c_str()));

        // We always use row-major matrix layout (and when we invoke fxc/dxc we pass in the
        // appropriate flags to request this behavior), so we need to inform Slang that
        // this is what we want/expect so that it can compute correct reflection information.
        //
        spSetTargetMatrixLayoutMode(slangRequest, 0, SLANG_MATRIX_LAYOUT_ROW_MAJOR);

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

        spSetTargetFloatingPointMode(slangRequest, 0, slangFpMode);

        // Configure any flags for the Slang compilation step
        SlangCompileFlags slangFlags = 0;

        // Don't actually perform semantic checking: just pass through functions bodies to downstream compiler
        slangFlags |= SLANG_COMPILE_FLAG_NO_CHECKING | SLANG_COMPILE_FLAG_SPLIT_MIXED_TYPES;
        spSetCompileFlags(slangRequest, slangFlags);

        // Now lets add all our input shader code, one-by-one
        int translationUnitsAdded = 0;

        for(auto src : mDesc.mSources)
        {
            // Register the translation unit with Slang
            int translationUnitIndex = spAddTranslationUnit(slangRequest, SLANG_SOURCE_LANGUAGE_SLANG, nullptr);
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
                if (!findFileInDataDirectories(src.pLibrary->getFilename(), fullpath))
                {
                    logError(std::string("Can't find file ") + src.pLibrary->getFilename(), true);
                    return VersionData();
                }
                spAddTranslationUnitSourceFile(slangRequest, translationUnitIndex, fullpath.c_str());
            }
            else
            {
                assert(src.type == Desc::Source::Type::String);
                spAddTranslationUnitSourceString(slangRequest, translationUnitIndex, "", src.str.c_str());
            }
        }

        // Now we make a separate pass and add the entry points.
        // Each entry point references the index of the source
        // it uses, and luckily, the Slang API can use these
        // indices directly.
        for(uint32_t i = 0; i < kShaderCount; ++i)
        {
            auto& entryPoint = mDesc.mEntryPoints[i];

            // Skip unused entry points
            if(entryPoint.index < 0)
                continue;

            spAddEntryPoint(
                slangRequest,
                entryPoint.index,
                entryPoint.name.c_str(),
                getSlangStage(ShaderType(i)));
        }

        int anySlangErrors = spCompile(slangRequest);
        log += spGetDiagnosticOutput(slangRequest);
        if(anySlangErrors)
        {
            spDestroyCompileRequest(slangRequest);
            return VersionData();
        }

        // Extract the generated code for each stage
        int entryPointCounter = 0;
        Shader::Blob shaderBlob[kShaderCount];

        for (uint32_t i = 0; i < kShaderCount; i++)
        {
            auto& entryPoint = mDesc.mEntryPoints[i];
            // Skip unused entry points
            if(entryPoint.index < 0)
                continue;

            int entryPointIndex = entryPointCounter++;
            int targetIndex = 0; // We always compile for a single target

            spGetEntryPointCodeBlob(slangRequest, entryPointIndex, targetIndex, shaderBlob[i].writeRef());
        }

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

        spDestroyCompileRequest(slangRequest);

        // Now that we've preprocessed things, dispatch to the actual program creation logic,
        // which may vary in subclasses of `Program`
        programVersion.pVersion = createProgramVersion(log, shaderBlob, programVersion.reflectors);

        return programVersion;
    }

    ProgramVersion::SharedPtr Program::createProgramVersion(std::string& log, const Shader::Blob shaderBlob[kShaderCount], const ProgramReflectors& reflectors) const
    {
        // create the shaders
        Shader::SharedPtr shaders[kShaderCount] = {};
        for (uint32_t i = 0; i < kShaderCount; i++)
        {
            if (shaderBlob[i])
            { 
                shaders[i] = createShaderFromBlob(shaderBlob[i], ShaderType(i), mDesc.mEntryPoints[i].name, mDesc.getCompilerFlags(), log);
                if (!shaders[i]) return nullptr;
            }
        }

        if (shaders[(uint32_t)ShaderType::Compute])
        {
            return ProgramVersion::create(reflectors.pReflector, shaders[(uint32_t)ShaderType::Compute], log, getProgramDescString());
        }
        else
        {
            return  ProgramVersion::create(
                reflectors.pReflector,
                shaders[(uint32_t)ShaderType::Vertex],
                shaders[(uint32_t)ShaderType::Pixel],
                shaders[(uint32_t)ShaderType::Geometry],
                shaders[(uint32_t)ShaderType::Hull],
                shaders[(uint32_t)ShaderType::Domain],
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
