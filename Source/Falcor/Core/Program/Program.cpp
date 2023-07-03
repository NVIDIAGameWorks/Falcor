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
#include "Program.h"
#include "ProgramManager.h"
#include "ProgramVars.h"
#include "Core/ObjectPython.h"
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
const std::string kSupportedShaderModels[] = {
    "6_0", "6_1", "6_2", "6_3", "6_4", "6_5",
#if FALCOR_HAS_D3D12_AGILITY_SDK
    "6_6",
#endif
};

Program::Desc::Desc() = default;

Program::Desc::Desc(const std::filesystem::path& path)
{
    addShaderLibrary(path);
}

Program::Desc& Program::Desc::setLanguagePrelude(const std::string_view prelude)
{
    mLanguagePrelude = prelude;
    return *this;
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

Program::Desc& Program::Desc::entryPoint(ShaderType shaderType, const std::string& name)
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

Program::Program(ref<Device> pDevice, const Desc& desc, const DefineList& defineList)
    : mpDevice(pDevice), mDesc(desc), mDefineList(defineList), mTypeConformanceList(desc.mTypeConformances)
{
    mpDevice->getProgramManager()->registerProgramForReload(this);
    validateEntryPoints();
}

Program::~Program()
{
    mpDevice->getProgramManager()->unregisterProgramForReload(this);

    // Invalidate program versions.
    for (auto& version : mProgramVersions)
        version.second->mpProgram = nullptr;
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

std::string Program::getProgramDescString() const
{
    std::string desc;

    int32_t groupCount = (int32_t)mDesc.mGroups.size();

    for (size_t i = 0; i < mDesc.mSources.size(); i++)
    {
        const auto& src = mDesc.mSources[i];
        if (i != 0)
            desc += " ";
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

            if (ee != 0)
                desc += ", ";
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
    TypeConformance conformance = TypeConformance(typeName, interfaceType);
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
    TypeConformance conformance = TypeConformance(typeName, interfaceType);
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

const ref<const ProgramVersion>& Program::getActiveVersion() const
{
    if (mLinkRequired)
    {
        const auto& it = mProgramVersions.find(ProgramVersionKey{mDefineList, mTypeConformanceList});
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
                mProgramVersions[ProgramVersionKey{mDefineList, mTypeConformanceList}] = mpActiveVersion;
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

bool Program::link() const
{
    while (1)
    {
        // Create the program
        std::string log;
        auto pVersion = mpDevice->getProgramManager()->createProgramVersion(*this, log);

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

void Program::breakStrongReferenceToDevice()
{
    mpDevice.breakStrongReference();
}

FALCOR_SCRIPT_BINDING(Program)
{
    pybind11::class_<Program, ref<Program>>(m, "Program");
}
} // namespace Falcor
