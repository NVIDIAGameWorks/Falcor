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
#include "Core/Error.h"
#include "Core/ObjectPython.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include "Core/API/ParameterBlock.h"
#include "Core/API/RtStateObject.h"
#include "Core/API/PythonHelpers.h"
#include "Utils/StringUtils.h"
#include "Utils/Logger.h"
#include "Utils/Timing/CpuTimer.h"
#include "Utils/CryptoUtils.h"
#include "Utils/Scripting/ScriptBindings.h"

#include <slang.h>

#include <set>

namespace Falcor
{

void ProgramDesc::finalize()
{
    uint32_t globalIndex = 0;
    for (auto& entryPointGroup : entryPointGroups)
        for (auto& entryPoint : entryPointGroup.entryPoints)
            entryPoint.globalIndex = globalIndex++;
}

Program::Program(ref<Device> pDevice, ProgramDesc desc, DefineList defineList)
    : mpDevice(std::move(pDevice)), mDesc(std::move(desc)), mDefineList(std::move(defineList)), mTypeConformanceList(mDesc.typeConformances)
{
    mDesc.finalize();

    // If not shader model was requested, use the default shader model for the device.
    if (mDesc.shaderModel == ShaderModel::Unknown)
        mDesc.shaderModel = mpDevice->getDefaultShaderModel();

    // Check that shader model is supported on the device.
    if (!mpDevice->isShaderModelSupported(mDesc.shaderModel))
        FALCOR_THROW("Requested Shader Model {} is not supported by the device", enumToString(mDesc.shaderModel));

    if (mDesc.hasEntryPoint(ShaderType::RayGeneration))
    {
        if (desc.maxTraceRecursionDepth == uint32_t(-1))
            FALCOR_THROW("Can't create a raytacing program without specifying maximum trace recursion depth");
        if (desc.maxPayloadSize == uint32_t(-1))
            FALCOR_THROW("Can't create a raytacing program without specifying maximum ray payload size");
    }

    validateEntryPoints();

    mpDevice->getProgramManager()->registerProgramForReload(this);
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
    for (const auto& group : mDesc.entryPointGroups)
    {
        for (const auto& e : group.entryPoints)
        {
            if (!entryPointNamesAndTypes.insert(NameTypePair(e.exportName, e.type)).second)
            {
                logWarning("Duplicate program entry points '{}' of type '{}'.", e.exportName, e.type);
            }
        }
    }
}

std::string Program::getProgramDescString() const
{
    std::string desc;

    for (const auto& shaderModule : mDesc.shaderModules)
    {
        for (const auto& source : shaderModule.sources)
        {
            switch (source.type)
            {
            case ProgramDesc::ShaderSource::Type::File:
                desc += source.path.string();
                break;
            case ProgramDesc::ShaderSource::Type::String:
                desc += "<string>";
                break;
            default:
                FALCOR_UNREACHABLE();
            }
            desc += " ";
        }
    }

    desc += "(";
    size_t entryPointIndex = 0;
    for (const auto& entryPointGroup : mDesc.entryPointGroups)
    {
        for (const auto& entryPoint : entryPointGroup.entryPoints)
        {
            if (entryPointIndex++ > 0)
                desc += ", ";
            desc += entryPoint.exportName;
        }
    }
    desc += ")";

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
                FALCOR_THROW("Program linkage failed");
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
            std::string msg = "Failed to link program:\n" + getProgramDescString() + "\n\n" + log;
            bool showMessageBox = is_set(getErrorDiagnosticFlags(), ErrorDiagnosticFlags::ShowMessageBoxOnError);
            if (showMessageBox && reportErrorAndAllowRetry(msg))
                continue;
            FALCOR_THROW(msg);
        }
        else
        {
            if (!log.empty())
            {
                logWarning("Warnings in program:\n{}\n{}", getProgramDescString(), log);
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

ref<RtStateObject> Program::getRtso(RtProgramVars* pVars)
{
    auto pProgramVersion = getActiveVersion();
    auto pProgramKernels = pProgramVersion->getKernels(mpDevice, pVars);

    mRtsoGraph.walk((void*)pProgramKernels.get());

    ref<RtStateObject> pRtso = mRtsoGraph.getCurrentNode();

    if (pRtso == nullptr)
    {
        RtStateObjectDesc desc;
        desc.pProgramKernels = pProgramKernels;
        desc.maxTraceRecursionDepth = mDesc.maxTraceRecursionDepth;
        desc.pipelineFlags = mDesc.rtPipelineFlags;

        StateGraph::CompareFunc cmpFunc = [&desc](ref<RtStateObject> pRtso) -> bool { return pRtso && (desc == pRtso->getDesc()); };

        if (mRtsoGraph.scanForMatchingNode(cmpFunc))
        {
            pRtso = mRtsoGraph.getCurrentNode();
        }
        else
        {
            pRtso = mpDevice->createRtStateObject(desc);
            mRtsoGraph.setCurrentNodeData(pRtso);
        }
    }

    return pRtso;
}

FALCOR_SCRIPT_BINDING(Program)
{
    using namespace pybind11::literals;

    FALCOR_SCRIPT_BINDING_DEPENDENCY(Types)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(ProgramReflection)

    pybind11::enum_<SlangCompilerFlags> slangCompilerFlags(m, "SlangCompilerFlags");
    slangCompilerFlags.value("None_", SlangCompilerFlags::None);
    slangCompilerFlags.value("TreatWarningsAsErrors", SlangCompilerFlags::TreatWarningsAsErrors);
    slangCompilerFlags.value("DumpIntermediates", SlangCompilerFlags::DumpIntermediates);
    slangCompilerFlags.value("FloatingPointModeFast", SlangCompilerFlags::FloatingPointModeFast);
    slangCompilerFlags.value("FloatingPointModePrecise", SlangCompilerFlags::FloatingPointModePrecise);
    slangCompilerFlags.value("GenerateDebugInfo", SlangCompilerFlags::GenerateDebugInfo);
    slangCompilerFlags.value("MatrixLayoutColumnMajor", SlangCompilerFlags::MatrixLayoutColumnMajor);
    ScriptBindings::addEnumBinaryOperators(slangCompilerFlags);

    pybind11::class_<ProgramDesc> programDesc(m, "ProgramDesc");

    pybind11::class_<ProgramDesc::ShaderModule>(programDesc, "ShaderModule")
        .def("add_file", &ProgramDesc::ShaderModule::addFile, "path"_a)
        .def("add_string", &ProgramDesc::ShaderModule::addString, "string"_a, "path"_a = std::filesystem::path());

    pybind11::class_<ProgramDesc::EntryPointGroup>(programDesc, "EntryPointGroup")
        .def_property(
            "type_conformances",
            [](const ProgramDesc::EntryPointGroup& self) { return typeConformanceListToPython(self.typeConformances); },
            [](ProgramDesc::EntryPointGroup& self, const pybind11::dict& dict)
            { return self.typeConformances = typeConformanceListFromPython(dict); }
        );

    programDesc.def(pybind11::init<>());
    programDesc.def_readwrite("shader_model", &ProgramDesc::shaderModel);
    programDesc.def_readwrite("compiler_flags", &ProgramDesc::compilerFlags);
    programDesc.def_readwrite("compiler_arguments", &ProgramDesc::compilerArguments);
    programDesc.def(
        "add_shader_module",
        pybind11::overload_cast<std::string>(&ProgramDesc::addShaderModule),
        "name"_a = "",
        pybind11::return_value_policy::reference
    );
    programDesc.def("cs_entry", &ProgramDesc::csEntry, "name"_a);

    pybind11::class_<Program, ref<Program>> program(m, "Program");

    program.def_property_readonly("reflector", &Program::getReflector);
    program.def_property(
        "defines",
        [](const Program& self) { return defineListToPython(self.getDefines()); },
        [](Program& self, const pybind11::dict& dict) { self.setDefines(defineListFromPython(dict)); }
    );
    program.def("add_define", &Program::addDefine, "name"_a, "value"_a = "");
    program.def("remove_define", &Program::removeDefine, "name"_a);
    program.def_property(
        "type_conformances",
        [](const Program& self) { return typeConformanceListToPython(self.getTypeConformances()); },
        [](Program& self, const pybind11::dict& dict) { return self.setTypeConformances(typeConformanceListFromPython(dict)); }
    );
    program.def("add_type_conformance", &Program::addTypeConformance, "type_name"_a, "interface_type"_a, "id"_a);
    program.def("remove_type_conformance", &Program::removeTypeConformance, "type_name"_a, "interface_type"_a);
}

} // namespace Falcor
