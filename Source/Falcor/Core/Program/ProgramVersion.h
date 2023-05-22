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
#pragma once
#include "ProgramReflection.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/API/fwd.h"
#include "Core/API/Shader.h"
#include "Core/API/Handles.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <slang.h>

namespace Falcor
{
class FALCOR_API Program;
class FALCOR_API ProgramVars;
class FALCOR_API ProgramVersion;

/**
 * A collection of one or more entry points in a program kernels object.
 */
class FALCOR_API EntryPointGroupKernels : public Object
{
public:
    /**
     * Types of entry point groups.
     */
    enum class Type
    {
        Compute,        ///< A group consisting of a single compute kernel
        Rasterization,  ///< A group consisting of rasterization shaders to be used together as a pipeline.
        RtSingleShader, ///< A group consisting of a single ray tracing shader
        RtHitGroup,     ///< A ray tracing "hit group"
    };

    using Shaders = std::vector<ref<Shader>>;

    static ref<const EntryPointGroupKernels> create(Type type, const Shaders& shaders, const std::string& exportName);

    virtual ~EntryPointGroupKernels() = default;

    Type getType() const { return mType; }
    const Shader* getShader(ShaderType type) const;
    const Shader* getShaderByIndex(int32_t index) const { return mShaders[index].get(); }
    const std::string& getExportName() const { return mExportName; }

protected:
    EntryPointGroupKernels(Type type, const Shaders& shaders, const std::string& exportName);
    EntryPointGroupKernels() = default;
    EntryPointGroupKernels(const EntryPointGroupKernels&) = delete;
    EntryPointGroupKernels& operator=(const EntryPointGroupKernels&) = delete;

    Type mType;
    Shaders mShaders;
    std::string mExportName;
};

/**
 * Low-level program object
 * This class abstracts the API's program creation and management
 */
class FALCOR_API ProgramKernels : public Object
{
public:
    typedef std::vector<ref<const EntryPointGroupKernels>> UniqueEntryPointGroups;

    /**
     * Create a new program object for graphics.
     * @param[in] The program reflection object
     * @param[in] pVS Vertex shader object
     * @param[in] pPS Fragment shader object
     * @param[in] pGS Geometry shader object
     * @param[in] pHS Hull shader object
     * @param[in] pDS Domain shader object
     * @param[out] Log In case of error, this will contain the error log string
     * @param[in] DebugName Optional. A meaningful name to use with log messages
     * @return New object in case of success, otherwise nullptr
     */
    static ref<ProgramKernels> create(
        Device* pDevice,
        const ProgramVersion* pVersion,
        slang::IComponentType* pSpecializedSlangGlobalScope,
        const std::vector<slang::IComponentType*>& pTypeConformanceSpecializedEntryPoints,
        const ref<const ProgramReflection>& pReflector,
        const UniqueEntryPointGroups& uniqueEntryPointGroups,
        std::string& log,
        const std::string& name = ""
    );

    virtual ~ProgramKernels() = default;

    /**
     * Get an attached shader object, or nullptr if no shader is attached to the slot.
     */
    const Shader* getShader(ShaderType type) const;

    /**
     * Get the program name
     */
    const std::string& getName() const { return mName; }

    /**
     * Get the reflection object
     */
    const ref<const ProgramReflection>& getReflector() const { return mpReflector; }

    ProgramVersion const* getProgramVersion() const { return mpVersion; }

    const UniqueEntryPointGroups& getUniqueEntryPointGroups() const { return mUniqueEntryPointGroups; }

    const ref<const EntryPointGroupKernels>& getUniqueEntryPointGroup(uint32_t index) const { return mUniqueEntryPointGroups[index]; }

    gfx::IShaderProgram* getGfxProgram() const { return mGfxProgram; }

protected:
    ProgramKernels(
        const ProgramVersion* pVersion,
        const ref<const ProgramReflection>& pReflector,
        const UniqueEntryPointGroups& uniqueEntryPointGroups,
        const std::string& name = ""
    );

    Slang::ComPtr<gfx::IShaderProgram> mGfxProgram;
    const std::string mName;

    UniqueEntryPointGroups mUniqueEntryPointGroups;

    void* mpPrivateData;
    const ref<const ProgramReflection> mpReflector;

    ProgramVersion const* mpVersion = nullptr;
};

class ProgramVersion : public Object
{
public:
    using DefineList = Shader::DefineList;

    /**
     * Get the program that this version was created from
     */
    Program* getProgram() const { return mpProgram; }

    /**
     * Get the defines that were used to create this version
     */
    const DefineList& getDefines() const { return mDefines; }

    /**
     * Get the program name
     */
    const std::string& getName() const { return mName; }

    /**
     * Get the reflection object.
     * @return A program reflection object.
     */
    const ref<const ProgramReflection>& getReflector() const
    {
        FALCOR_ASSERT(mpReflector);
        return mpReflector;
    }

    /**
     * Get executable kernels based on state in a `ProgramVars`
     */
    // TODO @skallweit passing pDevice here is a bit of a WAR
    ref<const ProgramKernels> getKernels(Device* pDevice, ProgramVars const* pVars) const;

    slang::ISession* getSlangSession() const;
    slang::IComponentType* getSlangGlobalScope() const;
    slang::IComponentType* getSlangEntryPoint(uint32_t index) const;

protected:
    friend class Program;
    friend class RtProgram;
    friend class ProgramManager;

    static ref<ProgramVersion> createEmpty(Program* pProgram, slang::IComponentType* pSlangGlobalScope);

    ProgramVersion(Program* pProgram, slang::IComponentType* pSlangGlobalScope);

    void init(
        const DefineList& defineList,
        const ref<const ProgramReflection>& pReflector,
        const std::string& name,
        const std::vector<Slang::ComPtr<slang::IComponentType>>& pSlangEntryPoints
    );

    mutable Program* mpProgram;
    DefineList mDefines;
    ref<const ProgramReflection> mpReflector;
    std::string mName;
    Slang::ComPtr<slang::IComponentType> mpSlangGlobalScope;
    std::vector<Slang::ComPtr<slang::IComponentType>> mpSlangEntryPoints;

    // Cached version of compiled kernels for this program version
    mutable std::unordered_map<std::string, ref<const ProgramKernels>> mpKernels;
};
} // namespace Falcor
