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
#include "DefineList.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/API/fwd.h"
#include "Core/API/Types.h"
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
 * Represents a single program entry point and its associated kernel code.
 *
 * In GFX, we do not generate actual shader code at program creation.
 * The actual shader code will only be generated and cached when all specialization arguments
 * are known, which is right before a draw/dispatch command is issued, and this is done
 * internally within GFX.
 * The `EntryPointKernel` implementation here serves as a helper utility for application code that
 * uses raw graphics API to get shader kernel code from an ordinary slang source.
 * Since most users/render-passes do not need to get shader kernel code, we defer
 * the call to slang's `getEntryPointCode` function until it is actually needed.
 * to avoid redundant shader compiler invocation.
 */
class FALCOR_API EntryPointKernel : public Object
{
    FALCOR_OBJECT(EntryPointKernel)
public:
    struct BlobData
    {
        const void* data;
        size_t size;
    };

    /**
     * Create a shader object
     * @param[in] linkedSlangEntryPoint The Slang IComponentType that defines the shader entry point.
     * @param[in] type The Type of the shader
     * @return If success, a new shader object, otherwise nullptr
     */
    static ref<EntryPointKernel> create(
        Slang::ComPtr<slang::IComponentType> linkedSlangEntryPoint,
        ShaderType type,
        const std::string& entryPointName
    )
    {
        return ref<EntryPointKernel>(new EntryPointKernel(linkedSlangEntryPoint, type, entryPointName));
    }

    /**
     * Get the shader Type
     */
    ShaderType getType() const { return mType; }

    /**
     * Get the name of the entry point.
     */
    const std::string& getEntryPointName() const { return mEntryPointName; }

    BlobData getBlobData() const
    {
        if (!mpBlob)
        {
            Slang::ComPtr<ISlangBlob> pDiagnostics;
            if (SLANG_FAILED(mLinkedSlangEntryPoint->getEntryPointCode(0, 0, mpBlob.writeRef(), pDiagnostics.writeRef())))
            {
                FALCOR_THROW(std::string("Shader compilation failed. \n") + (const char*)pDiagnostics->getBufferPointer());
            }
        }

        BlobData result;
        result.data = mpBlob->getBufferPointer();
        result.size = mpBlob->getBufferSize();
        return result;
    }

protected:
    EntryPointKernel(Slang::ComPtr<slang::IComponentType> linkedSlangEntryPoint, ShaderType type, const std::string& entryPointName)
        : mLinkedSlangEntryPoint(linkedSlangEntryPoint), mType(type), mEntryPointName(entryPointName)
    {}

    Slang::ComPtr<slang::IComponentType> mLinkedSlangEntryPoint;
    ShaderType mType;
    std::string mEntryPointName;
    mutable Slang::ComPtr<ISlangBlob> mpBlob;
};

/**
 * A collection of one or more entry points in a program kernels object.
 */
class FALCOR_API EntryPointGroupKernels : public Object
{
    FALCOR_OBJECT(EntryPointGroupKernels)
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

    static ref<const EntryPointGroupKernels> create(
        Type type,
        const std::vector<ref<EntryPointKernel>>& kernels,
        const std::string& exportName
    );

    virtual ~EntryPointGroupKernels() = default;

    Type getType() const { return mType; }
    const EntryPointKernel* getKernel(ShaderType type) const;
    const EntryPointKernel* getKernelByIndex(size_t index) const { return mKernels[index].get(); }
    const std::string& getExportName() const { return mExportName; }

protected:
    EntryPointGroupKernels(Type type, const std::vector<ref<EntryPointKernel>>& shaders, const std::string& exportName);
    EntryPointGroupKernels() = default;
    EntryPointGroupKernels(const EntryPointGroupKernels&) = delete;
    EntryPointGroupKernels& operator=(const EntryPointGroupKernels&) = delete;

    Type mType;
    std::vector<ref<EntryPointKernel>> mKernels;
    std::string mExportName;
};

/**
 * Low-level program object
 * This class abstracts the API's program creation and management
 */
class FALCOR_API ProgramKernels : public Object
{
    FALCOR_OBJECT(ProgramKernels)
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
    const EntryPointKernel* getKernel(ShaderType type) const;

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
    FALCOR_OBJECT(ProgramVersion)
public:
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
    const std::vector<Slang::ComPtr<slang::IComponentType>>& getSlangEntryPoints() const { return mpSlangEntryPoints; }

protected:
    friend class Program;
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
