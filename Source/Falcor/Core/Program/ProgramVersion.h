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
#pragma once
#include "Core/Program/ProgramReflection.h"
#include "Core/API/Shader.h"
#include "Core/API/RootSignature.h"

#include <slang/slang.h>

namespace Falcor
{
    class dlldecl Program;
    class dlldecl ProgramVars;
    class dlldecl ProgramVersion;

#if 0
    /** A collection of one or more entry points in a program kernels object.
    */
    class dlldecl EntryPointGroup
    {
    public:
        using SharedPtr = std::shared_ptr<EntryPointGroup>;
        using SharedConstPtr = std::shared_ptr<const EntryPointGroup>;

        using Type = EntryPointGroupReflection::Type;

        virtual ~EntryPointGroup() = default;

        Type getType() const { return mType; }

    protected:
        EntryPointGroup() = default;
        EntryPointGroup(const EntryPointGroup&) = delete;
        EntryPointGroup& operator=(const EntryPointGroup&) = delete;

        Type mType;
    };
#endif

    /** A collection of one or more entry points in a program kernels object.
    */
    class dlldecl EntryPointGroupKernels
    {
    public:
        using SharedPtr = std::shared_ptr<EntryPointGroupKernels>;
        using SharedConstPtr = std::shared_ptr<const EntryPointGroupKernels>;

        /** Types of entry point groups.
        */
        enum class Type
        {
            Compute,            ///< A group consisting of a single compute kernel
            Rasterization,      ///< A group consisting of rasterization shaders to be used together as a pipeline.
            RtSingleShader,     ///< A group consisting of a single ray tracing shader
            RtHitGroup,         ///< A ray tracing "hit group"
        };

        using Shaders = std::vector<Shader::SharedPtr>;

        static SharedPtr create(Type type, const Shaders& shaders);

        virtual ~EntryPointGroupKernels() = default;

        Type getType() const { return mType; }
        const Shader* getShader(ShaderType type) const;
        const Shader* getShaderByIndex(int32_t index) const { return mShaders[index].get(); }

    protected:
        EntryPointGroupKernels(Type type, const Shaders& shaders);
        EntryPointGroupKernels() = default;
        EntryPointGroupKernels(const EntryPointGroupKernels&) = delete;
        EntryPointGroupKernels& operator=(const EntryPointGroupKernels&) = delete;

        Type mType;
        Shaders mShaders;
    };

    class dlldecl RtEntryPointGroupKernels : public EntryPointGroupKernels
    {
    public:
        static SharedPtr create(
            Type type,
            const Shaders& shaders,
            std::string const& exportName,
            RootSignature::SharedPtr const& localRootSignature,
            uint32_t maxPayloadSize,
            uint32_t maxAttributeSize);

        const std::string& getExportName() const { return mExportName; }

        const RootSignature::SharedPtr& getLocalRootSignature() const { return mLocalRootSignature; }

        uint32_t getMaxPayloadSize() const { return mMaxPayloadSize; }
        uint32_t getMaxAttributesSize() const { return mMaxAttributesSize; }

    protected:
        RtEntryPointGroupKernels(
            Type type,
            const Shaders& shaders,
            std::string const& exportName,
            RootSignature::SharedPtr const& localRootSignature,
            uint32_t maxPayloadSize,
            uint32_t maxAttributeSize);

        std::string mExportName;
        RootSignature::SharedPtr mLocalRootSignature;
        uint32_t mMaxPayloadSize;
        uint32_t mMaxAttributesSize;
    };

    /** Low-level program object
        This class abstracts the API's program creation and management
    */
    class dlldecl ProgramKernels : public std::enable_shared_from_this<ProgramKernels>
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramKernels>;
        using SharedConstPtr = std::shared_ptr<const ProgramKernels>;

        typedef std::vector<EntryPointGroupKernels::SharedPtr> UniqueEntryPointGroups;

        /** Create a new program object for graphics.
            \param[in] The program reflection object
            \param[in] pVS Vertex shader object
            \param[in] pPS Fragment shader object
            \param[in] pGS Geometry shader object
            \param[in] pHS Hull shader object
            \param[in] pDS Domain shader object
            \param[out] Log In case of error, this will contain the error log string
            \param[in] DebugName Optional. A meaningful name to use with log messages
            \return New object in case of success, otherwise nullptr
        */
        static SharedPtr create(
            const ProgramVersion* pVersion,
            const ProgramReflection::SharedPtr& pReflector,
            const UniqueEntryPointGroups& uniqueEntryPointGroups,
            std::string& log,
            const std::string& name = "");

        virtual ~ProgramKernels() = default;

        /** Get an attached shader object, or nullptr if no shader is attached to the slot.
        */
        const Shader* getShader(ShaderType type) const;

        /** Get the program name
        */
        const std::string& getName() const {return mName;}

        /** Get the reflection object
        */
        const ProgramReflection::SharedPtr& getReflector() const { return mpReflector; }

        RootSignature::SharedPtr const& getRootSignature() const { return mpRootSignature; }

        std::shared_ptr<const ProgramVersion> getProgramVersion() const;

        const UniqueEntryPointGroups& getUniqueEntryPointGroups() const { return mUniqueEntryPointGroups; }

        const EntryPointGroupKernels::SharedPtr& getUniqueEntryPointGroup(uint32_t index) const { return mUniqueEntryPointGroups[index]; }

    protected:
        ProgramKernels(
            const ProgramVersion* pVersion,
            const ProgramReflection::SharedPtr& pReflector,
            const UniqueEntryPointGroups& uniqueEntryPointGroups,
            const std::string& name = "");

        ProgramHandle mApiHandle = ProgramHandle();
        const std::string mName;

        UniqueEntryPointGroups mUniqueEntryPointGroups;

        void* mpPrivateData;
        const ProgramReflection::SharedPtr mpReflector;

        ProgramVersion const* mpVersion = nullptr;
        RootSignature::SharedPtr mpRootSignature;
    };

    class ProgramVersion : public std::enable_shared_from_this<ProgramVersion>
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramVersion>;
        using SharedConstPtr = std::shared_ptr<const ProgramVersion>;
        using DefineList = Shader::DefineList;

        /** Get the program that this version was created from
        */
        std::shared_ptr<Program> getProgram() const { return mpProgram; }

        /** Get the defines that were used to create this version
        */
        DefineList const& getDefines() const { return mDefines; }

        /** Get the program name
        */
        const std::string& getName() const { return mName; }

        /** Get the reflection object.
            \return A program reflection object.
        */
        const ProgramReflection::SharedPtr& getReflector() const { assert(mpReflector); return mpReflector; }

        /** Get executable kernels based on state in a `ProgramVars`
        */
        ProgramKernels::SharedConstPtr getKernels(ProgramVars const* pVars) const;

        slang::ISession* getSlangSession() const;
        slang::IComponentType* getSlangGlobalScope() const;
        slang::IComponentType* getSlangEntryPoint(uint32_t index) const;

    protected:
        friend class Program;
        friend class RtProgram;

        static SharedPtr createEmpty(Program* pProgram, slang::IComponentType* pSlangGlobalScope);

        ProgramVersion(Program* pProgram, slang::IComponentType* pSlangGlobalScope);

        void init(
            const DefineList&                                   defineList,
            const ProgramReflection::SharedPtr&                 pReflector,
            const std::string&                                  name,
            std::vector<ComPtr<slang::IComponentType>> const&   pSlangEntryPoints);

        std::shared_ptr<Program>        mpProgram;
        DefineList                      mDefines;
        ProgramReflection::SharedPtr    mpReflector;
        std::string                     mName;
        ComPtr<slang::IComponentType>   mpSlangGlobalScope;
        std::vector<ComPtr<slang::IComponentType>> mpSlangEntryPoints;

        // Cached version of compiled kernels for this program version
        mutable std::unordered_map<std::string, ProgramKernels::SharedPtr> mpKernels;
    };
}
