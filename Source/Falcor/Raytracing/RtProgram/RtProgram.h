/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/Program/Program.h"
#include "Core/API/RootSignature.h"
#include "Raytracing/RtStateObject.h"
#include "Raytracing/ShaderTable.h"

namespace Falcor
{
    /** Ray tracing program. See GraphicsProgram and ComputeProgram to manage other types of programs.

        A ray tracing program consists of one or more raygen shaders, and optionally miss shaders and
        hit groups. Each hit group can consist of a closest hit, any hit, and/or intersection shaders.

        The ray tracing program just defines what shaders exist and is not tied to a particular scene.
        A separate RtBindingTable object describes which raygen and miss shaders to run, as well as
        the mapping from scene geometries to hit groups. Not all miss shaders and hit groups need to be
        assigned. Empty entries will be ignored and not trigger a shader execution upon miss/hit.

        The user is responsible for creating the program and one or more associated binding tables.
        Note that the same program can be reused with different binding tables.
        For example, the ray tracing program can define the set of all possible entry points,
        and then depending on scene contents, the binding table picks out the ones it needs.
    */
    class dlldecl RtProgram : public Program
    {
    public:
        using SharedPtr = std::shared_ptr<RtProgram>;
        using SharedConstPtr = std::shared_ptr<const RtProgram>;

        using DefineList = Program::DefineList;

        struct ShaderID
        {
            int32_t groupIndex = -1;    ///< Entry point group index.
            bool isValid() const { return groupIndex >= 0; }
        };

        /** Description of a raytracing program to be created.
        */
        class dlldecl Desc
        {
        public:
            Desc() { init(); }
            Desc(const std::string& filename) : mBaseDesc(filename) { init(); }

            /** Add a file of source code to use.
                This also sets the given file as the "active" source for subsequent entry points.
                \param[in] filename Path to the source code.
            */
            Desc& addShaderLibrary(const std::string& filename);

            /** Add a raygen shader.
                \param[in] raygen Entry point for the raygen shader.
                \return Shader ID for raygen shader. This is used when building the binding table.
            */
            ShaderID addRayGen(const std::string& raygen);

            /** Add a miss shader.
                \param[in] miss Entry point for the miss shader.
                \return Shader ID for miss shader. This is used when building the binding table.
            */
            ShaderID addMiss(const std::string& miss);

            /** Add a hit group.
                A hit group consists of any combination of closest hit, any hit, and intersection shaders.
                Note that a hit group that contains an intersection shader only be used with procedural geometry.
                A hit group that does not contain an intersection shader can only be used with triangle geometry.
                It is valid to create a hit group entirely without entry points. Geometry using it will act
                as an occluder blocking miss shader exuection, but hits will not spawn any shader executions.
                \param[in] closestHit Entry point for the closest hit shader.
                \param[in] anyHit Entry point for the any hit shader.
                \param[in] intersection Entry point for the intersection shader.
                \return Shader ID for hit group. This is used when building the binding table.
            */
            ShaderID addHitGroup(const std::string& closestHit, const std::string& anyHit = "", const std::string& intersection = "");

            /** Set the max recursion depth.
                \param[in] maxDepth The maximum ray recursion depth (0 = raygen).
            */
            void setMaxTraceRecursionDepth(uint32_t maxDepth) { mMaxTraceRecursionDepth = maxDepth; }

            /** Set the max payload size.
                \param[in] maxPayloadSize The maximum ray payload size in bytes.
            */
            void setMaxPayloadSize(uint32_t maxPayloadSize) { mMaxPayloadSize = maxPayloadSize; }

            /** Set the max attribute size.
                \param[in] maxAttributeSize The maximum attribute size in bytes.
            */
            void setMaxAttributeSize(uint32_t maxAttributeSize) { mMaxAttributeSize = maxAttributeSize; }

            /** Set raytracing pipeline flags.
                These flags are added to any TraceRay() call within this pipeline, and may be used to
                optimize the pipeline for particular primitives types. Requires Tier 1.1 support.
                \param[in] flags Pipeline flags.
            */
            void setPipelineFlags(D3D12_RAYTRACING_PIPELINE_FLAGS flags) { mPipelineFlags = flags; }

            /** Set the compiler flags. Replaces any previously set flags.
            */
            Desc& setCompilerFlags(Shader::CompilerFlags flags) { mBaseDesc.setCompilerFlags(flags); return *this; }

            /** Set the shader model. The default is SM 6.5 for DXR Tier 1.1 support.
            */
            Desc& setShaderModel(const std::string& sm) { mBaseDesc.setShaderModel(sm); return *this; };

            /** Add a macro definition. The definition is available to all shaders in the program.
                \param[in] define Name of macro definition.
                \param[in] value Value of macro definition.
            */
            Desc& addDefine(const std::string& define, const std::string& value);

            /** Add a list of macro definitions. The definitions are available to all shaders in the program.
                \param[in] defines List of macro defintitions, each consists of a name and a value.
            */
            Desc& addDefines(const DefineList& defines);

            /** Get the max recursion depth.
            */
            uint32_t getMaxTraceRecursionDepth() const { return mMaxTraceRecursionDepth; }

            /** Get the max payload size.
            */
            uint32_t getMaxPayloadSize() const { return mMaxPayloadSize; }

            /** Get the max attribute size.
            */
            uint32_t getMaxAttributeSize() const { return mMaxAttributeSize; }

            /** Get raytracing pipeline flags.
            */
            D3D12_RAYTRACING_PIPELINE_FLAGS getPipelineFlags() const { return mPipelineFlags; }

        private:
            friend class RtProgram;

            void init();

            Program::Desc mBaseDesc;
            DefineList mDefineList;
            uint32_t mRayGenCount = 0;

            // These parameters impact performance and must be explicitly set.
            uint32_t mMaxTraceRecursionDepth = -1;
            uint32_t mMaxPayloadSize = -1;
            uint32_t mMaxAttributeSize = D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES;
            D3D12_RAYTRACING_PIPELINE_FLAGS mPipelineFlags = D3D12_RAYTRACING_PIPELINE_FLAG_NONE;
        };

        /** Create a new ray tracing program.
            \param[in] desc The program description.
            \return A new object, or an exception is thrown if creation failed.
        */
        static RtProgram::SharedPtr create(Desc desc);

        /** Get the raytracing state object for this program.
        */
        RtStateObject::SharedPtr getRtso(RtProgramVars* pVars);

        Desc const& getRtDesc() const { return mRtDesc; }

    protected:
        EntryPointGroupKernels::SharedPtr createEntryPointGroupKernels(
            const std::vector<Shader::SharedPtr>& shaders,
            EntryPointGroupReflection::SharedPtr const& pReflector) const override;

    private:
        RtProgram(RtProgram const&) = delete;
        RtProgram& operator=(RtProgram const&) = delete;

        RtProgram(const Desc& desc);

        Desc mRtDesc;

        using StateGraph = Falcor::StateGraph<RtStateObject::SharedPtr, void*>;
        StateGraph mRtsoGraph;
    };
}
