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
#pragma once
#include "Program.h"
#include "Core/Macros.h"
#include "Core/State/StateGraph.h"
#include "Core/API/Raytracing.h"
#include "Core/API/RtStateObject.h"
#include <memory>
#include <string_view>
#include <string>
#include <vector>

namespace Falcor
{
    class RtProgramVars;

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
    class FALCOR_API RtProgram : public Program
    {
    public:
        using SharedPtr = std::shared_ptr<RtProgram>;
        using SharedConstPtr = std::shared_ptr<const RtProgram>;

        struct ShaderID
        {
            int32_t groupIndex = -1;    ///< Entry point group index.
            bool isValid() const { return groupIndex >= 0; }
        };

        /** Description of a raytracing program to be created.
        */
        class FALCOR_API Desc
        {
        public:
            /** Begin building a description, that initially has no source files or entry points.
            */
            Desc() { init(); }

            /** Begin building a description, based on a single path for source code.
                This is equivalent to: `Desc().addShaderLibrary(path)`
                \param[in] path Path to the source code.
            */
            explicit Desc(const std::string& path) : mBaseDesc(path) { init(); }

            /** Add a file of source code to use.
                This also sets the given file as the "active" source for subsequent entry points.
                \param[in] path Path to the source code.
            */
            Desc& addShaderLibrary(const std::string& path) { mBaseDesc.addShaderLibrary(path); return *this; }

            /** Add a string of source code to use.
                This also sets the given string as the "active" source for subsequent entry points.
                Note that the source string has to be added *before* any source that imports it.
                \param[in] shader Source code.
                \param[in] moduleName Slang module name. If not creating a new translation unit, this can be left empty.
                \param[in] modulePath Virtual file path to module created from string. This is just used for diagnostics purposes and can be left empty.
                \param[in] createTranslationUnit Whether a new Slang translation unit should be created, otherwise the source is added to the previous translation unit.
            */
            Desc& addShaderString(std::string_view shader, std::string_view moduleName, std::string_view modulePath = "", bool createTranslationUnit = true) { mBaseDesc.addShaderString(shader, moduleName, modulePath, createTranslationUnit); return *this; }

            /** Add a shader module.
                This also sets the given module as "active" for subsequent entry points.
                Note that the module has to be added *before* any module that imports it.
            */
            Desc& addShaderModule(const ShaderModule& module) { mBaseDesc.addShaderModule(module); return *this; }

            /** Add a list of shader modules.
                Note that the modules have to be added *before* any module that imports them.
            */
            Desc& addShaderModules(const ShaderModuleList& modules) { mBaseDesc.addShaderModules(modules); return *this; }

            /** Add a raygen shader.
                \param[in] raygen Entry point for the raygen shader.
                \param[in] typeConformances Optional list of type conformances for the raygen shader.
                \param[in] entryPointNameSuffix Optional suffix added to the entry point names in the generated code.
                \return Shader ID for raygen shader. This is used when building the binding table.
            */
            ShaderID addRayGen(const std::string& raygen, const TypeConformanceList& typeConformances = TypeConformanceList(), const std::string& entryPointNameSuffix = "");

            /** Add a miss shader.
                \param[in] miss Entry point for the miss shader.
                \param[in] typeConformances Optional list of type conformances for the miss shader.
                \param[in] entryPointNameSuffix Optional suffix added to the entry point names in the generated code.
                \return Shader ID for miss shader. This is used when building the binding table.
            */
            ShaderID addMiss(const std::string& miss, const TypeConformanceList& typeConformances = TypeConformanceList(), const std::string& entryPointNameSuffix = "");

            /** Add a hit group.
                A hit group consists of any combination of closest hit, any hit, and intersection shaders.
                Note that a hit group that contains an intersection shader only be used with procedural geometry.
                A hit group that does not contain an intersection shader can only be used with triangle geometry.
                It is valid to create a hit group entirely without entry points. Geometry using it will act
                as an occluder blocking miss shader exuection, but hits will not spawn any shader executions.
                \param[in] closestHit Entry point for the closest hit shader.
                \param[in] anyHit Entry point for the any hit shader.
                \param[in] intersection Entry point for the intersection shader.
                \param[in] typeConformances Optional list of type conformances for the hit group.
                \param[in] entryPointNameSuffix Optional suffix added to the entry point names in the generated code.
                \return Shader ID for hit group. This is used when building the binding table.
            */
            ShaderID addHitGroup(const std::string& closestHit, const std::string& anyHit = "", const std::string& intersection = "", const TypeConformanceList& typeConformances = TypeConformanceList(), const std::string& entryPointNameSuffix = "");

            /** Get the max recursion depth.
            */
            uint32_t getMaxTraceRecursionDepth() const { return mMaxTraceRecursionDepth; }

            /** Set the max recursion depth.
                \param[in] maxDepth The maximum ray recursion depth (0 = raygen).
            */
            Desc& setMaxTraceRecursionDepth(uint32_t maxDepth) { mMaxTraceRecursionDepth = maxDepth; return *this; }

            /** Get the max payload size.
            */
            uint32_t getMaxPayloadSize() const { return mMaxPayloadSize; }

            /** Set the max payload size.
                \param[in] maxPayloadSize The maximum ray payload size in bytes.
            */
            Desc& setMaxPayloadSize(uint32_t maxPayloadSize) { mMaxPayloadSize = maxPayloadSize; return *this; }

            /** Get the max attribute size.
            */
            uint32_t getMaxAttributeSize() const { return mMaxAttributeSize; }

            /** Set the max attribute size.
                \param[in] maxAttributeSize The maximum attribute size in bytes.
            */
            Desc& setMaxAttributeSize(uint32_t maxAttributeSize) { mMaxAttributeSize = maxAttributeSize; return *this; }

            /** Get raytracing pipeline flags.
            */
            RtPipelineFlags getPipelineFlags() const { return mPipelineFlags; }

            /** Set raytracing pipeline flags.
                These flags are added to any TraceRay() call within this pipeline, and may be used to
                optimize the pipeline for particular primitives types. Requires Tier 1.1 support.
                \param[in] flags Pipeline flags.
            */
            Desc& setPipelineFlags(RtPipelineFlags flags) { mPipelineFlags = flags; return *this; }

            /** Add a list of type conformances.
                The type conformances are linked into all shaders in the program.
                \param[in] typeConformances List of type conformances.
            */
            Desc& addTypeConformances(const TypeConformanceList& typeConformances) { mBaseDesc.addTypeConformances(typeConformances); return *this; }

            /** Enable/disable treat-warnings-as-error compilation flag.
            */
            Desc& warningsAsErrors(bool enable) { mBaseDesc.warningsAsErrors(enable); return *this; }

            /** Enable/disable pre-processed shader dump.
            */
            Desc& dumpIntermediates(bool enable) { mBaseDesc.dumpIntermediates(enable); return *this; }

            /** Set the shader model. The default is SM 6.5 for DXR Tier 1.1 support.
            */
            Desc& setShaderModel(const std::string& sm) { mBaseDesc.setShaderModel(sm); return *this; };

            /** Get the compiler flags.
            */
            Shader::CompilerFlags getCompilerFlags() const { return mBaseDesc.getCompilerFlags(); }

            /** Set the compiler flags. Replaces any previously set flags.
            */
            Desc& setCompilerFlags(Shader::CompilerFlags flags) {mBaseDesc.setCompilerFlags(flags); return *this; }

            /** Get additional compiler arguments.
            */
            const ArgumentList& getCompilerArguments() const { return mBaseDesc.getCompilerArguments(); }

            /** Set additional compiler arguments. Replaces any previously set arguments.
            */
            Desc& setCompilerArguments(const ArgumentList& arguments) { mBaseDesc.setCompilerArguments(arguments); return *this; }

        private:
            friend class RtProgram;

            void init();

            Program::Desc mBaseDesc;
            uint32_t mRayGenCount = 0;

            // These parameters impact performance and must be explicitly set.
            uint32_t mMaxTraceRecursionDepth = -1;
            uint32_t mMaxPayloadSize = -1;
            uint32_t mMaxAttributeSize = getRaytracingMaxAttributeSize();
            RtPipelineFlags mPipelineFlags = RtPipelineFlags::None;
        };

        /** Create a new ray tracing program.
            \param[in] desc The program description.
            \param[in] programDefines Optional list of macro definitions to set into the program.
            \return A new object, or an exception is thrown if creation failed.
        */
        static RtProgram::SharedPtr create(Desc desc, const DefineList& programDefines = DefineList());

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

        RtProgram(const Desc& desc, const DefineList& programDefines);

        Desc mRtDesc;

        using StateGraph = Falcor::StateGraph<RtStateObject::SharedPtr, void*>;
        StateGraph mRtsoGraph;
    };
}
