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
#pragma once
#include "Framework.h"
#include <string>
#include <map>
#include <vector>
#include "API/Shader.h"
#include "Graphics/Program/ProgramReflection.h"
#include "API/LowLevel/RootSignature.h"
#include "Utils/Graph.h"

namespace Falcor
{
    class ConstantBuffer;
    class Program;
    class ProgramVars;

    /** Low-level program object
        This class abstracts the API's program creation and management
    */
    class ProgramKernels : public std::enable_shared_from_this<ProgramKernels>
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramKernels>;
        using SharedConstPtr = std::shared_ptr<const ProgramKernels>;

        /** Create a new program object.
            \param[in]  ppShaders Zero or mroe shaders to go into the program
            \param[in]  shaderCount The number of shaders in ppShaders
            \param[in]  pRootSignature The root signature of the compiled kernels.
            \param[out] log In case of error, this will contain the error log string
            \param[in]  name Optional. A meaningful name to use with log messages
            \return New object in case of success, otherwise nullptr
        */
        static SharedPtr create(
            const ProgramReflection::SharedPtr& pReflector,
            Shader::SharedPtr const*            ppShaders,
            size_t                              shaderCount,
            const RootSignature::SharedPtr&     pRootSignature,
            std::string&                        log, 
            const std::string&                  name = "");

        /** Create a new program object for graphics.
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
            const ProgramReflection::SharedPtr& pReflector,
            const Shader::SharedPtr& pVS,
            const Shader::SharedPtr& pPS,
            const Shader::SharedPtr& pGS,
            const Shader::SharedPtr& pHS,
            const Shader::SharedPtr& pDS,
            const RootSignature::SharedPtr& pRootSignature,
            std::string& log, 
            const std::string& name = "");

        /** Create a new program object for compute.
            \param[in] pCs Compute shader object
            \param[out] Log In case of error, this will contain the error log string
            \param[in] DebugName Optional. A meaningful name to use with log messages
            \return New object in case of success, otherwise nullptr
        */
        static SharedPtr create(
            const ProgramReflection::SharedPtr& pReflector,
            const Shader::SharedPtr& pCS,
            const RootSignature::SharedPtr& pRootSignature,
            std::string& log,
            const std::string& name = "");

        virtual ~ProgramKernels();

        /** Get an attached shader object, or nullptr if no shader is attached to the slot.
        */
        const Shader* getShader(ShaderType Type) const { return mpShaders[(uint32_t)Type].get(); }

        /** Get the program name
        */
        const std::string& getName() const {return mName;}

        /** Get the reflection object
        */

        ProgramReflection::SharedConstPtr getReflector() const { return mpReflector; }
        
        /** Get the root signature object associated with this ProgramKernel
        */
        RootSignature::SharedPtr const& getRootSignature() const { return mpRootSignature; }

        /** Get a list of indices translating a parameter-block's set index to the root-signature entry index
        */
        const std::vector<uint32_t>& getParameterBlockRootIndices(uint32_t blockIndex) const { return mParameterBlocks[blockIndex].rootIndex; }

    protected:
        ProgramKernels(
            const ProgramReflection::SharedPtr& pReflector, 
            Shader::SharedPtr const*            ppShaders,
            size_t                              shaderCount,
            const RootSignature::SharedPtr&     pRootSignature,
            const std::string&                  name = "");

        virtual bool init(std::string& log);
        void deleteApiHandle();
        ProgramHandle mApiHandle = ProgramHandle();
        const std::string mName;

        static const uint32_t kShaderCount = (uint32_t)ShaderType::Count;
        Shader::SharedConstPtr mpShaders[kShaderCount];

        void* mpPrivateData;
        ProgramReflection::SharedConstPtr mpReflector;
        RootSignature::SharedPtr mpRootSignature;

        struct BlockData
        {
            std::vector<uint32_t> rootIndex;        // Maps the block's set-index to the root-signature entry
        };
        std::vector<BlockData> mParameterBlocks; 
    };

    /** A `Program` specialized to particular `#define`s
    */
    class ProgramVersion : public std::enable_shared_from_this<ProgramVersion>
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramVersion>;
        using SharedConstPtr = std::shared_ptr<const ProgramVersion>;
        using DefineList = Shader::DefineList;

        static SharedPtr create(
            std::shared_ptr<Program>     const& pProgram,
            DefineList                   const& defines,
            ProgramReflectors            const& reflectors,
            std::string                  const& name,
            SlangCompileRequest*                pSlangRequest);

        ~ProgramVersion();

        /** Get the program that this version was created from
        */
        std::shared_ptr<Program> getProgram() const { return mpProgram; }

        /** Get the defines that were used to create this version
        */
        DefineList const& getDefines() const { return mDefines; }

        /** Get the program name
        */
        const std::string& getName() const {return mName;}

        /** Get the reflection object
        */
        ProgramReflection::SharedPtr getReflector() const { return mReflectors.pReflector; }

        ProgramReflection::SharedPtr getLocalReflector() const { return mReflectors.pLocalReflector; }
        ProgramReflection::SharedPtr getGlobalReflector() const { return mReflectors.pGlobalReflector; }

        /** Get executable kernels based on state in a `ProgramVars`
        */
        ProgramKernels::SharedConstPtr getKernels(ProgramVars const* pVars) const;

        ParameterBlockReflection::SharedConstPtr getParameterBlockReflectorForType(std::string const& name) const;

    protected:
        ProgramVersion(
            std::shared_ptr<Program>     const& pProgram,
            DefineList                   const& defines,
            ProgramReflectors            const& reflectors,
            std::string                  const& name,
            SlangCompileRequest*                pSlangRequest);

        ProgramKernels::SharedConstPtr createKernels(ProgramVars const* pVars) const;

        std::shared_ptr<Program>        mpProgram;
        DefineList                      mDefines;
        ProgramReflectors               mReflectors;
        std::string                     mName;

        // The Slang compile request that created this version.
        //
        // Used to look up types and layouts when parameter blocks get created.
        SlangCompileRequest* mpSlangRequest = nullptr;

        // A cache of parameter block reflection objects based on Slang
        // type that have been looked up on-demand.
        //
        typedef std::map<std::string, ParameterBlockReflection::SharedConstPtr> ParameterBlockTypes;
        mutable ParameterBlockTypes mParameterBlockTypes;

        // Cached version of compiled kernels for this program version
        struct KernelGraphEntry
        {
            ProgramKernels::SharedConstPtr  pKernels;
            std::vector<uint32_t>           parameterBlockTypes;
        };
        using KernelGraph = Graph<KernelGraphEntry, uint32_t>;
        KernelGraph::SharedPtr mpKernelGraph;
    };
}