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
#include "Core/Macros.h"
#include "Core/State/ComputeState.h"
#include "Core/Program/ComputeProgram.h"
#include "Core/Program/Program.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Program/ShaderVar.h"
#include <filesystem>
#include <string>

namespace Falcor
{
    class FALCOR_API ComputePass
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<ComputePass>;

        /** Create a new compute pass from file.
            \param[in] path Compute program file path.
            \param[in] csEntry Name of the entry point in the program. If not specified "main" will be used.
            \param[in] defines Optional list of macro definitions to set into the program.
            \param[in] createVars Create program vars automatically, otherwise use setVars().
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const std::filesystem::path& path, const std::string& csEntry = "main", const Program::DefineList& defines = Program::DefineList(), bool createVars = true);

        /** Create a new compute pass.
            \param[in] desc The program's description.
            \param[in] defines Optional list of macro definitions to set into the program.
            \param[in] createVars Create program vars automatically, otherwise use setVars().
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create(const Program::Desc& desc, const Program::DefineList& defines = Program::DefineList(), bool createVars = true);

        /** Execute the pass using the given compute-context
            \param[in] pContext The compute context
            \param[in] nThreadX The number of threads to dispatch in the X dimension (note that this is not the number of thread groups)
            \param[in] nThreadY The number of threads to dispatch in the Y dimension (note that this is not the number of thread groups)
            \param[in] nThreadZ The number of threads to dispatch in the Z dimension (note that this is not the number of thread groups)
        */
        virtual void execute(ComputeContext* pContext, uint32_t nThreadX, uint32_t nThreadY, uint32_t nThreadZ = 1);

        /** Execute the pass using the given compute-context
            \param[in] pContext The compute context
            \param[in] nThreads The number of threads to dispatch in the XYZ dimensions (note that this is not the number of thread groups)
        */
        virtual void execute(ComputeContext* pContext, const uint3& nThreads) { execute(pContext, nThreads.x, nThreads.y, nThreads.z); }

        /** Execute the pass using indirect dispatch given the compute-context and argument buffer
            \param[in] pContext The compute context
            \param[in] pArgBuffer Argument buffer
            \param[in] argBufferOffset Offset in argument buffer
        */
        virtual void executeIndirect(ComputeContext* context, const Buffer* pArgBuffer, uint64_t argBufferOffset = 0);

        /** Check if a vars object exists. If not, use setVars() to set or create a new vars object.
            \return True if a vars object exists.
        */
        bool hasVars() const { return mpVars != nullptr; }

        /** Get the vars.
        */
        const ComputeVars::SharedPtr& getVars() const { FALCOR_ASSERT(mpVars); return mpVars; };

        ShaderVar getRootVar() const { return mpVars->getRootVar(); }

        /** Add a define
        */
        void addDefine(const std::string& name, const std::string& value = "", bool updateVars = false);

        /** Remove a define
        */
        void removeDefine(const std::string& name, bool updateVars = false);

        /** Get the program
        */
        ComputeProgram::SharedPtr getProgram() const { return mpState->getProgram(); }

        /** Set a vars object. Allows the user to override the internal vars, for example when one wants to share a vars object between different passes.
            The function throws an exception on error.
            \param[in] pVars The new GraphicsVars object. If this is nullptr, then the pass will automatically create a new vars object.
        */
        void setVars(const ComputeVars::SharedPtr& pVars);

        /** Get the thread group size from the program
        */
        uint3 getThreadGroupSize() const { return mpState->getProgram()->getReflector()->getThreadGroupSize(); }

    protected:
        ComputePass(const Program::Desc& desc, const Program::DefineList& defines, bool createVars);
        ComputeVars::SharedPtr mpVars;
        ComputeState::SharedPtr mpState;
    };
}
