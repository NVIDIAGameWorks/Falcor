/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "..\RenderPass.h"
#include "Core\Program\ProgramVarsHelpers.h"

namespace Falcor
{
    class dlldecl ComputePass : public std::enable_shared_from_this<ComputePass>
    {
    public:
        using SharedPtr = VarsSharedPtr<ComputePass>;

        /** Create a new pass from a CS file
            \param[in] csFile Filename
            \param[in] csEntry Optional. Compute shader entry point name
            \param[in] defines Program defines
            \param[in] createVars Create program vars automatically, otherwise use setVars().
        */
        static SharedPtr create(const std::string& csFile, const std::string& csEntry = "main", const Program::DefineList& defines = Program::DefineList(), bool createVars = true);

        /** Create a new object
            \param[in] desc The program's description
            \param[in] defines Optional. A list of macro definitions to be patched into the shaders.
            \param[in] createVars Create program vars automatically, otherwise use setVars().
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
        virtual void execute(ComputeContext* pContext, const glm::uvec3& nThreads) { execute(pContext, nThreads.x, nThreads.y, nThreads.z); }

        /** Get the vars
        */
        const ComputeVars::SharedPtr& getVars() const { assert(mpVars); return mpVars; };

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
            \param[in] pVars The new GraphicsVars object. If this is nullptr, then the pass will automatically create a new GraphicsVars object
        */  
        void setVars(const ComputeVars::SharedPtr& pVars);

        /** Get the thread group size from the program
        */
        uvec3 getThreadGroupSize() const { return mpState->getProgram()->getReflector()->getThreadGroupSize(); }
    protected:
        ComputePass(const Program::Desc& desc, const Program::DefineList& defines, bool createVars);
        ComputeVars::SharedPtr mpVars;
        ComputeState::SharedPtr mpState;
    };
}
