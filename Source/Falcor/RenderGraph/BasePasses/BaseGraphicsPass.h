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
#include "Core/Program/GraphicsProgram.h"
#include "Core/Program/ProgramVars.h"
#include "Core/State/GraphicsState.h"

namespace Falcor
{
    class dlldecl BaseGraphicsPass
    {
    public:
        virtual ~BaseGraphicsPass() = default;

        /** Add a define
        */
        void addDefine(const std::string& name, const std::string& value = "", bool updateVars = false);

        /** Remove a define
        */
        void removeDefine(const std::string& name, bool updateVars = false);

        /** Get the program
        */
        GraphicsProgram::SharedPtr getProgram() const { return mpState->getProgram(); }

        /** Get the state
        */
        const GraphicsState::SharedPtr& getState() const { return mpState; }

        /** Get the vars
        */
        const GraphicsVars::SharedPtr& getVars() const { return mpVars; }

        ShaderVar getRootVar() const { return mpVars->getRootVar(); }

        /** Set a vars object. Allows the user to override the internal vars, for example when one wants to share a vars object between different passes.
            \param[in] pVars The new GraphicsVars object. If this is nullptr, then the pass will automatically create a new GraphicsVars object
        */
        void setVars(const GraphicsVars::SharedPtr& pVars);

    protected:
        /** Create a new object.
            \param[in] progDesc The program description.
            \param[in] programDefines List of macro definitions to set into the program. The macro definitions will be set on all shader stages.
            \return A new object, or an exception is thrown if creation failed.
        */
        BaseGraphicsPass(const Program::Desc& progDesc, const Program::DefineList& programDefines);

        GraphicsVars::SharedPtr mpVars;
        GraphicsState::SharedPtr mpState;
    };
}
