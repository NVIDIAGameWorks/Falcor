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

namespace Falcor
{
    // Forward declares for gui draw func
    class Gui;
    class ParameterBlock;

    class dlldecl VariablesBufferUI
    {
    public:
        VariablesBufferUI(ParameterBlock& variablesBufferRef) : mVariablesBufferRef(variablesBufferRef) {}
        void renderUI(Gui::Widgets& widget);

    private:
        ParameterBlock& mVariablesBufferRef;
        static std::unordered_map<std::string, int32_t> mGuiArrayIndices;

        /** Recursive function for displaying shader reflection member
            \param[in] widget The GUI widget to use for rendering
            \param[in] memberName The member name
            \param[in] var The current shader var
            \return true if something changed and data needs to be sent to the GPU
        */
        void renderUIVarInternal(Gui::Widgets& widget, const std::string& memberName, const ShaderVar& var);

        /** Recursive function for traversing reflection data and display ui
            \param[in] widget The GUI widget to use for rendering
            \param[in] var The current shader var
            \return true if something changed and data needs to be sent to the GPU
        */
        void renderUIInternal(Gui::Widgets& widget, const ShaderVar& var);

        /** Render gui widget for reflected data
            \param[in] widget The GUI widget to use for rendering
            \param[in] memberName string containing the name of the data member to render
            \param[in] var The current shader var
            \param[in] memberSize size of the data in the member
            \param[in] memberTypeString
            \param[in] memberType
            \param[in] arraySize
            \return true if something changed and data needs to be sent to the GPU
        */
        void renderUIBasicVarInternal(Gui::Widgets& widget, const std::string& memberName, const ShaderVar& var, size_t memberSize, const std::string& memberTypeString, const ReflectionBasicType::Type& memberType, size_t arraySize);
    };
}
