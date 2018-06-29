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
#include "Graphics/Program/ProgramReflection.h"
#include "API/VariablesBuffer.h"

namespace Falcor
{
    // Forward declares for gui draw func
    class Gui;

    class VariablesBufferUI
    {
    public:
        VariablesBufferUI(VariablesBuffer& variablesBufferRef) : mVariabelsBufferRef(variablesBufferRef) {}

        void renderUI(Gui* pGui, const char* uiGroup);

    private:

        VariablesBuffer& mVariabelsBufferRef;
        static std::unordered_map<std::string, int32_t> mGuiArrayIndices;

        /** Call the corresponding gui function using the reflected data
        \param[in] pGui Pointer to the gui structure for rendering
        \param[in] type Reflection type to look up corresponding widget
        \param[in] offset offset into the data array of the constant buffer
        \param[in] name String name for widget to display
        \return true if data was changed from the widget. internal used to call upload to gpu
        */
        bool renderGuiWidgetFromType(Gui* pGui, ReflectionBasicType::Type type, size_t offset, const std::string& name);

        /** Recursive function for traversing reflection data and display ui
        \param[in] pGui Pointer to the gui structure for rendering
        \param[in] pStruct Pointer to structure to iterate and display for the gui
        \param[in] currentStructName Current struct name to append for full reflection name
        \param[in] startOffset Starting offset in memory for nested structures
        */
        void renderUIInternal(Gui* pGui, const ReflectionStructType* pStruct, const std::string& currentStructName, size_t startOffset, bool& dirtyFlag);

        /** Render gui widget for reflected data
        \param[in] pGui Pointer to the gui structure for rendering
        \param[in] memberName string containing the name of the data member to render
        \param[in] memberOffset offset into the data array
        \param[in] memberSize size of the data in the member
        \param[in] memberType reflection type enum for the basic type
        */
        void renderUIMemberInternal(Gui* pGui, const std::string& memberName, size_t memberOffset, size_t memberSize, const std::string& memberTypeString, const ReflectionBasicType::Type& memberType);
    };
}
