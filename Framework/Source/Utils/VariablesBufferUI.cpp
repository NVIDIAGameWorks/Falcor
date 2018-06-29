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

#include "VariablesBufferUI.h"
#include "Utils/Gui.h"

namespace Falcor
{
    std::unordered_map<std::string, int32_t> VariablesBufferUI::mGuiArrayIndices;

    bool VariablesBufferUI::renderGuiWidgetFromType(Gui* pGui, ReflectionBasicType::Type type, size_t offset, const std::string& name)//, std::vector<int32_t>& data)
    {
        bool returnValue = false;

#define to_gui_widget(widgetName, baseType) \
            returnValue = pGui-> concat_strings(add, widgetName)(name.c_str(), *reinterpret_cast<baseType*>(mVariabelsBufferRef.mData.data() + offset)); \
            offset += sizeof(baseType);

        switch (type)
        {
        case ReflectionBasicType::Type::Bool4:
            to_gui_widget(Bool4Var, glm::bvec4);
            break;
        case ReflectionBasicType::Type::Bool3:
            to_gui_widget(Bool3Var, glm::bvec3);
            break;
        case ReflectionBasicType::Type::Bool2:
            to_gui_widget(Bool2Var, glm::bvec2);
            break;
        case ReflectionBasicType::Type::Bool:
            to_gui_widget(CheckBox, bool);
            break;
        case ReflectionBasicType::Type::Uint4:
        case ReflectionBasicType::Type::Uint64_4:
        case ReflectionBasicType::Type::Int4:
        case ReflectionBasicType::Type::Int64_4:
            to_gui_widget(Int4Var, glm::ivec4);
            break;
        case ReflectionBasicType::Type::Uint3:
        case ReflectionBasicType::Type::Uint64_3:
        case ReflectionBasicType::Type::Int3:
        case ReflectionBasicType::Type::Int64_3:
            to_gui_widget(Int3Var, glm::ivec3);
            break;
        case ReflectionBasicType::Type::Uint2:
        case ReflectionBasicType::Type::Uint64_2:
        case ReflectionBasicType::Type::Int2:
        case ReflectionBasicType::Type::Int64_2:
            to_gui_widget(Int2Var, glm::ivec2);
            break;
        case ReflectionBasicType::Type::Uint:
        case ReflectionBasicType::Type::Uint64:
        case ReflectionBasicType::Type::Int:
        case ReflectionBasicType::Type::Int64:
            to_gui_widget(IntVar, int);
            break;
        case ReflectionBasicType::Type::Float:
            to_gui_widget(FloatVar, float);
            break;
        case ReflectionBasicType::Type::Float2:
            to_gui_widget(Float2Var, glm::vec2);
            break;
        case ReflectionBasicType::Type::Float3:
            to_gui_widget(Float3Var, glm::vec3);
            break;
        case ReflectionBasicType::Type::Float4:
            to_gui_widget(Float4Var, glm::vec4);
            break;
        case ReflectionBasicType::Type::Float2x2:
            to_gui_widget(Matrix2x2Var, glm::mat2x2);
            break;
        case ReflectionBasicType::Type::Float2x3:
            to_gui_widget(Matrix2x3Var, glm::mat2x3);
            break;
        case ReflectionBasicType::Type::Float2x4:
            to_gui_widget(Matrix2x4Var, glm::mat2x4);
            break;
        case ReflectionBasicType::Type::Float3x2:
            to_gui_widget(Matrix3x2Var, glm::mat3x2);
            break;
        case ReflectionBasicType::Type::Float3x3:
            to_gui_widget(Matrix3x3Var, glm::mat3x3);
            break;
        case ReflectionBasicType::Type::Float3x4:
            to_gui_widget(Matrix3x4Var, glm::mat3x4);
            break;
        case ReflectionBasicType::Type::Float4x2:
            to_gui_widget(Matrix4x2Var, glm::mat4x2);
            break;
        case ReflectionBasicType::Type::Float4x3:
            to_gui_widget(Matrix4x3Var, glm::mat4x3);
            break;
        case ReflectionBasicType::Type::Float4x4:
            to_gui_widget(Matrix4x4Var, glm::mat4x4);
            break;
        case ReflectionBasicType::Type::Unknown:
            break;
        default:
            should_not_get_here();
            break;
        }
#undef to_gui_widget

        return returnValue;
    }

    void VariablesBufferUI::renderUIMemberInternal(Gui* pGui, const std::string& memberName, size_t memberOffset, size_t memberSize, const std::string& memberTypeString, const ReflectionBasicType::Type& memberType)
    {
        // Display reflection data and gather offset
        pGui->addText("Name: ", false);
        pGui->addText(memberName.c_str(), true);
        pGui->addText("Offset: ", false);
        pGui->addText(std::to_string(memberOffset).c_str(), true);
        pGui->addText("	Size: ", true);
        pGui->addText(std::to_string(memberSize).c_str(), true);
        pGui->addText("	Type: ", true);
        pGui->addText(memberTypeString.c_str(), true);

        // Display data from the stage memory
        mVariabelsBufferRef.mDirty |= renderGuiWidgetFromType(pGui, memberType, memberOffset, memberName);
    }

    void VariablesBufferUI::renderUIInternal(Gui* pGui, const ReflectionStructType* pStruct, const std::string& currentStructName, size_t startOffset, bool& dirtyFlag)
    {
        for (auto memberIt : *pStruct)
        {
            size_t numMembers = 1;
            size_t memberSize = 0;
            ReflectionBasicType::Type memberType = ReflectionBasicType::Type::Unknown;
            std::string memberName = (memberIt)->getName();
            const ReflectionBasicType* pBasicType = (memberIt)->getType()->asBasicType();
            const ReflectionArrayType* pArrayType = nullptr;
            bool baseTypeIsStruct = false;
            bool arrayGroupStatus = false;
            size_t currentOffset = startOffset + (memberIt)->getOffset();

            // First test is not basic type
            if (!pBasicType)
            {
                // recurse through struct if possible
                const ReflectionStructType* pStructType = (memberIt)->getType()->asStructType();
                if (pStructType)
                {
                    // Iterate through the internal struct
                    if (pGui->beginGroup(memberName))
                    {
                        memberName.push_back('.');
                        renderUIInternal(pGui, pStructType, memberName, currentOffset, dirtyFlag);
                        memberName.pop_back();

                        pGui->endGroup();
                    }
                    pGui->addSeparator();

                    // skip to next member
                    continue;
                }

                // if array type gather info for iterating through elements
                pArrayType = (memberIt)->getType()->asArrayType();

                if (pArrayType)
                {
                    pGui->addSeparator();

                    // only iterate through array if it is displaying
                    arrayGroupStatus = pGui->beginGroup(memberName + "[]");
                    if (!arrayGroupStatus)
                    {
                        pGui->addSeparator();
                        continue;
                    }

                    const ReflectionBasicType* elementBasicType = pArrayType->getType()->asBasicType();
                    numMembers = pArrayType->getArraySize();
                    memberSize = pArrayType->getArrayStride();

                    if (elementBasicType)
                    {
                        memberType = elementBasicType->getType();
                    }
                    else
                    {
                        // for special case of array of structures
                        baseTypeIsStruct = true;
                    }
                }
                else if (!pStructType)
                {
                    // Other types could be presented here
                    return;
                }
            }
            else
            {
                // information if only basic type
                memberType = pBasicType->getType();
                memberSize = pBasicType->getSize();
            }


            // Display member of the array
            std::string displayName = memberName;

            if (numMembers > 1)
            {
                // display information for specific index of array
                int32_t& memberIndex = mGuiArrayIndices[displayName];
                pGui->addIntVar((std::string("Index (Size : ") + std::to_string(numMembers) + ") ").c_str(), memberIndex, 0, static_cast<int>(numMembers) - 1);
                currentOffset += (memberSize * memberIndex);
                displayName.append("[").append(std::to_string(memberIndex)).append("]");
            }

            if (baseTypeIsStruct)
            {
                // For arrays of structs, display dropdown for struct before recursing through struct members
                if (pGui->beginGroup(displayName))
                {
                    displayName.push_back('.');
                    renderUIInternal(pGui, pArrayType->getType()->asStructType(), displayName, currentOffset, dirtyFlag);
                    pGui->endGroup();
                }
            }
            else
            {
                // for basic types
                renderUIMemberInternal(pGui, displayName, currentOffset, memberSize, to_string(memberType), memberType);
            }

            currentOffset += memberSize;


            if (arrayGroupStatus)
            {
                pGui->endGroup();
            }
        }
    }

    void VariablesBufferUI::renderUI(Gui* pGui, const char* uiGroup)
    {
        const ReflectionStructType* pStruct = mVariabelsBufferRef.mpReflector->asResourceType()->getStructType()->asStructType();

        if (!uiGroup || pGui->beginGroup(uiGroup))
        {
            pGui->addSeparator();

            // begin recursion on first struct
            renderUIInternal(pGui, pStruct, "", 0, mVariabelsBufferRef.mDirty);

            // dirty flag for uploading will be set by GUI
            mVariabelsBufferRef.uploadToGPU();

            pGui->endGroup();
        }
    }
}
