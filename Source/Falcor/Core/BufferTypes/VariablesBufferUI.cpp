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
#include "stdafx.h"
#include "VariablesBufferUI.h"
#include "Utils/UI/Gui.h"

namespace Falcor
{
    std::unordered_map<std::string, int32_t> VariablesBufferUI::mGuiArrayIndices;

    bool renderGuiWidgetFromType(Gui::Widgets& widget, ReflectionBasicType::Type type, size_t offset, const std::string& name, std::vector<uint8_t>& data)
    {
        bool returnValue = false;

#define to_gui_widget(widgetName, baseType) \
            returnValue = widget.widgetName(name.c_str(), *reinterpret_cast<baseType*>(data.data() + offset)); \
            offset += sizeof(baseType);
#define to_gui_widget_matrix(widgetName, baseType) \
            returnValue = widget.widgetName<baseType>(name.c_str(), *reinterpret_cast<baseType*>(data.data() + offset)); \
            offset += sizeof(baseType);
#define to_gui_widget_bvec(widgetName, baseType) \
            { \
                uint32_t* pUintData = reinterpret_cast<uint32_t*>(data.data() + offset); \
                baseType tempBVec; \
                for (int32_t i = 0; i < tempBVec.length(); ++i) { tempBVec[i] = pUintData[i]; } \
                returnValue = widget.widgetName(name.c_str(), tempBVec); \
                for (int32_t i = 0; i < tempBVec.length(); ++i) { pUintData[i] = tempBVec[i]; offset += sizeof(uint32_t); } \
            }

        switch (type)
        {
        case ReflectionBasicType::Type::Bool4:
            to_gui_widget_bvec(checkbox, glm::bvec4);
            break;
        case ReflectionBasicType::Type::Bool3:
            to_gui_widget_bvec(checkbox, glm::bvec3);
            break;
        case ReflectionBasicType::Type::Bool2:
            to_gui_widget_bvec(checkbox, glm::bvec2);
            break;
        case ReflectionBasicType::Type::Bool:
            to_gui_widget(checkbox, bool);
            break;
        case ReflectionBasicType::Type::Uint4:
        case ReflectionBasicType::Type::Uint64_4:
        case ReflectionBasicType::Type::Int4:
        case ReflectionBasicType::Type::Int64_4:
            to_gui_widget(var, glm::ivec4);
            break;
        case ReflectionBasicType::Type::Uint3:
        case ReflectionBasicType::Type::Uint64_3:
        case ReflectionBasicType::Type::Int3:
        case ReflectionBasicType::Type::Int64_3:
            to_gui_widget(var, glm::ivec3);
            break;
        case ReflectionBasicType::Type::Uint2:
        case ReflectionBasicType::Type::Uint64_2:
        case ReflectionBasicType::Type::Int2:
        case ReflectionBasicType::Type::Int64_2:
            to_gui_widget(var, glm::ivec2);
            break;
        case ReflectionBasicType::Type::Uint:
        case ReflectionBasicType::Type::Uint64:
        case ReflectionBasicType::Type::Int:
        case ReflectionBasicType::Type::Int64:
            to_gui_widget(var, int);
            break;
        case ReflectionBasicType::Type::Float:
            to_gui_widget(var, float);
            break;
        case ReflectionBasicType::Type::Float2:
            to_gui_widget(var, glm::vec2);
            break;
        case ReflectionBasicType::Type::Float3:
            to_gui_widget(var, glm::vec3);
            break;
        case ReflectionBasicType::Type::Float4:
            to_gui_widget(var, glm::vec4);
            break;
        case ReflectionBasicType::Type::Float2x2:
            to_gui_widget_matrix(matrix, glm::mat2x2);
            break;
        case ReflectionBasicType::Type::Float2x3:
            to_gui_widget_matrix(matrix, glm::mat2x3);
            break;
        case ReflectionBasicType::Type::Float2x4:
            to_gui_widget_matrix(matrix, glm::mat2x4);
            break;
        case ReflectionBasicType::Type::Float3x2:
            to_gui_widget_matrix(matrix, glm::mat3x2);
            break;
        case ReflectionBasicType::Type::Float3x3:
            to_gui_widget_matrix(matrix, glm::mat3x3);
            break;
        case ReflectionBasicType::Type::Float3x4:
            to_gui_widget_matrix(matrix, glm::mat3x4);
            break;
        case ReflectionBasicType::Type::Float4x2:
            to_gui_widget_matrix(matrix, glm::mat4x2);
            break;
        case ReflectionBasicType::Type::Float4x3:
            to_gui_widget_matrix(matrix, glm::mat4x3);
            break;
        case ReflectionBasicType::Type::Float4x4:
            to_gui_widget_matrix(matrix, glm::mat4x4);
            break;
        case ReflectionBasicType::Type::Unknown:
            break;
        default:
            should_not_get_here();
            break;
        }
#undef to_gui_widget_bvec
#undef to_gui_widget

        return returnValue;
    }

    void VariablesBufferUI::renderUIMemberInternal(Gui::Widgets& widget, const std::string& memberName, size_t memberOffset, size_t memberSize, const std::string& memberTypeString, const ReflectionBasicType::Type& memberType, size_t arraySize)
    {
        // Display data from the stage memory
        mVariablesBufferRef.mDirty |= renderGuiWidgetFromType(widget, memberType, memberOffset, memberName, mVariablesBufferRef.mData);

        // Display name and then reflection data as tooltip
        std::string toolTipString = "Offset: " + std::to_string(memberOffset);
        toolTipString.append("\nSize: " + std::to_string(memberSize));
        if (arraySize > 1)
        {
            toolTipString.append("\nArray Size: " + std::to_string(arraySize));
        }
        toolTipString.append("\nType: " + memberTypeString);

        widget.tooltip(toolTipString.c_str(), true);
    }

    void VariablesBufferUI::renderUIVarInternal(Gui::Widgets& widget, const ReflectionVar::SharedConstPtr& pMember, const std::string& currentStructName, size_t startOffset, bool& dirtyFlag)
    {
        size_t numMembers = 1;
        size_t memberSize = 0;
        ReflectionBasicType::Type memberType = ReflectionBasicType::Type::Unknown;
        std::string memberName = (pMember)->getName();
        const ReflectionBasicType* pBasicType = (pMember)->getType()->asBasicType();
        const ReflectionArrayType* pArrayType = nullptr;
        bool baseTypeIsStruct = false;
        size_t currentOffset = startOffset + (pMember)->getOffset();
        std::shared_ptr<Gui::Group> arrayGroup;

        // First test is not basic type
        if (!pBasicType)
        {
            // recurse through struct if possible
            const ReflectionStructType* pStructType = (pMember)->getType()->asStructType();
            if (pStructType)
            {
                auto group = Gui::Group(widget, memberName);
                if (group.open())
                {
                    // Iterate through the internal struct
                    group.separator();

                    memberName.push_back('.');
                    renderUIInternal(group, pStructType, memberName, currentOffset, dirtyFlag);
                    memberName.pop_back();

                    group.separator();
                    group.release();
                }

                // skip to next member
                return;
            }

            // if array type gather info for iterating through elements
            pArrayType = (pMember)->getType()->asArrayType();

            if (pArrayType)
            {
                const ReflectionBasicType* elementBasicType = pArrayType->getType()->asBasicType();
                numMembers = pArrayType->getArraySize();
                memberSize = pArrayType->getArrayStride();

                // only iterate through array if it is displaying
                arrayGroup = std::shared_ptr<Gui::Group>(new Gui::Group(widget.gui(), memberName + "[" + std::to_string(numMembers) + "]"));
                if (!arrayGroup->open())
                {
                    return;
                }

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
        int32_t& memberIndex = mGuiArrayIndices[displayName];

        if (numMembers > 1)
        {
            // display information for specific index of array
            std::string indexLabelString = (displayName + std::to_string(numMembers));
            arrayGroup->var(("Array Index" + std::string("##") + indexLabelString).c_str(), memberIndex, 0, static_cast<int32_t>(numMembers - 1));

            currentOffset += (memberSize * memberIndex);
            displayName.append("[" + std::to_string(memberIndex) + ":" + std::to_string(numMembers) + "]");
        }

        if (baseTypeIsStruct)
        {
            auto group = Gui::Group(widget, displayName);
            if (group.open())
            {
                // For arrays of structs, display dropdown for struct before recursing through struct members
                displayName.push_back('.');
                renderUIInternal(group, pArrayType->getType()->asStructType(), displayName, currentOffset, dirtyFlag);
                group.release();
            }
        }
        else
        {
            // for basic types
            renderUIMemberInternal(widget, displayName, currentOffset, memberSize, to_string(memberType), memberType, numMembers);
        }

        if (numMembers > 1)
        {
            arrayGroup->release();
        }
    }

    void VariablesBufferUI::renderUIInternal(Gui::Widgets& widget, const ReflectionType* pType, const std::string& currentStructName, size_t startOffset, bool& dirtyFlag)
    {
        const ReflectionStructType* pStruct = pType->asStructType();
        if (pStruct)
        {
            for (auto pMember : *pStruct)
            {
                // test if a struct member is a structured buffer
                const ReflectionResourceType* pResourceType = pMember->getType()->asResourceType();
                if (pResourceType && pResourceType->getStructuredBufferType() != ReflectionResourceType::StructuredType::Invalid)
                {
                    renderUIInternal(widget, pMember->getType().get(), currentStructName + pMember->getName(), startOffset, dirtyFlag);
                }
                else
                {
                    renderUIVarInternal(widget, pMember, currentStructName, startOffset, dirtyFlag);
                }
            }
        }
        else
        {
            // for structured buffers
            if (pType->asResourceType()->getStructuredBufferType() != ReflectionResourceType::StructuredType::Invalid)
            {
                std::string displayName = currentStructName; // find a way to get the name of the structured buffer
                int32_t& memberIndex = mGuiArrayIndices[displayName];
                int32_t oldMemberIndex = memberIndex;
                size_t structSize = pType->getSize();

                if (widget.var(("Structured Buffer Index" + std::string("##") + displayName).c_str(), memberIndex, 0))
                {
                    // since we can't get the actual number of structs by default, we test the validity of the offset
                    size_t offset = (structSize * memberIndex);
                    if (mVariablesBufferRef.mData.size() <= offset)
                    {
                        memberIndex = oldMemberIndex;
                    }
                }

                startOffset += (structSize * memberIndex);
                displayName.append("[" + std::to_string(memberIndex) + "]");

                auto group = Gui::Group(widget, displayName.c_str());
                if (group.open())
                {
                    renderUIInternal(widget, pType->asResourceType()->getStructType().get(), displayName, startOffset, dirtyFlag);
                    group.release();
                }
            }
            else
            {
                renderUIInternal(widget, pType->asResourceType()->getStructType().get(), currentStructName, startOffset, dirtyFlag);
            }
        }
    }

    void VariablesBufferUI::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (!uiGroup) uiGroup = "Variables Buffer";

        auto group = Gui::Group(pGui, uiGroup);
        if (group.open())
        {   
            // begin recursion on first struct
            renderUIInternal(group, mVariablesBufferRef.mpReflector.get(), "", 0, mVariablesBufferRef.mDirty);

            // dirty flag for uploading will be set by GUI
            mVariablesBufferRef.uploadToGPU();

            group.release();
        }
    }
}
