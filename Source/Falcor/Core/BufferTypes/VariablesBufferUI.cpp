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
#include "stdafx.h"
#include "VariablesBufferUI.h"
#include "Utils/UI/Gui.h"

namespace Falcor
{
    std::unordered_map<std::string, int32_t> VariablesBufferUI::mGuiArrayIndices;

    bool renderGuiWidgetFromType(Gui::Widgets& widget, ReflectionBasicType::Type type, uint8_t* data, const std::string& name)
    {
        bool returnValue = false;

#define to_gui_widget(widgetName, baseType) \
            returnValue = widget.widgetName(name.c_str(), *reinterpret_cast<baseType*>(data)); \

#define to_gui_widget_matrix(widgetName, baseType) \
            returnValue = widget.widgetName<baseType>(name.c_str(), *reinterpret_cast<baseType*>(data)); \

#define to_gui_widget_bvec(widgetName, baseType) \
            { \
                uint32_t* pUintData = reinterpret_cast<uint32_t*>(data); \
                baseType tempBVec; \
                for (int32_t i = 0; i < tempBVec.length(); ++i) { tempBVec[i] = pUintData[i]; } \
                returnValue = widget.widgetName(name.c_str(), tempBVec); \
                for (int32_t i = 0; i < tempBVec.length(); ++i) { pUintData[i] = tempBVec[i]; data += sizeof(uint32_t); } \
            }

        switch (type)
        {
        case ReflectionBasicType::Type::Bool4:
            to_gui_widget_bvec(checkbox, bool4);
            break;
        case ReflectionBasicType::Type::Bool3:
            to_gui_widget_bvec(checkbox, bool3);
            break;
        case ReflectionBasicType::Type::Bool2:
            to_gui_widget_bvec(checkbox, bool2);
            break;
        case ReflectionBasicType::Type::Bool:
            to_gui_widget(checkbox, bool);
            break;
        case ReflectionBasicType::Type::Uint4:
        case ReflectionBasicType::Type::Uint64_4:
        case ReflectionBasicType::Type::Int4:
        case ReflectionBasicType::Type::Int64_4:
            to_gui_widget(var, int4);
            break;
        case ReflectionBasicType::Type::Uint3:
        case ReflectionBasicType::Type::Uint64_3:
        case ReflectionBasicType::Type::Int3:
        case ReflectionBasicType::Type::Int64_3:
            to_gui_widget(var, int3);
            break;
        case ReflectionBasicType::Type::Uint2:
        case ReflectionBasicType::Type::Uint64_2:
        case ReflectionBasicType::Type::Int2:
        case ReflectionBasicType::Type::Int64_2:
            to_gui_widget(var, int2);
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
            to_gui_widget(var, float2);
            break;
        case ReflectionBasicType::Type::Float3:
            to_gui_widget(var, float3);
            break;
        case ReflectionBasicType::Type::Float4:
            to_gui_widget(var, float4);
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

    void VariablesBufferUI::renderUIBasicVarInternal(
        Gui::Widgets&                       widget,
        const std::string&                  memberName,
        const ShaderVar&                    var,
        size_t                              memberSize,
        const std::string&                  memberTypeString,
        const ReflectionBasicType::Type&    memberType,
        size_t                              arraySize)
    {
        // Display data from the stage memory
        auto data = (uint8_t*)(var.getRawData());
        bool dirty = renderGuiWidgetFromType(widget, memberType, data, memberName);

        // Display name and then reflection data as tooltip
        std::string toolTipString = "Offset: " + std::to_string(var.getByteOffset());
        toolTipString.append("\nSize: " + std::to_string(memberSize));
        if (arraySize > 1)
        {
            toolTipString.append("\nArray Size: " + std::to_string(arraySize));
        }
        toolTipString.append("\nType: " + memberTypeString);

        widget.tooltip(toolTipString.c_str(), true);
    }

    void VariablesBufferUI::renderUIVarInternal(Gui::Widgets& widget, const std::string& memberName, const ShaderVar& var)
    {
        size_t numMembers = 1;
        size_t memberSize = 0;
        ReflectionBasicType::Type memberType = ReflectionBasicType::Type::Unknown;

        auto pType = var.getType();
        const ReflectionBasicType* pBasicType = pType->asBasicType();
        const ReflectionArrayType* pArrayType = nullptr;
        bool baseTypeIsStruct = false;

        Gui::Group arrayGroup;

        // First test is not basic type
        if (!pBasicType)
        {
            // recurse through struct if possible
            const ReflectionStructType* pStructType = pType->asStructType();
            if (pStructType)
            {
                auto group = Gui::Group(widget, memberName);
                if (group.open())
                {
                    // Iterate through the internal struct
                    group.separator();

                    renderUIInternal(group, var);

                    group.separator();
                    group.release();
                }

                // skip to next member
                return;
            }

            // if array type gather info for iterating through elements
            pArrayType = pType->asArrayType();

            if (pArrayType)
            {
                const ReflectionBasicType* elementBasicType = pArrayType->getElementType()->asBasicType();
                numMembers = pArrayType->getElementCount();
                memberSize = pArrayType->getElementByteStride();

                // only iterate through array if it is displaying
                arrayGroup = Gui::Group(widget, memberName + "[" + std::to_string(numMembers) + "]");
                if(!arrayGroup.open()) return;

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
            else if (!pStructType) return;
        }
        else
        {
            // information if only basic type
            memberType = pBasicType->getType();
        }


        // Display member of the array
        std::string displayName = memberName;
        auto displayCursor = var;

        // TODO: Using `mGuiArrayIndices` here and computing a `displayName` here
        // is not using the ImGui library appropriately. Instead, it should use
        // the built-int facilities in ImGui for handling transient storage:
        //
        //      auto pStorage = ImGui::GetStateStorage();
        //      auto& memberIndex = *pStorage->GetIntRef(GetID());
        //      ...
        //
        int32_t& memberIndex = mGuiArrayIndices[displayName];
        bool dirty = false;
        if (numMembers > 1)
        {
            // display information for specific index of array
            std::string indexLabelString = (displayName + std::to_string(numMembers));
            dirty = arrayGroup.var(("Array Index" + std::string("##") + indexLabelString).c_str(), memberIndex, 0, static_cast<int32_t>(numMembers - 1));

            displayCursor = displayCursor[memberIndex];
            displayName.append("[" + std::to_string(memberIndex) + ":" + std::to_string(numMembers) + "]");
        }


        if (baseTypeIsStruct)
        {
            Gui::Group group(widget, displayName);
            if (group.open())
            {
                renderUIInternal(group, displayCursor);
                group.release();
            }
        }
        else
        {
            // for basic types
            renderUIBasicVarInternal(widget, displayName, displayCursor, memberSize, to_string(memberType), memberType, numMembers);
        }
    }

    void VariablesBufferUI::renderUIInternal(Gui::Widgets& widget, const ShaderVar& var)
    {
        auto pType = var.getType();
        if (auto pStruct = pType->asStructType())
        {
            auto memberCount = pStruct->getMemberCount();
            for(uint32_t m = 0; m < memberCount; ++m)
            {
                auto pMember = pStruct->getMember(m);
                auto memberName = pMember->getName();

                // TODO: We should wrap the following with
                // `ImGui::PushID` and `ImGui::PopID` to ensure
                // that the ID stack can be used to ensure
                // unique IDs for distinct members.

                return renderUIVarInternal(widget, memberName, var[memberName]);
            }
        }
        else
        {
            if (pType->asResourceType()->getStructuredBufferType() != ReflectionResourceType::StructuredType::Invalid)
            {
#if 0
                // TODO: ideally structured buffers (and arrays in general) should
                // display as a list view in the GUI, rather than an element
                // at a time.
                //
                size_t structSize = pType->asResourceType()->getStructType()->getByteSize();

                // TODO: need to allocate space for element index in ImGui itself...

                uint32_t memberIndex = 0;
                widget.var("Element Index", memberIndex, 0);
                renderUIInternal(widget, var[memberIndex]);
#endif
            }
            else
            {
                return renderUIVarInternal(widget, "", var);
            }
        }
    }

    void VariablesBufferUI::renderUI(Gui::Widgets& widget)
    {
        // begin recursion on first struct
        renderUIInternal(widget,mVariablesBufferRef.getRootVar());
    }
}
