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
#include "Framework.h"
#include "ConstantBuffer.h"
#include "Graphics/Program/ProgramVersion.h"
#include "Buffer.h"
#include "glm/glm.hpp"
#include "Texture.h"
#include "Graphics/Program/ProgramReflection.h"
#include "API/Device.h"

#include "Renderer.h"
#include "Utils/Gui.h"

namespace Falcor
{
    ConstantBuffer::~ConstantBuffer() = default;

    ConstantBuffer::ConstantBuffer(const std::string& name, const ReflectionResourceType::SharedConstPtr& pReflectionType, size_t size) :
        VariablesBuffer(name, pReflectionType, size, 1, Buffer::BindFlags::Constant, Buffer::CpuAccess::Write)
    {
    }

    ConstantBuffer::SharedPtr ConstantBuffer::create(const std::string& name, const ReflectionResourceType::SharedConstPtr& pReflectionType, size_t overrideSize)
    {
        size_t size = (overrideSize == 0) ? pReflectionType->getSize() : overrideSize;
        SharedPtr pBuffer = SharedPtr(new ConstantBuffer(name, pReflectionType, size));
        return pBuffer;
    }

    ConstantBuffer::SharedPtr ConstantBuffer::create(Program::SharedPtr& pProgram, const std::string& name, size_t overrideSize)
    {
        const auto& pProgReflector = pProgram->getReflector();
        const auto& pParamBlockReflection = pProgReflector->getDefaultParameterBlock();
        ReflectionVar::SharedConstPtr pBufferReflector = pParamBlockReflection ? pParamBlockReflection->getResource(name) : nullptr;

        if (pBufferReflector)
        {
            ReflectionResourceType::SharedConstPtr pResType = pBufferReflector->getType()->asResourceType()->inherit_shared_from_this::shared_from_this();
            if(pResType && pResType->getType() == ReflectionResourceType::Type::ConstantBuffer)
            {
                return create(name, pResType, overrideSize);
            }
        }
        logError("Can't find a constant buffer named \"" + name + "\" in the program");
        return nullptr;
    }

    bool ConstantBuffer::uploadToGPU(size_t offset, size_t size)
    {
        if (mDirty) mpCbv = nullptr;
        return VariablesBuffer::uploadToGPU(offset, size);
    }

    ConstantBufferView::SharedPtr ConstantBuffer::getCbv() const
    {
        if (mpCbv == nullptr)
        {
            mpCbv = ConstantBufferView::create(Resource::shared_from_this());
        }
        return mpCbv;
    }

    bool ConstantBuffer::getGuiWidgetFromType(Gui* pGui, ReflectionBasicType::Type type, size_t offset, const std::string& name)
    {
        unsigned displayIndex = 0;
        bool returnValue = false;

#define concatStrings_(a, b) a##b
#define concatStrings(a, b) concatStrings_(a, b)
#define to_gui_widget(widgetName, baseType) \
        returnValue |= pGui-> concatStrings(add, widgetName)(name.c_str(), *reinterpret_cast<baseType*>(mData.data() + offset)); \
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
            break;
        }
#undef to_gui_widget
#undef concatStrings
#undef concatStrings_

        return returnValue;
    }

    void ConstantBuffer::renderUIMemberInternal(Gui* pGui, const std::string& memberName, size_t memberOffset, size_t memberSize, const std::string& memberTypeString, const ReflectionBasicType::Type& memberType)
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
        mDirty |= getGuiWidgetFromType(pGui, memberType, memberOffset, memberName);

        pGui->addSeparator();
    }

    void ConstantBuffer::renderUIInternal(Gui* pGui, const ReflectionStructType* pStruct, const std::string& currentStructName, size_t startOffset)
    {
        static std::unordered_map<std::string, int32> sGuiArrayIndices;

        for (auto memberIt = pStruct->begin(); memberIt != pStruct->end(); ++memberIt)
        {
            size_t numMembers = 1;
            size_t memberSize = 0;
            ReflectionBasicType::Type memberType = ReflectionBasicType::Type::Unknown;
            std::string memberName = (*memberIt)->getName();
            const ReflectionBasicType* pBasicType = (*memberIt)->getType()->asBasicType();
            const ReflectionArrayType* pArrayType = nullptr;
            bool baseTypeIsStruct = false;
            bool arrayGroupStatus = false;
            size_t currentOffset = startOffset + (*memberIt)->getOffset();

            // First test is not basic type
            if (!pBasicType)
            {
                // recurse through struct if possible
                const ReflectionStructType* pStructType = (*memberIt)->getType()->asStructType();
                if (pStructType)
                {
                    // Iterate through the internal struct
                    if (pGui->beginGroup(memberName))
                    {
                        memberName.push_back('.');
                        renderUIInternal(pGui, pStructType, memberName, currentOffset);
                        memberName.pop_back();

                        pGui->endGroup();
                    }
                    pGui->addSeparator();

                    // skip to next member
                    continue;
                }

                // if array type gather info for iterating through elements
                pArrayType = (*memberIt)->getType()->asArrayType();

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
            unsigned memberIndex = 0;
            std::string displayName = memberName;
            
            if (numMembers > 1)
            {
                // display information for specific index of array
                int32& refGuiArrayIndex = sGuiArrayIndices[currentStructName + displayName];
                pGui->addIntVar((std::string("Index (Size : ") + std::to_string(numMembers) + ") ").c_str(), refGuiArrayIndex, 0, static_cast<int>(numMembers) - 1);
                memberIndex = refGuiArrayIndex;
                currentOffset += (memberSize * memberIndex);
                displayName.append("[").append(std::to_string(memberIndex)).append("]");
            }

            if (baseTypeIsStruct)
            {
                // For arrays of structs, display dropdown for struct before recursing through struct members
                if (pGui->beginGroup(displayName))
                {
                    displayName.push_back('.');
                    renderUIInternal(pGui, pArrayType->getType()->asStructType(), displayName, currentOffset);
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

    void ConstantBuffer::renderUI(Gui* pGui)
    {
        const ReflectionStructType* pStruct = mpReflector->asResourceType()->getStructType()->asStructType();

        if (pGui->beginGroup(std::string("ConstantBuffer: ").append(mName)))
        {
            pGui->addSeparator();

            // begin recursion on first struct
            renderUIInternal(pGui, pStruct, "", 0);

            // dirty flag for uploading will be set by GUI
            uploadToGPU();

            pGui->endGroup();
        }
    }
}