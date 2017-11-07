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
#include "ProgramReflection.h"
#include "Utils/StringUtils.h"
#include <unordered_set>
using namespace slang;

namespace Falcor
{
    static std::unordered_set<std::string> gParameterBlocksRegistry;

    bool isParameterBlock(VariableLayoutReflection* pSlangVar)
    {
        // A candidate for a parameter block must be a top-level constant buffer containing a single struct
        TypeLayoutReflection* pSlangType = pSlangVar->getTypeLayout();
        if (pSlangType->getTotalArrayElementCount() == 0 && pSlangType->unwrapArray()->getKind() == TypeReflection::Kind::ConstantBuffer)
        {
            if (pSlangType->unwrapArray()->getElementCount() == 1)
            {
                TypeLayoutReflection* pFieldLayout = pSlangType->unwrapArray()->getFieldByIndex(0)->getTypeLayout();
                if (pFieldLayout->getTotalArrayElementCount() == 0 && pFieldLayout->unwrapArray()->getKind() == TypeReflection::Kind::Struct)
                {
                    return true;
                }
            }
        }
        return false;
    }

    static ReflectionType::Type getResourceType(TypeReflection* pSlangType)
    {
        switch (pSlangType->unwrapArray()->getKind())
        {
        case TypeReflection::Kind::ConstantBuffer:
            return ReflectionType::Type::ConstantBuffer;
        case TypeReflection::Kind::SamplerState:
            return ReflectionType::Type::Sampler;
        case TypeReflection::Kind::ShaderStorageBuffer:
            return ReflectionType::Type::StructuredBuffer;
        case TypeReflection::Kind::Resource:
            switch (pSlangType->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK)
            {
            case SLANG_STRUCTURED_BUFFER:
                return ReflectionType::Type::StructuredBuffer;

            case SLANG_BYTE_ADDRESS_BUFFER:
                return ReflectionType::Type::RawBuffer;
            case SLANG_TEXTURE_BUFFER:
                return ReflectionType::Type::TypedBuffer;
            default:
                return ReflectionType::Type::Texture;
            }
            break;

        default:
            break;
        }
        return ReflectionType::Type::Unknown;
    }

    static ReflectionType::ShaderAccess getShaderAccess(TypeReflection* pSlangType)
    {
        // Compute access for an array using the underlying type...
        pSlangType = pSlangType->unwrapArray();

        switch (pSlangType->getKind())
        {
        case TypeReflection::Kind::SamplerState:
        case TypeReflection::Kind::ConstantBuffer:
            return ReflectionType::ShaderAccess::Read;
            break;

        case TypeReflection::Kind::Resource:
        case TypeReflection::Kind::ShaderStorageBuffer:
            switch (pSlangType->getResourceAccess())
            {
            case SLANG_RESOURCE_ACCESS_NONE:
                return ReflectionType::ShaderAccess::Undefined;

            case SLANG_RESOURCE_ACCESS_READ:
                return ReflectionType::ShaderAccess::Read;

            default:
                return ReflectionType::ShaderAccess::ReadWrite;
            }
            break;

        default:
            should_not_get_here();
            return ReflectionType::ShaderAccess::Undefined;
        }
    }

    static ReflectionType::ReturnType getReturnType(TypeReflection* pType)
    {
        // Could be a resource that doesn't have a specific element type (e.g., a raw buffer)
        if (!pType)
            return ReflectionType::ReturnType::Unknown;

        switch (pType->getScalarType())
        {
        case TypeReflection::ScalarType::Float32:
            return ReflectionType::ReturnType::Float;
        case TypeReflection::ScalarType::Int32:
            return ReflectionType::ReturnType::Int;
        case TypeReflection::ScalarType::UInt32:
            return ReflectionType::ReturnType::Uint;
        case TypeReflection::ScalarType::Float64:
            return ReflectionType::ReturnType::Double;

            // Could be a resource that uses an aggregate element type (e.g., a structured buffer)
        case TypeReflection::ScalarType::None:
            return ReflectionType::ReturnType::Unknown;

        default:
            should_not_get_here();
            return ReflectionType::ReturnType::Unknown;
        }
    }

    static ReflectionType::Dimensions getResourceDimensions(SlangResourceShape shape)
    {
        switch (shape)
        {
        case SLANG_TEXTURE_1D:
            return ReflectionType::Dimensions::Texture1D;
        case SLANG_TEXTURE_1D_ARRAY:
            return ReflectionType::Dimensions::Texture1DArray;
        case SLANG_TEXTURE_2D:
            return ReflectionType::Dimensions::Texture2D;
        case SLANG_TEXTURE_2D_ARRAY:
            return ReflectionType::Dimensions::Texture2DArray;
        case SLANG_TEXTURE_2D_MULTISAMPLE:
            return ReflectionType::Dimensions::Texture2DMS;
        case SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return ReflectionType::Dimensions::Texture2DMSArray;
        case SLANG_TEXTURE_3D:
            return ReflectionType::Dimensions::Texture3D;
        case SLANG_TEXTURE_CUBE:
            return ReflectionType::Dimensions::TextureCube;
        case SLANG_TEXTURE_CUBE_ARRAY:
            return ReflectionType::Dimensions::TextureCubeArray;

        case SLANG_TEXTURE_BUFFER:
        case SLANG_STRUCTURED_BUFFER:
        case SLANG_BYTE_ADDRESS_BUFFER:
            return ReflectionType::Dimensions::Buffer;

        default:
            return ReflectionType::Dimensions::Unknown;
        }
    }

    ReflectionVar::SharedPtr reflectVariable(VariableLayoutReflection* pSlangLayout);

    ReflectionType::SharedPtr reflectType(TypeLayoutReflection* pSlangType)
    {
        ReflectionType::Dimensions dims = getResourceDimensions(pSlangType->getResourceShape());
        ReflectionType::ShaderAccess shaderAccess = getShaderAccess(pSlangType->getType());
        ReflectionType::ReturnType retType = getReturnType(pSlangType->getType());
        ReflectionType::Type type = getResourceType(pSlangType->getType());

        ReflectionType::SharedPtr pType = ReflectionType::create(type, 0, (uint32_t)pSlangType->getElementCount(), 0, shaderAccess, retType, dims);
        for (uint32_t i = 0; i < pSlangType->getFieldCount(); i++)
        {
            ReflectionVar::SharedPtr pVar = reflectVariable(pSlangType->getFieldByIndex(i));
            pType->addMember(pVar);
        }
        return pType;
    }

    ReflectionVar::SharedPtr reflectVariable(VariableLayoutReflection* pSlangLayout)
    {
        std::string name(pSlangLayout->getName());
        ReflectionType::SharedPtr pType = reflectType(pSlangLayout->getTypeLayout());
        uint32_t index = pSlangLayout->getBindingIndex();
        uint32_t space = pSlangLayout->getBindingSpace();
        uint32_t offset = (uint32_t)pSlangLayout->getOffset();
        return ReflectionVar::create(name, pType, offset, space);
    }

    ProgramReflection::SharedPtr ProgramReflection::create(slang::ShaderReflection* pSlangReflector, std::string& log)
    {
        return SharedPtr(new ProgramReflection(pSlangReflector, log));
    }

    ProgramReflection::ProgramReflection(slang::ShaderReflection* pSlangReflector, std::string& log)
    {
        ReflectionType::SharedPtr pGlobalBlockType = ReflectionType::create(ReflectionType::Type::ParameterBlock, 0, 0, 0);
        for (uint32_t i = 0; i < pSlangReflector->getParameterCount(); i++)
        {
            VariableLayoutReflection* pSlangLayout = pSlangReflector->getParameterByIndex(i);
            ReflectionVar::SharedPtr pVar = reflectVariable(pSlangLayout);

            if (isParameterBlock(pSlangLayout))
            {
                ReflectionType::SharedPtr pBlock = ReflectionType::create(ReflectionType::Type::ParameterBlock, 0, 0, 0);
                pBlock->addMember(pVar);
                std::string name = std::string(pSlangLayout->getName());
                addParameterBlock(name, pBlock);
            }
            else
            {
                pGlobalBlockType->addMember(pVar);
            }
        }

        if (pGlobalBlockType->getMemberCount())
        {
            addParameterBlock("", pGlobalBlockType);
        }
    }

    void ProgramReflection::addParameterBlock(const std::string& name, const ReflectionType::SharedConstPtr& pType)
    {

    }

    ReflectionType::SharedPtr ReflectionType::create(Type type, uint32_t offset, uint32_t arraySize, uint32_t arrayStride, ShaderAccess shaderAccess, ReturnType retType, Dimensions dims)
    {
        return SharedPtr(new ReflectionType(type, offset, arraySize, arrayStride, shaderAccess, retType, dims));
    }

    ReflectionType::ReflectionType(Type type, uint32_t offset, uint32_t arraySize, uint32_t arrayStride, ShaderAccess shaderAccess, ReturnType retType, Dimensions dims)
    {

    }
    void ReflectionType::addMember(const std::shared_ptr<const ReflectionVar>& pVar)
    {
        mMembers.push_back(pVar);
    }

    ReflectionVar::SharedPtr ReflectionVar::create(const std::string& name, const ReflectionType::SharedConstPtr& pType, uint32_t offset, uint32_t regSpace)
    {
        return SharedPtr(new ReflectionVar(name, pType, offset, regSpace));
    }

    ReflectionVar::ReflectionVar(const std::string& name, const ReflectionType::SharedConstPtr& pType, uint32_t offset, uint32_t regSpace)
    {

    }
}