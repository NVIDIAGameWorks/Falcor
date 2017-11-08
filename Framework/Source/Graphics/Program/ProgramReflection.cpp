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
using namespace slang;

namespace Falcor
{
    std::unordered_set<std::string> ProgramReflection::sParameterBlockRegistry;

    bool isParameterBlockReflection(VariableLayoutReflection* pSlangVar, const std::unordered_set<std::string>& parameterBlockRegistry)
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
                    const std::string name = "";
                    return parameterBlockRegistry.find(name) != parameterBlockRegistry.end();
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
        case TypeReflection::Kind::Struct:
            return ReflectionType::Type::Struct;
        default:
            return ReflectionType::Type::Unknown;
        }
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

    ReflectionType::Type getVariableType(TypeReflection::ScalarType slangScalarType, uint32_t rows, uint32_t columns)
    {
        switch (slangScalarType)
        {
        case TypeReflection::ScalarType::None:
            // This isn't a scalar/matrix/vector, so it can't
            // be encoded in the `enum` that Falcor provides.
            return ReflectionType::Type::Unknown;

        case TypeReflection::ScalarType::Bool:
            assert(rows == 1);
            switch (columns)
            {
            case 1:
                return ReflectionType::Type::Bool;
            case 2:
                return ReflectionType::Type::Bool2;
            case 3:
                return ReflectionType::Type::Bool3;
            case 4:
                return ReflectionType::Type::Bool4;
            }
        case TypeReflection::ScalarType::UInt32:
            assert(rows == 1);
            switch (columns)
            {
            case 1:
                return ReflectionType::Type::Uint;
            case 2:
                return ReflectionType::Type::Uint2;
            case 3:
                return ReflectionType::Type::Uint3;
            case 4:
                return ReflectionType::Type::Uint4;
            }
        case TypeReflection::ScalarType::Int32:
            assert(rows == 1);
            switch (columns)
            {
            case 1:
                return ReflectionType::Type::Int;
            case 2:
                return ReflectionType::Type::Int2;
            case 3:
                return ReflectionType::Type::Int3;
            case 4:
                return ReflectionType::Type::Int4;
            }
        case TypeReflection::ScalarType::Float32:
            switch (rows)
            {
            case 1:
                switch (columns)
                {
                case 1:
                    return ReflectionType::Type::Float;
                case 2:
                    return ReflectionType::Type::Float2;
                case 3:
                    return ReflectionType::Type::Float3;
                case 4:
                    return ReflectionType::Type::Float4;
                }
                break;
            case 2:
                switch (columns)
                {
                case 2:
                    return ReflectionType::Type::Float2x2;
                case 3:
                    return ReflectionType::Type::Float2x3;
                case 4:
                    return ReflectionType::Type::Float2x4;
                }
                break;
            case 3:
                switch (columns)
                {
                case 2:
                    return ReflectionType::Type::Float3x2;
                case 3:
                    return ReflectionType::Type::Float3x3;
                case 4:
                    return ReflectionType::Type::Float3x4;
                }
                break;
            case 4:
                switch (columns)
                {
                case 2:
                    return ReflectionType::Type::Float4x2;
                case 3:
                    return ReflectionType::Type::Float4x3;
                case 4:
                    return ReflectionType::Type::Float4x4;
                }
                break;
            }
        }

        should_not_get_here();
        return ReflectionType::Type::Unknown;
    }

    static bool isResourceType(ReflectionType::Type t)
    {
        switch (t)
        {
        case ReflectionType::Type::Texture:
        case ReflectionType::Type::StructuredBuffer:
        case ReflectionType::Type::RawBuffer:
        case ReflectionType::Type::TypedBuffer:
        case ReflectionType::Type::Sampler:
        case ReflectionType::Type::ConstantBuffer:
            return true;
        default:
            return false;
        }
    }
    ReflectionVar::SharedPtr reflectVariable(VariableLayoutReflection* pSlangLayout, size_t offset, uint32_t bindIndex, uint32_t regSpace);

    ReflectionType::SharedPtr reflectType(TypeLayoutReflection* pSlangType, size_t offset, uint32_t bindIndex, uint32_t regSpace)
    {
        ReflectionType::Type type = getResourceType(pSlangType->getType());
        ReflectionType::Dimensions dims = ReflectionType::Dimensions::Unknown;
        ReflectionType::ShaderAccess shaderAccess = ReflectionType::ShaderAccess::Undefined;
        ReflectionType::ReturnType retType = ReflectionType::ReturnType::Unknown;

        if (type == ReflectionType::Type::Unknown)
        {
            type = getVariableType(pSlangType->unwrapArray()->getScalarType(), pSlangType->unwrapArray()->getRowCount(), pSlangType->unwrapArray()->getColumnCount());
        }
        else if (isResourceType(type))
        {
            dims = getResourceDimensions(pSlangType->getResourceShape());
            shaderAccess = getShaderAccess(pSlangType->getType());
            retType = getReturnType(pSlangType->getType());
        }

        uint32_t arraySize = 0;
        uint32_t arrayStride = 0;
        if (pSlangType->getType()->getKind() == TypeReflection::Kind::Array)
        {
            arraySize = (uint32_t)pSlangType->getElementCount();
            arrayStride = (uint32_t)pSlangType->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
        }

        auto pElementLayout = (type == ReflectionType::Type::Struct) ? pSlangType->unwrapArray() : pSlangType->unwrapArray()->getElementTypeLayout();
        ReflectionType::SharedPtr pType = ReflectionType::create(type, pElementLayout->getSize(), 0, arraySize, arrayStride, shaderAccess, retType, dims);

        for (uint32_t i = 0; i < pElementLayout->getFieldCount(); i++)
        {
            ReflectionVar::SharedPtr pVar = reflectVariable(pElementLayout->getFieldByIndex(i), offset, bindIndex, regSpace);
            pType->addMember(pVar);
        }
        return pType;
    }

    ReflectionVar::SharedPtr reflectVariable(VariableLayoutReflection* pSlangLayout, size_t offset, uint32_t bindIndex, uint32_t regSpace)
    {
        std::string name(pSlangLayout->getName());
        uint32_t index = pSlangLayout->getBindingIndex() + bindIndex;
        uint32_t space = pSlangLayout->getBindingSpace() + regSpace;
        size_t curOffset = (uint32_t)pSlangLayout->getOffset() + offset;
        ReflectionType::SharedPtr pType = reflectType(pSlangLayout->getTypeLayout(), curOffset, index, space);

        switch (pType->getType())
        {
        case ReflectionType::Type::Texture:
        case ReflectionType::Type::StructuredBuffer:
        case ReflectionType::Type::RawBuffer:
        case ReflectionType::Type::TypedBuffer:
        case ReflectionType::Type::Sampler:
        case ReflectionType::Type::ConstantBuffer:
            return ReflectionVar::create(name, pType, index, space);
        default:
            return ReflectionVar::create(name, pType, curOffset, space);
        }
    }

    ProgramReflection::SharedPtr ProgramReflection::create(slang::ShaderReflection* pSlangReflector, std::string& log)
    {
        return SharedPtr(new ProgramReflection(pSlangReflector, log));
    }

    ProgramReflection::ProgramReflection(slang::ShaderReflection* pSlangReflector, std::string& log)
    {
        ParameterBlockReflection::SharedPtr pGlobalBlock = ParameterBlockReflection::create("");
        for (uint32_t i = 0; i < pSlangReflector->getParameterCount(); i++)
        {
            VariableLayoutReflection* pSlangLayout = pSlangReflector->getParameterByIndex(i);
            ReflectionVar::SharedPtr pVar = reflectVariable(pSlangLayout, 0, 0, 0);

            if (isParameterBlockReflection(pSlangLayout, sParameterBlockRegistry))
            {
                std::string name = std::string(pSlangLayout->getName());
                ParameterBlockReflection::SharedPtr pBlock = ParameterBlockReflection::create(name);
                pBlock->addResource(pVar->getName(), pVar);
                addParameterBlock(pBlock);
            }
            else
            {
                pGlobalBlock->addResource(pVar->getName(), pVar);
            }
        }

        if (pGlobalBlock->isEmpty() == false)
        {
            addParameterBlock(pGlobalBlock);
        }
    }

    void ProgramReflection::addParameterBlock(const ParameterBlockReflection::SharedConstPtr& pBlock)
    {
        assert(mParameterBlocks.find(pBlock->getName()) == mParameterBlocks.end());
        mParameterBlocks[pBlock->getName()] = pBlock;
    }

    void ProgramReflection::registerParameterBlock(const std::string& name)
    {
        sParameterBlockRegistry.insert(name);
    }

    void ProgramReflection::unregisterParameterBlock(const std::string& name)
    {
        sParameterBlockRegistry.erase(name);
    }

    ReflectionType::SharedPtr ReflectionType::create(Type type, size_t size, uint32_t offset, uint32_t arraySize, uint32_t arrayStride, ShaderAccess shaderAccess, ReturnType retType, Dimensions dims)
    {
        return SharedPtr(new ReflectionType(type, size, offset, arraySize, arrayStride, shaderAccess, retType, dims));
    }

    ReflectionType::ReflectionType(Type type, size_t size, uint32_t offset, uint32_t arraySize, uint32_t arrayStride, ShaderAccess shaderAccess, ReturnType retType, Dimensions dims) :
        mType(type), mSize(size), mOffset(offset), mArraySize(arraySize), mArrayStride(arrayStride), mShaderAccess(shaderAccess), mReturnType(retType), mDimensions(dims)
    {

    }
    void ReflectionType::addMember(const std::shared_ptr<const ReflectionVar>& pVar)
    {
        assert(mNameToIndex.find(pVar->getName()) == mNameToIndex.end());
        mMembers.push_back(pVar);
        mNameToIndex[pVar->getName()] = mMembers.size() - 1;
    }

    ReflectionVar::SharedPtr ReflectionVar::create(const std::string& name, const ReflectionType::SharedConstPtr& pType, size_t offset, uint32_t regSpace)
    {
        return SharedPtr(new ReflectionVar(name, pType, offset, regSpace));
    }

    ReflectionVar::ReflectionVar(const std::string& name, const ReflectionType::SharedConstPtr& pType, size_t offset, uint32_t regSpace) : mName(name), mpType(pType), mOffset(offset), mRegSpace(regSpace)
    {

    }

    ParameterBlockReflection::SharedPtr ParameterBlockReflection::create(const std::string& name)
    {
        return SharedPtr(new ParameterBlockReflection(name));
    }

    ParameterBlockReflection::ParameterBlockReflection(const std::string& name) : mName(name)
    {

    }

    bool ParameterBlockReflection::isEmpty() const
    {
        return mResources.empty() && mConstantBuffers.empty() && mStructuredBuffers.empty() && mSamplers.empty();
    }

    static void flattenResources(const std::string& name, const ReflectionVar::SharedConstPtr& pVar, std::vector<std::pair<std::string, ReflectionVar::SharedConstPtr>>& pResources)
    {
        const ReflectionType* pType = pVar->getType().get();
        std::string namePrefix = name + (name.size() ? "." : "");
        for (const auto& pMember : *pType)
        {
            if (isResourceType(pMember->getType()->getType()))
            {
                pResources.push_back({ namePrefix + pMember->getName() , pMember });
                continue;
            }
            else
            {
                std::string newName = name + (name.size() ? "." : "");
                if (pMember->getType()->getArraySize())
                {
                    for (uint32_t j = 0; j < pMember->getType()->getArraySize(); j++)
                    {
                        flattenResources(namePrefix + pMember->getName() + '[' + std::to_string(j) + ']', pMember, pResources);
                    }
                }
                else
                {
                    flattenResources(namePrefix + pMember->getName(), pMember, pResources);
                }
            }
        }
    }

    void ParameterBlockReflection::addResource(const std::string& fullName, const ReflectionVar::SharedConstPtr& pVar)
    {
        decltype(mResources)* pMap = nullptr;

        switch (pVar->getType()->getType())
        {
        case ReflectionType::Type::ConstantBuffer:
            pMap = &mConstantBuffers;
            break;
        case ReflectionType::Type::Sampler:
            pMap = &mSamplers;
            break;
        case ReflectionType::Type::StructuredBuffer:
            pMap = &mStructuredBuffers;
            break;
        case ReflectionType::Type::Texture:
        case ReflectionType::Type::RawBuffer:
        case ReflectionType::Type::TypedBuffer:
            pMap = &mResources;
            break;
        default:
            break;
        }
        assert(pMap);
        assert((*pMap).find(fullName) == (*pMap).end());
        (*pMap)[fullName] = pVar;

        // If this is a constant-buffer, it might contain resources. Extract them.
        if (pVar->getType()->getType() == ReflectionType::Type::ConstantBuffer)
        {
            std::vector<std::pair<std::string, ReflectionVar::SharedConstPtr>> pResources;
            flattenResources("", pVar, pResources);
            for (const auto& r : pResources)
            {
                addResource(r.first, r.second);
            }
        }
    }
}