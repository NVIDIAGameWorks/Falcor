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
    // Represents a "breadcrumb trail" leading from a particular variable
    // back to the path over member-access and array-indexing operations
    // that led to it.
    // E.g., when trying to construct information for `foo.bar[3].baz`
    // we might have a path that consists of:
    //
    // - An entry for the field `baz` in type `Bar` (which knows its offset)
    // - An entry for element 3 in the type `Bar[]`
    // - An entry for the field `bar` in type `Foo`
    // - An entry for the top-level shader parameter `foo`
    //
    // To compute the correct offset for `baz` we can walk up this chain
    // and add up offsets (taking element stride into account for arrays).
    //
    // In simple cases, one can track this info top-down, by simply keeping
    // a "running total" offset, but that doesn't account for the fact that
    // `baz` might be a texture, UAV, sampler, or uniform, and the offset
    // we'd need to track for each case is different.
    struct ReflectionPath
    {
        const ReflectionPath* pParent = nullptr;
        VariableLayoutReflection* pVar = nullptr;
        TypeLayoutReflection* pTypeLayout = nullptr;
        uint32_t childIndex = 0;
    };

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

    static ReflectionResourceType::Type getResourceType(TypeReflection* pSlangType)
    {
        switch (pSlangType->unwrapArray()->getKind())
        {
        case TypeReflection::Kind::ConstantBuffer:
            return ReflectionResourceType::Type::ConstantBuffer;
        case TypeReflection::Kind::SamplerState:
            return ReflectionResourceType::Type::Sampler;
        case TypeReflection::Kind::ShaderStorageBuffer:
            return ReflectionResourceType::Type::StructuredBuffer;
        case TypeReflection::Kind::Resource:
            switch (pSlangType->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK)
            {
            case SLANG_STRUCTURED_BUFFER:
                return ReflectionResourceType::Type::StructuredBuffer;

            case SLANG_BYTE_ADDRESS_BUFFER:
                return ReflectionResourceType::Type::RawBuffer;
            case SLANG_TEXTURE_BUFFER:
                return ReflectionResourceType::Type::TypedBuffer;
            default:
                return ReflectionResourceType::Type::Texture;
            }
            break;
        default:
            should_not_get_here();
            return ReflectionResourceType::Type(-1);
        }
    }

    static ReflectionResourceType::ShaderAccess getShaderAccess(TypeReflection* pSlangType)
    {
        // Compute access for an array using the underlying type...
        pSlangType = pSlangType->unwrapArray();

        switch (pSlangType->getKind())
        {
        case TypeReflection::Kind::SamplerState:
        case TypeReflection::Kind::ConstantBuffer:
            return ReflectionResourceType::ShaderAccess::Read;
            break;

        case TypeReflection::Kind::Resource:
        case TypeReflection::Kind::ShaderStorageBuffer:
            switch (pSlangType->getResourceAccess())
            {
            case SLANG_RESOURCE_ACCESS_NONE:
                return ReflectionResourceType::ShaderAccess::Undefined;

            case SLANG_RESOURCE_ACCESS_READ:
                return ReflectionResourceType::ShaderAccess::Read;

            default:
                return ReflectionResourceType::ShaderAccess::ReadWrite;
            }
            break;

        default:
            return ReflectionResourceType::ShaderAccess::Undefined;
        }
    }

    static ReflectionResourceType::ReturnType getReturnType(TypeReflection* pType)
    {
        // Could be a resource that doesn't have a specific element type (e.g., a raw buffer)
        if (!pType)
            return ReflectionResourceType::ReturnType::Unknown;

        switch (pType->getScalarType())
        {
        case TypeReflection::ScalarType::Float32:
            return ReflectionResourceType::ReturnType::Float;
        case TypeReflection::ScalarType::Int32:
            return ReflectionResourceType::ReturnType::Int;
        case TypeReflection::ScalarType::UInt32:
            return ReflectionResourceType::ReturnType::Uint;
        case TypeReflection::ScalarType::Float64:
            return ReflectionResourceType::ReturnType::Double;

            // Could be a resource that uses an aggregate element type (e.g., a structured buffer)
        case TypeReflection::ScalarType::None:
            return ReflectionResourceType::ReturnType::Unknown;

        default:
            return ReflectionResourceType::ReturnType::Unknown;
        }
    }

    static ReflectionResourceType::Dimensions getResourceDimensions(SlangResourceShape shape)
    {
        switch (shape)
        {
        case SLANG_TEXTURE_1D:
            return ReflectionResourceType::Dimensions::Texture1D;
        case SLANG_TEXTURE_1D_ARRAY:
            return ReflectionResourceType::Dimensions::Texture1DArray;
        case SLANG_TEXTURE_2D:
            return ReflectionResourceType::Dimensions::Texture2D;
        case SLANG_TEXTURE_2D_ARRAY:
            return ReflectionResourceType::Dimensions::Texture2DArray;
        case SLANG_TEXTURE_2D_MULTISAMPLE:
            return ReflectionResourceType::Dimensions::Texture2DMS;
        case SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return ReflectionResourceType::Dimensions::Texture2DMSArray;
        case SLANG_TEXTURE_3D:
            return ReflectionResourceType::Dimensions::Texture3D;
        case SLANG_TEXTURE_CUBE:
            return ReflectionResourceType::Dimensions::TextureCube;
        case SLANG_TEXTURE_CUBE_ARRAY:
            return ReflectionResourceType::Dimensions::TextureCubeArray;

        case SLANG_TEXTURE_BUFFER:
        case SLANG_STRUCTURED_BUFFER:
        case SLANG_BYTE_ADDRESS_BUFFER:
            return ReflectionResourceType::Dimensions::Buffer;

        default:
            return ReflectionResourceType::Dimensions::Unknown;
        }
    }

    ReflectionBasicType::Type getVariableType(TypeReflection::ScalarType slangScalarType, uint32_t rows, uint32_t columns)
    {
        switch (slangScalarType)
        {
        case TypeReflection::ScalarType::Bool:
            assert(rows == 1);
            switch (columns)
            {
            case 1:
                return ReflectionBasicType::Type::Bool;
            case 2:
                return ReflectionBasicType::Type::Bool2;
            case 3:
                return ReflectionBasicType::Type::Bool3;
            case 4:
                return ReflectionBasicType::Type::Bool4;
            }
        case TypeReflection::ScalarType::UInt32:
            assert(rows == 1);
            switch (columns)
            {
            case 1:
                return ReflectionBasicType::Type::Uint;
            case 2:
                return ReflectionBasicType::Type::Uint2;
            case 3:
                return ReflectionBasicType::Type::Uint3;
            case 4:
                return ReflectionBasicType::Type::Uint4;
            }
        case TypeReflection::ScalarType::Int32:
            assert(rows == 1);
            switch (columns)
            {
            case 1:
                return ReflectionBasicType::Type::Int;
            case 2:
                return ReflectionBasicType::Type::Int2;
            case 3:
                return ReflectionBasicType::Type::Int3;
            case 4:
                return ReflectionBasicType::Type::Int4;
            }
        case TypeReflection::ScalarType::Float32:
            switch (rows)
            {
            case 1:
                switch (columns)
                {
                case 1:
                    return ReflectionBasicType::Type::Float;
                case 2:
                    return ReflectionBasicType::Type::Float2;
                case 3:
                    return ReflectionBasicType::Type::Float3;
                case 4:
                    return ReflectionBasicType::Type::Float4;
                }
                break;
            case 2:
                switch (columns)
                {
                case 2:
                    return ReflectionBasicType::Type::Float2x2;
                case 3:
                    return ReflectionBasicType::Type::Float2x3;
                case 4:
                    return ReflectionBasicType::Type::Float2x4;
                }
                break;
            case 3:
                switch (columns)
                {
                case 2:
                    return ReflectionBasicType::Type::Float3x2;
                case 3:
                    return ReflectionBasicType::Type::Float3x3;
                case 4:
                    return ReflectionBasicType::Type::Float3x4;
                }
                break;
            case 4:
                switch (columns)
                {
                case 2:
                    return ReflectionBasicType::Type::Float4x2;
                case 3:
                    return ReflectionBasicType::Type::Float4x3;
                case 4:
                    return ReflectionBasicType::Type::Float4x4;
                }
                break;
            }
        }

        should_not_get_here();
        return ReflectionBasicType::Type(-1);
    }

    static ReflectionResourceType::StructuredType getStructuredBufferType(TypeReflection* pSlangType)
    {
        auto invalid = ReflectionResourceType::StructuredType::Invalid;

        if (pSlangType->getKind() != TypeReflection::Kind::Resource)
            return invalid; // not a structured buffer

        if (pSlangType->getResourceShape() != SLANG_STRUCTURED_BUFFER)
            return invalid; // not a structured buffer

        switch (pSlangType->getResourceAccess())
        {
        default:
            should_not_get_here();
            return invalid;

        case SLANG_RESOURCE_ACCESS_READ:
            return ReflectionResourceType::StructuredType::Default;

        case SLANG_RESOURCE_ACCESS_READ_WRITE:
        case SLANG_RESOURCE_ACCESS_RASTER_ORDERED:
            return ReflectionResourceType::StructuredType::Counter;
        case SLANG_RESOURCE_ACCESS_APPEND:
            return ReflectionResourceType::StructuredType::Append;
        case SLANG_RESOURCE_ACCESS_CONSUME:
            return ReflectionResourceType::StructuredType::Consume;
        }
    };

    ReflectionVar::SharedPtr reflectVariable(VariableLayoutReflection* pSlangLayout, const ReflectionPath* pPath);
    ReflectionType::SharedPtr reflectType(TypeLayoutReflection* pSlangType, const ReflectionPath* pPath);


    static size_t getRegisterIndexFromPath(const ReflectionPath* pPath, SlangParameterCategory category)
    {
        uint32_t offset = 0;
        for (auto pp = pPath; pp; pp = pp->pParent)
        {
            if (pp->pVar)
            {
                const auto& h = pp->pVar->getName();
                offset += (uint32_t)pp->pVar->getOffset(category);
                continue;
            }
            else if (pp->pTypeLayout)
            {
                switch (pp->pTypeLayout->getKind())
                {
                case TypeReflection::Kind::Array:
                    offset += (uint32_t)pp->pTypeLayout->getElementStride(category) * pp->childIndex;
                    continue;

                case TypeReflection::Kind::Struct:
                    offset += (uint32_t)pp->pTypeLayout->getFieldByIndex(int(pp->childIndex))->getOffset(category);
                    continue;

                default:
                    break;
                }
            }

            logError("internal error: invalid reflection path");
            return 0;
        }
        return offset;
    }

    static size_t getUniformOffset(const ReflectionPath* pPath)
    {
        return getRegisterIndexFromPath(pPath, SLANG_PARAMETER_CATEGORY_UNIFORM);
    }

    static uint32_t getRegisterSpaceFromPath(const ReflectionPath* pPath, SlangParameterCategory category)
    {
        return (uint32_t)pPath->pVar->getBindingSpace(category);
    }

    ReflectionType::SharedPtr reflectResourceType(TypeLayoutReflection* pSlangType, const ReflectionPath* pPath)
    {
        ReflectionResourceType::Type type = getResourceType(pSlangType->getType());
        ReflectionResourceType::Dimensions dims = getResourceDimensions(pSlangType->getResourceShape());;
        ReflectionResourceType::ShaderAccess shaderAccess = getShaderAccess(pSlangType->getType());
        ReflectionResourceType::ReturnType retType = getReturnType(pSlangType->getType());
        ReflectionResourceType::StructuredType structuredType = getStructuredBufferType(pSlangType->getType());
        ReflectionResourceType::SharedPtr pType = ReflectionResourceType::create(type, dims, structuredType, retType, shaderAccess);

        if (type == ReflectionResourceType::Type::ConstantBuffer || type == ReflectionResourceType::Type::StructuredBuffer)
        {
            const auto& pElementLayout = pSlangType->getElementTypeLayout();
            auto& pBufferType = reflectType(pElementLayout, pPath);
            ReflectionStructType::SharedPtr pStructType = std::dynamic_pointer_cast<ReflectionStructType>(pBufferType);
            pType->setStructType(pStructType);
        }

        return pType;
    }

    ReflectionType::SharedPtr reflectStructType(TypeLayoutReflection* pSlangType, const ReflectionPath* pPath)
    {
        ReflectionStructType::SharedPtr pType = ReflectionStructType::create(getUniformOffset(pPath), pSlangType->getSize());
        for (uint32_t i = 0; i < pSlangType->getFieldCount(); i++)
        {
            ReflectionPath fieldPath;
            fieldPath.pParent = pPath;
            fieldPath.pTypeLayout = pSlangType;
            fieldPath.childIndex = i;
            fieldPath.pVar = pSlangType->getFieldByIndex(i);
            ReflectionVar::SharedPtr pVar = reflectVariable(fieldPath.pVar, &fieldPath);
            pType->addMember(pVar);
        }
        return pType;
    }

    ReflectionType::SharedPtr reflectArrayType(TypeLayoutReflection* pSlangType, const ReflectionPath* pPath)
    {
        uint32_t arraySize = (uint32_t)pSlangType->getElementCount();
        uint32_t arrayStride = (uint32_t)pSlangType->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);

        ReflectionPath newPath;
        newPath.pParent = pPath;
        newPath.pTypeLayout = pSlangType;
        newPath.childIndex = 0;

        ReflectionType::SharedPtr pType = reflectType(pSlangType->getElementTypeLayout(), &newPath);
        ReflectionArrayType::SharedPtr pArrayType = ReflectionArrayType::create(getUniformOffset(pPath), arraySize, arrayStride, pType);
        return pArrayType;
    }

    ReflectionType::SharedPtr reflectBasicType(TypeLayoutReflection* pSlangType, const ReflectionPath* pPath)
    {
        ReflectionBasicType::Type type = getVariableType(pSlangType->getScalarType(), pSlangType->getRowCount(), pSlangType->getColumnCount());
        ReflectionType::SharedPtr pType = ReflectionBasicType::create(getUniformOffset(pPath), type, false);
        return pType;
    }

    ReflectionType::SharedPtr reflectType(TypeLayoutReflection* pSlangType, const ReflectionPath* pPath)
    {
        auto kind = pSlangType->getType()->getKind();
        switch (kind)
        {
        case TypeReflection::Kind::Resource:
        case TypeReflection::Kind::SamplerState:
        case TypeReflection::Kind::ConstantBuffer:
        case TypeReflection::Kind::ShaderStorageBuffer:
        case TypeReflection::Kind::TextureBuffer:
            return reflectResourceType(pSlangType, pPath);
        case TypeReflection::Kind::Struct:
            return reflectStructType(pSlangType, pPath);
        case TypeReflection::Kind::Array:
            return reflectArrayType(pSlangType, pPath);
        default:
            return reflectBasicType(pSlangType, pPath);
        }
    }

    static ParameterCategory getParameterCategory(VariableLayoutReflection* pSlangLayout)
    {
        const auto& pTypeLayout = pSlangLayout->getTypeLayout();
        ParameterCategory category = pTypeLayout->getParameterCategory();
        if (category == ParameterCategory::Mixed && pTypeLayout->getKind() == TypeReflection::Kind::ConstantBuffer)
        {
            category = ParameterCategory::ConstantBuffer;
        }
        return category;
    }

    ReflectionVar::SharedPtr reflectVariable(VariableLayoutReflection* pSlangLayout, const ReflectionPath* pPath)
    {
        assert(pPath);
        std::string name(pSlangLayout->getName());
        if (name == "gCsmData")
        {
            name = name;
        }
        ReflectionType::SharedPtr pType = reflectType(pSlangLayout->getTypeLayout(), pPath);
        ReflectionVar::SharedPtr pVar;

        if (pType->unwrapArray()->asResourceType())
        {
            ParameterCategory category = getParameterCategory(pSlangLayout);
            uint32_t index = (uint32_t)getRegisterIndexFromPath(pPath, category);
            uint32_t space = getRegisterSpaceFromPath(pPath, category);
            pVar = ReflectionVar::create(name, pType, index, space);
        }
        else
        {
            pVar = ReflectionVar::create(name, pType, getUniformOffset(pPath), 0);
        }
        return pVar;
    }

    ReflectionVar::SharedPtr reflectTopLevelVariable(VariableLayoutReflection* pSlangLayout)
    {
        ReflectionPath path;
        path.pVar = pSlangLayout;
        return reflectVariable(pSlangLayout, &path);
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
            ReflectionVar::SharedPtr pVar = reflectTopLevelVariable(pSlangLayout);

            if (isParameterBlockReflection(pSlangLayout, sParameterBlockRegistry))
            {
                std::string name = std::string(pSlangLayout->getName());
                ParameterBlockReflection::SharedPtr pBlock = ParameterBlockReflection::create(name);
                pBlock->addResource(pVar);
                addParameterBlock(pBlock);
            }
            else
            {
                pGlobalBlock->addResource(pVar);
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
        if (pBlock->getName().size() == 0) mpGlobalBlock = pBlock;
    }

    void ProgramReflection::registerParameterBlock(const std::string& name)
    {
        sParameterBlockRegistry.insert(name);
    }

    void ProgramReflection::unregisterParameterBlock(const std::string& name)
    {
        sParameterBlockRegistry.erase(name);
    }

    void ReflectionStructType::addMember(const std::shared_ptr<const ReflectionVar>& pVar)
    {
        if (mNameToIndex.find(pVar->getName()) != mNameToIndex.end())
        {
            size_t index = mNameToIndex[pVar->getName()];
            if (*pVar != *mMembers[index])
            {
                logError("Mismatch in variable declarations between different shader stages. Variable name is `" + pVar->getName() + "', struct name is '" + mName);
            }
            return;
        }
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
        mpResourceVars = ReflectionStructType::create(0, 0);
    }

    bool ParameterBlockReflection::isEmpty() const
    {
        return mResources.empty();
    }

    static void flattenResources(const ReflectionVar::SharedConstPtr& pVar, ParameterBlockReflection::ResourceVec& resources)
    {
        const ReflectionType* pType = pVar->getType().get();
        if (pType->asResourceType())
        {
            resources.push_back(pVar);
            return;
        }
        const ReflectionArrayType* pArrayType = pType->asArrayType();

        if (pArrayType)
        {
            if (pArrayType->unwrapArray()->asResourceType())
            {
                resources.push_back(pVar);
            }
            return;
        }

        const ReflectionStructType* pStructType = pType->asStructType();
        if (pStructType)
        {
            for (const auto& pMember : *pStructType)
            {
                flattenResources(pMember, resources);
            }
        }
    }

    static void flattenResources(const ReflectionStructType* pStructType, ParameterBlockReflection::ResourceVec& resources)
    {
        for (const auto& pMember : *pStructType)
        {
            flattenResources(pMember, resources);
        }
    }

    static bool doesTypeContainsResources(const ReflectionType* pType)
    {
        const ReflectionType* pUnwrapped = pType->unwrapArray();
        if (pUnwrapped->asResourceType()) return true;
        const ReflectionStructType* pStruct = pUnwrapped->asStructType();
        if (pStruct)
        {
            for (const auto& pMember : *pStruct)
            {
                if (doesTypeContainsResources(pMember->getType().get())) return true;
            }
        }
        return false;
    }

    const ReflectionVar::SharedConstPtr ParameterBlockReflection::getResource(const std::string& name) const
    {
        return mpResourceVars->findMember(name);
    }

    void ParameterBlockReflection::addResource(const ReflectionVar::SharedConstPtr& pVar)
    {
        const ReflectionResourceType* pResourceType = pVar->getType()->unwrapArray()->asResourceType();
        assert(pResourceType);
        mResources.push_back(pVar);
        mpResourceVars->addMember(pVar);

        // If this is a constant-buffer, it might contain resources. Extract them.
        if (pResourceType->getType() == ReflectionResourceType::Type::ConstantBuffer)
        {
            const ReflectionStructType* pStruct = pResourceType->getStructType().get();
            assert(pStruct);
            for (const auto& pMember : *pStruct)
            {
                if (doesTypeContainsResources(pMember->getType().get()))
                {
                    mpResourceVars->addMember(pMember);
                }
            }
            flattenResources(pStruct, mResources);
        }
    }

    const ParameterBlockReflection::SharedConstPtr& ProgramReflection::getParameterBlock(const std::string& name) const
    {
        return mParameterBlocks.at(name);
    }

    ReflectionVar::SharedConstPtr ReflectionType::findMember(const std::string& name) const
    {
        return findMemberInternal(name, 0, 0, 0, 0);
    }

    ReflectionVar::SharedConstPtr ReflectionBasicType::findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace) const
    {
        // We shouldn't get here
        logWarning("Can't find variable + " + name);
        return nullptr;
    }

    ReflectionVar::SharedConstPtr ReflectionResourceType::findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace) const
    {
        if (mpStructType)
        {
            return mpStructType->findMemberInternal(name, strPos, offset, regIndex, regSpace);
        }
        else
        {
            logWarning("Can't find variable '" + name + "'");
            return nullptr;
        }
    }

    ReflectionVar::SharedConstPtr ReflectionArrayType::findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace) const
    {
        if (name[strPos] == '[') ++strPos;
        if (name.size() <= strPos)
        {
            logWarning("Looking for a variable named " + name + " which requires an array-index, but no index provided");
            return nullptr;
        }
        size_t endPos = name.find(']', strPos);
        if (endPos == std::string::npos)
        {
            logWarning("Missing `]` when parsing array variable '" + name + "'");
            return nullptr;
        }

        // Get the array index
        std::string indexStr = name.substr(strPos, endPos);
        uint32_t index = (uint32_t)std::stoi(indexStr);
        if (index >= mArraySize)
        {
            logWarning("Array index out of range when parsing variable '" + name + "'. Must be less than " + std::to_string(mArraySize));
            return nullptr;
        }
        offset += index * mArrayStride;
        // Find the offset of the leaf
        if (endPos + 1 == name.size())
        {
            ReflectionVar::SharedPtr pVar = ReflectionVar::create(name.substr(0, endPos + 1), mpType, offset, regSpace);
            return pVar;
        }

        return mpType->findMemberInternal(name, endPos + 1, offset, regIndex, regSpace);
    }

    ReflectionVar::SharedConstPtr ReflectionStructType::findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace) const
    {
        if (name[strPos] == '.') strPos++; // This happens for arrays-of-structs. The array will only skip the array index, which means the first character will be a '.'

        // Find the location of the next '.'
        size_t newPos = name.find_first_of(".[", strPos);
        std::string field = name.substr(strPos, newPos - strPos);
        size_t fieldIndex = getMemberIndex(field);
        if (fieldIndex == ReflectionType::kInvalidOffset)
        {
            logWarning("Can't find variable '" + name + "'");
            return nullptr;
        }

        const auto& pVar = getMember(fieldIndex);
        if (newPos == std::string::npos) return pVar;
        const auto& pNewType = pVar->getType().get();
        return pNewType->findMemberInternal(name, newPos + 1, pVar->getOffset(), pVar->getRegisterIndex(), pVar->getRegisterSpace());
    }

    size_t ReflectionStructType::getMemberIndex(const std::string& name) const
    {
        auto& it = mNameToIndex.find(name);
        if (it == mNameToIndex.end()) return kInvalidOffset;
        return it->second;
    }

    const ReflectionResourceType* ReflectionType::asResourceType() const
    {
        return dynamic_cast<const ReflectionResourceType*>(this);
    }

    const ReflectionBasicType* ReflectionType::asBasicType() const
    {
        return dynamic_cast<const ReflectionBasicType*>(this);
    }

    const ReflectionStructType* ReflectionType::asStructType() const
    {
        return dynamic_cast<const ReflectionStructType*>(this);
    }

    const ReflectionArrayType* ReflectionType::asArrayType() const
    {
        return dynamic_cast<const ReflectionArrayType*>(this);
    }

    const ReflectionType* ReflectionType::unwrapArray() const
    {
        const ReflectionType* pType = this;
        while (pType->asArrayType())
        {
            pType = pType->asArrayType()->getType().get();
        }
        return pType;
    }

    uint32_t ReflectionType::getTotalArraySize() const
    {
        const ReflectionArrayType* pArray = asArrayType();
        if (pArray == nullptr) return 0;
        uint32_t arraySize = 1;
        while (pArray)
        {
            arraySize *= pArray->getArraySize();
            pArray = pArray->getType()->asArrayType();
        }
        return arraySize;
    }

    ReflectionArrayType::SharedPtr ReflectionArrayType::create(size_t offset, uint32_t arraySize, uint32_t arrayStride, const ReflectionType::SharedConstPtr& pType)
    {
        return SharedPtr(new ReflectionArrayType(offset, arraySize, arrayStride, pType));
    }

    ReflectionArrayType::ReflectionArrayType(size_t offset, uint32_t arraySize, uint32_t arrayStride, const ReflectionType::SharedConstPtr& pType) :
        ReflectionType(offset), mArraySize(arraySize), mArrayStride(arrayStride), mpType(pType) {}

    ReflectionResourceType::SharedPtr ReflectionResourceType::create(Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess)
    {
        return SharedPtr(new ReflectionResourceType(type, dims, structuredType, retType, shaderAccess));
    }

    ReflectionResourceType::ReflectionResourceType(Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess) :
        ReflectionType(kInvalidOffset), mType(type), mStructuredType(structuredType), mReturnType(retType), mShaderAccess(shaderAccess), mDimensions(dims) {}

    ReflectionBasicType::SharedPtr ReflectionBasicType::create(size_t offset, Type type, bool isRowMajor)
    {
        return SharedPtr(new ReflectionBasicType(offset, type, isRowMajor));
    }

    ReflectionBasicType::ReflectionBasicType(size_t offset, Type type, bool isRowMajor) :
        ReflectionType(offset), mType(type), mIsRowMajor(isRowMajor) {}

    ReflectionStructType::SharedPtr ReflectionStructType::create(size_t offset, size_t size, const std::string& name)
    {
        return SharedPtr(new ReflectionStructType(offset, size, name));
    }

    ReflectionStructType::ReflectionStructType(size_t offset, size_t size, const std::string& name) :
        ReflectionType(offset), mSize(size), mName(name) {}

    ProgramReflection::ResourceBinding ProgramReflection::getResourceBinding(const std::string& name) const
    {
        ResourceBinding binding;
        if (mpGlobalBlock == nullptr) return binding;
        // Search the constant-buffers
        const ReflectionVar* pVar = mpGlobalBlock->getResource(name).get();
        if (pVar)
        {
            binding.regIndex = pVar->getRegisterIndex();
            binding.regSpace = pVar->getRegisterSpace();
        }
        return binding;
    }

    bool ReflectionArrayType::operator==(const ReflectionType& other) const
    {
        const ReflectionArrayType* pOther = other.asArrayType();
        if (!pOther) return false;
        return (*this == *pOther);
    }

    bool ReflectionResourceType::operator==(const ReflectionType& other) const
    {
        const ReflectionResourceType* pOther = other.asResourceType();
        if (!pOther) return false;
        return (*this == *pOther);
    }

    bool ReflectionStructType::operator==(const ReflectionType& other) const
    {
        const ReflectionStructType* pOther = other.asStructType();
        if (!pOther) return false;
        return (*this == *pOther);
    }

    bool ReflectionBasicType::operator==(const ReflectionType& other) const
    {
        const ReflectionBasicType* pOther = other.asBasicType();
        if (!pOther) return false;
        return (*this == *pOther);
    }

    bool ReflectionArrayType::operator==(const ReflectionArrayType& other) const
    {
        if (mArraySize != other.mArraySize) return false;
        if (mArrayStride != other.mArrayStride) return false;
        if (*mpType != *other.mpType) return false;
        return true;
    }

    bool ReflectionStructType::operator==(const ReflectionStructType& other) const
    {
        // We only care about the struct layout. Checking the members should be enough
        if (mMembers.size() != other.mMembers.size()) return false;
        for (size_t i = 0 ; i < mMembers.size() ; i++)
        {
            // Theoretically, the order of the struct members should match
            if (*mMembers[i] != *other.mMembers[i]) return false;
        }
        return true;
    }

    bool ReflectionBasicType::operator==(const ReflectionBasicType& other) const
    {
        if (mType != other.mType) return false;
        if (mIsRowMajor != other.mIsRowMajor) return false;
        return true;
    }

    bool ReflectionResourceType::operator==(const ReflectionResourceType& other) const
    {
        if (mDimensions != other.mDimensions) return false;
        if (mStructuredType != other.mStructuredType) return false;
        if (mReturnType != other.mReturnType) return false;
        if(mShaderAccess != other.mShaderAccess) return false;
        if(mType != other.mType) return false;
        bool hasStruct = (mpStructType != nullptr);
        bool otherHasStruct = (other.mpStructType != nullptr);
        if (hasStruct != otherHasStruct) return false;
        if (hasStruct && (*mpStructType != *other.mpStructType)) return false;

        return true;
    }

    bool ReflectionVar::operator==(const ReflectionVar& other) const
    {
        if (*mpType != *other.mpType) return false;
        if (mOffset != other.mOffset) return false;
        if (mRegSpace != other.mRegSpace) return false;
        if (mName != other.mName) return false;
        if (mSize != other.mSize) return false;

        return true;
    }
}
