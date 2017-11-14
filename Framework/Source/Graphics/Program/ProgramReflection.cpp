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

    ReflectionVar::SharedPtr reflectVariable(VariableLayoutReflection* pSlangLayout, size_t offset, uint32_t bindIndex, uint32_t regSpace);
    ReflectionType::SharedPtr reflectType(TypeLayoutReflection* pSlangType, size_t offset, uint32_t bindIndex, uint32_t regSpace);

    ReflectionType::SharedPtr reflectResourceType(TypeLayoutReflection* pSlangType, size_t offset, uint32_t bindIndex, uint32_t regSpace)
    {
        ReflectionResourceType::Type type = getResourceType(pSlangType->getType());
        ReflectionResourceType::Dimensions dims = getResourceDimensions(pSlangType->getResourceShape());;
        ReflectionResourceType::ShaderAccess shaderAccess = getShaderAccess(pSlangType->getType());
        ReflectionResourceType::ReturnType retType = getReturnType(pSlangType->getType());
        ReflectionResourceType::StructuredType structuredType = getStructuredBufferType(pSlangType->getType());
        ReflectionResourceType::SharedPtr pType = ReflectionResourceType::create(type, bindIndex, regSpace, dims, structuredType, retType, shaderAccess);

        if (type == ReflectionResourceType::Type::ConstantBuffer || type == ReflectionResourceType::Type::StructuredBuffer)
        {
            const auto& pElementLayout = pSlangType->getElementTypeLayout();
            auto& pBufferType = reflectType(pElementLayout, offset, bindIndex, regSpace);
            ReflectionStructType::SharedPtr pStructType = std::dynamic_pointer_cast<ReflectionStructType>(pBufferType);
            pType->setStructType(pStructType);
        }

        return pType;
    }

    ReflectionType::SharedPtr reflectStructType(TypeLayoutReflection* pSlangType, size_t offset, uint32_t bindIndex, uint32_t regSpace)
    {
        ReflectionStructType::SharedPtr pType = ReflectionStructType::create(offset, pSlangType->getSize());
        for (uint32_t i = 0; i < pSlangType->getFieldCount(); i++)
        {
            ReflectionVar::SharedPtr pVar = reflectVariable(pSlangType->getFieldByIndex(i), offset, bindIndex, regSpace);
            pType->addMember(pVar);
        }
        return pType;
    }

    ReflectionType::SharedPtr reflectArrayType(TypeLayoutReflection* pSlangType, size_t offset, uint32_t bindIndex, uint32_t regSpace)
    {
        uint32_t arraySize = (uint32_t)pSlangType->getElementCount();
        uint32_t arrayStride = (uint32_t)pSlangType->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
        
        ReflectionType::SharedPtr pType = reflectType(pSlangType->getElementTypeLayout(), offset, bindIndex, regSpace);
        ReflectionArrayType::SharedPtr pArrayType = ReflectionArrayType::create(offset, arraySize, arrayStride, pType);
        return pArrayType;
    }

    ReflectionType::SharedPtr reflectBasicType(TypeLayoutReflection* pSlangType, size_t offset, uint32_t bindIndex, uint32_t regSpace)
    {
        ReflectionBasicType::Type type = getVariableType(pSlangType->getScalarType(), pSlangType->getRowCount(), pSlangType->getColumnCount());
        ReflectionType::SharedPtr pType = ReflectionBasicType::create(offset, type, true);
        return pType;
    }

    ReflectionType::SharedPtr reflectType(TypeLayoutReflection* pSlangType, size_t offset, uint32_t bindIndex, uint32_t regSpace)
    {
        auto kind = pSlangType->getType()->getKind();
        switch (kind)
        {
        case TypeReflection::Kind::Resource:
        case TypeReflection::Kind::SamplerState:
        case TypeReflection::Kind::ConstantBuffer:
        case TypeReflection::Kind::ShaderStorageBuffer:
        case TypeReflection::Kind::TextureBuffer:
            return reflectResourceType(pSlangType, offset, bindIndex, regSpace);
        case TypeReflection::Kind::Struct:
            return reflectStructType(pSlangType, offset, bindIndex, regSpace);
        case TypeReflection::Kind::Array:
            return reflectArrayType(pSlangType, offset, bindIndex, regSpace);
        default:
            return reflectBasicType(pSlangType, offset, bindIndex, regSpace);
        }
    }

    ReflectionVar::SharedPtr reflectVariable(VariableLayoutReflection* pSlangLayout, size_t offset, uint32_t bindIndex, uint32_t regSpace)
    {
        std::string name(pSlangLayout->getName());
        uint32_t index = pSlangLayout->getBindingIndex() + bindIndex;
        uint32_t space = pSlangLayout->getBindingSpace() + regSpace;
        size_t curOffset = (uint32_t)pSlangLayout->getOffset() + offset;;
        ReflectionType::SharedPtr pType = reflectType(pSlangLayout->getTypeLayout(), curOffset, index, space);

        const ReflectionResourceType* pResType = dynamic_cast<const ReflectionResourceType*>(pType.get());
        if (pResType)
        {
            return ReflectionVar::create(name, pType, index, space);
        }
        else
        {
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

    void ReflectionStructType::addMember(const std::shared_ptr<const ReflectionVar>& pVar)
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
        return mResources.empty() && mConstantBuffers.empty() && mStructuredBuffers.empty();
    }

    static void flattenResources(const std::string& name, const ReflectionVar::SharedConstPtr& pVar, std::vector<std::pair<std::string, ReflectionVar::SharedConstPtr>>& pResources)
    {
        const ReflectionType* pType = pVar->getType().get();
        std::string namePrefix = name + (name.size() ? "." : "");
        const ReflectionStructType* pStructType = dynamic_cast<const ReflectionStructType*>(pType);
        if(pStructType)
        for (const auto& pMember : *pStructType)
        {
            const ReflectionResourceType* pResourceType = dynamic_cast<const ReflectionResourceType*>(pMember->getType().get());
            if (pResourceType)
            {
                pResources.push_back({ namePrefix + pMember->getName() , pMember });
                continue;
            }
            else
            {
                std::string newName = name + (name.size() ? "." : "");
                const ReflectionArrayType* pArrayType = dynamic_cast<const ReflectionArrayType*>(pMember->getType().get());
                if (pArrayType)
                {
                    for (uint32_t j = 0; j < pArrayType->getArraySize(); j++)
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

    static const ReflectionVar* findVarCommon(const ParameterBlockReflection::ResourceMap& map, const std::string& name)
    {
        const auto& it = map.find(name);
        if (it == map.end())
        {
            return nullptr;
        }
        return it->second.get();
    }

    const ReflectionVar* ParameterBlockReflection::getResource(const std::string& name) const
    {
        return findVarCommon(mResources, name);
    }

    const ReflectionVar* ParameterBlockReflection::getConstantBuffer(const std::string& name) const
    {
        return findVarCommon(mConstantBuffers, name);
    }

    const ReflectionVar* ParameterBlockReflection::getStructuredBuffer(const std::string& name) const
    {
        return findVarCommon(mStructuredBuffers, name);
    }

    void ParameterBlockReflection::addResource(const std::string& fullName, const ReflectionVar::SharedConstPtr& pVar)
    {
        decltype(mResources)* pMap = nullptr;
        const ReflectionResourceType* pResourceType = dynamic_cast<const ReflectionResourceType*>(pVar->getType().get());
        assert(pResourceType);
        switch (pResourceType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
            pMap = &mConstantBuffers;
            break;
        case ReflectionResourceType::Type::StructuredBuffer:
            pMap = &mStructuredBuffers;
            break;
        case ReflectionResourceType::Type::Sampler:
        case ReflectionResourceType::Type::Texture:
        case ReflectionResourceType::Type::RawBuffer:
        case ReflectionResourceType::Type::TypedBuffer:
            pMap = &mResources;
            break;
        default:
            break;
        }
        assert(pMap);
        assert((*pMap).find(fullName) == (*pMap).end());
        (*pMap)[fullName] = pVar;

        // If this is a constant-buffer, it might contain resources. Extract them.
        if (pResourceType->getType() == ReflectionResourceType::Type::ConstantBuffer)
        {
            std::vector<std::pair<std::string, ReflectionVar::SharedConstPtr>> pResources;
            flattenResources("", pVar, pResources);
            for (const auto& r : pResources)
            {
                addResource(r.first, r.second);
            }
        }
    }

    const ParameterBlockReflection::SharedConstPtr& ProgramReflection::getParameterBlock(const std::string& name) const
    {
        return mParameterBlocks.at(name);
    }

    const ReflectionVar* ReflectionBasicType::findMember(const std::string& name) const
    {
        logWarning("Can't find variable + " + name);
        return nullptr;
    }

    const ReflectionVar* ReflectionResourceType::findMember(const std::string& name) const
    {
        if (mpStructType)
        {
            return mpStructType->findMember(name);
        }
        else
        {
            logWarning("Can't find variable + " + name);
            return nullptr;
        }
    }

    const ReflectionVar* ReflectionArrayType::findMember(const std::string& name) const
    {
        if (!name.size() || name[0] != '[')
        {
            logWarning("Looking for a variable named " + name + " which requires an array-index, but no index provided");
            return nullptr;
        }
        should_not_get_here();
        return nullptr;
    }

    const ReflectionVar* ReflectionStructType::findMember(const std::string& name) const
    {
        // Find the location of the next '.'
        size_t newPos = name.find('.');
        std::string field = name.substr(0, newPos);
        size_t fieldIndex = getMemberIndex(field);
        if (fieldIndex == ReflectionType::kInvalidOffset)
        {
            logWarning("Can't find variable + " + name);
            return nullptr;
        }

        const auto& pVar = getMember(fieldIndex).get();
        if (newPos == std::string::npos) return pVar;
        const auto& pNewType = pVar->getType().get();
        return pNewType->findMember(name.substr(newPos + 1));
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

    ReflectionArrayType::SharedPtr ReflectionArrayType::create(size_t offset, uint32_t arraySize, uint32_t arrayStride, const ReflectionType::SharedConstPtr& pType)
    {
        return SharedPtr(new ReflectionArrayType(offset, arraySize, arrayStride, pType));
    }

    ReflectionArrayType::ReflectionArrayType(size_t offset, uint32_t arraySize, uint32_t arrayStride, const ReflectionType::SharedConstPtr& pType) :
        ReflectionType(offset), mArraySize(arraySize), mArrayStride(arrayStride), mpType(pType) {}

    ReflectionResourceType::SharedPtr ReflectionResourceType::create(Type type, uint32_t regIndex, uint32_t regSpace, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess)
    {
        return SharedPtr(new ReflectionResourceType(type, regIndex, regSpace, structuredType, retType, shaderAccess));
    }

    ReflectionResourceType::ReflectionResourceType(Type type, uint32_t regIndex, uint32_t regSpace, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess) :
        ReflectionType(kInvalidOffset), mType(type), mRegIndex(regIndex), mRegSpace(regSpace), mStructuredType(structuredType), mReturnType(retType), mShaderAccess(shaderAccess) {}

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
}