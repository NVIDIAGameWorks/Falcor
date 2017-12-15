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
#include "ProgramReflection.h"
#include "Utils/StringUtils.h"
using namespace slang;

namespace Falcor
{
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

    static ReflectionResourceType::Type getResourceType(TypeReflection* pSlangType)
    {
        switch (pSlangType->unwrapArray()->getKind())
        {
        case TypeReflection::Kind::ParameterBlock:
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

    // Once we've found the path from the root down to a particular leaf
    // variable, `getDescOffset` can be used to find the final summed-up descriptor offset of the element
    static uint32_t getDescOffsetFromPath(const ReflectionPath* pPath, uint32_t arraySize, SlangParameterCategory category)
    {
#ifndef FALCOR_VK
        return 0;
#else
        uint32_t offset = 0;
        for (auto pp = pPath; pp; pp = pp->pParent)
        {
            if ((pp->pTypeLayout) && (pp->pTypeLayout->getKind() == TypeReflection::Kind::Array))
            {
                offset += (uint32_t)pp->childIndex * arraySize;
                arraySize *= (uint32_t)pp->pTypeLayout->getElementCount();
            }
        }
        return offset;
#endif
    }

    static size_t getRegisterIndexFromPath(const ReflectionPath* pPath, SlangParameterCategory category)
    {
        uint32_t offset = 0;
        for (auto pp = pPath; pp; pp = pp->pParent)
        {
            if (pp->pVar)
            {
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
        uint32_t offset = 0;
        for (auto pp = pPath; pp; pp = pp->pParent)
        {
            if (pp->pVar)
            {
                offset += (uint32_t)pp->pVar->getBindingSpace(category);
                continue;
            }
            else if (pp->pTypeLayout)
            {
                switch (pp->pTypeLayout->getKind())
                {
                case TypeReflection::Kind::Array:
                    offset += (uint32_t)pp->pTypeLayout->getElementStride(SLANG_PARAMETER_CATEGORY_REGISTER_SPACE) * pp->childIndex;
                    continue;

                case TypeReflection::Kind::Struct:
                    offset += (uint32_t)pp->pTypeLayout->getFieldByIndex(int(pp->childIndex))->getBindingSpace(category);
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

    ReflectionType::SharedPtr reflectResourceType(TypeLayoutReflection* pSlangType, const ReflectionPath* pPath)
    {
        ReflectionResourceType::Type type = getResourceType(pSlangType->getType());
        ReflectionResourceType::Dimensions dims = getResourceDimensions(pSlangType->getResourceShape());;
        ReflectionResourceType::ShaderAccess shaderAccess = getShaderAccess(pSlangType->getType());
        ReflectionResourceType::ReturnType retType = getReturnType(pSlangType->getType());
        ReflectionResourceType::StructuredType structuredType = getStructuredBufferType(pSlangType->getType());
        ReflectionResourceType::SharedPtr pType = ReflectionResourceType::create(type, dims, structuredType, retType, shaderAccess);

        // #PARAMBLOCK Perhaps we can check if pSlangType->getElementTypeLayout() returns nullptr
        if (type == ReflectionResourceType::Type::ConstantBuffer || type == ReflectionResourceType::Type::StructuredBuffer)
        {
            const auto& pElementLayout = pSlangType->getElementTypeLayout();
            auto pBufferType = reflectType(pElementLayout, pPath);
            pType->setStructType(pBufferType);
        }

        return pType;
    }

    ReflectionType::SharedPtr reflectStructType(TypeLayoutReflection* pSlangType, const ReflectionPath* pPath)
    {
        ReflectionStructType::SharedPtr pType = ReflectionStructType::create(getUniformOffset(pPath), pSlangType->getSize(), "");
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
        ReflectionType::SharedPtr pType = ReflectionBasicType::create(getUniformOffset(pPath), type, false, pSlangType->getSize());
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
        case TypeReflection::Kind::ParameterBlock:
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
        if (category == ParameterCategory::Mixed)
        {
            switch (pTypeLayout->getKind())
            {
            case TypeReflection::Kind::ConstantBuffer:
            case TypeReflection::Kind::ParameterBlock:
                category = ParameterCategory::ConstantBuffer;
                break;
            }
        }
        return category;
    }

    ReflectionVar::SharedPtr reflectVariable(VariableLayoutReflection* pSlangLayout, const ReflectionPath* pPath)
    {
        assert(pPath);
        std::string name(pSlangLayout->getName());

        ReflectionType::SharedPtr pType = reflectType(pSlangLayout->getTypeLayout(), pPath);
        ReflectionVar::SharedPtr pVar;

        if (pType->unwrapArray()->asResourceType())
        {
            ParameterCategory category = getParameterCategory(pSlangLayout);
            uint32_t index = (uint32_t)getRegisterIndexFromPath(pPath, category);
            uint32_t space = getRegisterSpaceFromPath(pPath, category);
            uint32_t descOffset = getDescOffsetFromPath(pPath, max(1u, pType->getTotalArraySize()), category);
            pVar = ReflectionVar::create(name, pType, index, descOffset, space);
        }
        else
        {
            pVar = ReflectionVar::create(name, pType, pSlangLayout->getOffset());
        }
        return pVar;
    }

    ReflectionVar::SharedPtr reflectTopLevelVariable(VariableLayoutReflection* pSlangLayout)
    {
        ReflectionPath path;
        path.pVar = pSlangLayout;
        return reflectVariable(pSlangLayout, &path);
    }

    static void storeShaderVariable(const ReflectionPath& path, SlangParameterCategory category, const std::string& name, ProgramReflection::VariableMap& varMap, ProgramReflection::VariableMap* pVarMapBySemantic, uint32_t count, uint32_t stride)
    {
        ProgramReflection::ShaderVariable var;
        const auto& pTypeLayout = path.pVar->getTypeLayout();
        var.type = getVariableType(pTypeLayout->getScalarType(), pTypeLayout->getRowCount(), pTypeLayout->getColumnCount());

        uint32_t baseIndex = (uint32_t)getRegisterIndexFromPath(&path, category);
        for(uint32_t i = 0 ; i < max(count, 1u) ; i++)
        {
            var.bindLocation = baseIndex + (i*stride);
            var.semanticName = path.pVar->getSemanticName();
            if (count)
            {
                var.semanticName += '[' + std::to_string(i) + ']';
            }
            varMap[name] = var;
            if (pVarMapBySemantic)
            {
                (*pVarMapBySemantic)[var.semanticName] = var;
            }
        }
    }

    static void reflectVaryingParameter(const ReflectionPath& path, const std::string& name, SlangParameterCategory category, ProgramReflection::VariableMap& varMap, ProgramReflection::VariableMap* pVarMapBySemantic = nullptr)
    {
        TypeLayoutReflection* pTypeLayout = path.pVar->getTypeLayout();
        // Skip parameters that don't consume space in the given category
        if (pTypeLayout->getSize(category) == 0) return;

        TypeReflection::Kind kind = pTypeLayout->getKind();
        // If this is a leaf node, store it
        if ((kind == TypeReflection::Kind::Matrix) || (kind == TypeReflection::Kind::Vector) || (kind == TypeReflection::Kind::Scalar))
        {
            storeShaderVariable(path, category, name, varMap, pVarMapBySemantic, 0, 0);
        }
        else if (kind == TypeReflection::Kind::Array)
        {
            auto arrayKind = pTypeLayout->getElementTypeLayout()->getKind();
            assert((arrayKind == TypeReflection::Kind::Matrix) || (arrayKind == TypeReflection::Kind::Vector) || (arrayKind == TypeReflection::Kind::Scalar));
            uint32_t arraySize = (uint32_t)pTypeLayout->getTotalArrayElementCount();
            uint32_t arrayStride = (uint32_t)pTypeLayout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
            storeShaderVariable(path, category, name, varMap, pVarMapBySemantic, arraySize, arrayStride);
        }
        else if (kind == TypeReflection::Kind::Struct)
        {
            for (uint32_t f = 0; f < pTypeLayout->getFieldCount(); f++)
            {
                ReflectionPath newPath;
                newPath.pVar = pTypeLayout->getFieldByIndex(f);
                newPath.pParent = &path;
                newPath.childIndex = f;
                std::string memberName = name + '.' + newPath.pVar->getName();
                reflectVaryingParameter(newPath, memberName, category, varMap, pVarMapBySemantic);
            }
        }
        else
        {
            should_not_get_here();
        }
    }

    static void reflectShaderIO(EntryPointReflection* pEntryPoint, SlangParameterCategory category, ProgramReflection::VariableMap& varMap, ProgramReflection::VariableMap* pVarMapBySemantic = nullptr)
    {
        uint32_t entryPointParamCount = pEntryPoint->getParameterCount();
        for (uint32_t pp = 0; pp < entryPointParamCount; ++pp)
        {
            ReflectionPath path;
            path.pVar = pEntryPoint->getParameterByIndex(pp);
            reflectVaryingParameter(path, path.pVar->getName(), category, varMap, pVarMapBySemantic);
        }
    }

    ProgramReflection::SharedPtr ProgramReflection::create(slang::ShaderReflection* pSlangReflector, std::string& log)
    {
        return SharedPtr(new ProgramReflection(pSlangReflector, log));
    }

    ProgramReflection::BindType getBindTypeFromSetType(DescriptorSet::Type type)
    {
        switch (type)
        {
        case DescriptorSet::Type::Cbv:
            return ProgramReflection::BindType::Cbv;
        case DescriptorSet::Type::Sampler:
            return ProgramReflection::BindType::Sampler;
        case DescriptorSet::Type::StructuredBufferSrv:
        case DescriptorSet::Type::TextureSrv:
        case DescriptorSet::Type::TypedBufferSrv:
            return ProgramReflection::BindType::Srv;
        case DescriptorSet::Type::StructuredBufferUav:
        case DescriptorSet::Type::TextureUav:
        case DescriptorSet::Type::TypedBufferUav:
            return ProgramReflection::BindType::Uav;
        default:
            should_not_get_here();
            return ProgramReflection::BindType(-1);
        }
    }

    ProgramReflection::ProgramReflection(slang::ShaderReflection* pSlangReflector, std::string& log)
    {
        ParameterBlockReflection::SharedPtr pDefaultBlock = ParameterBlockReflection::create("");
        for (uint32_t i = 0; i < pSlangReflector->getParameterCount(); i++)
        {
            VariableLayoutReflection* pSlangLayout = pSlangReflector->getParameterByIndex(i);
            ReflectionVar::SharedPtr pVar = reflectTopLevelVariable(pSlangLayout);

            // In GLSL, the varying (in/out) variables are reflected as globals. Ignore them, we will reflect them later
            if (pVar->getType()->unwrapArray()->asResourceType() == nullptr) continue;

            if (pSlangLayout->getType()->unwrapArray()->getKind() == TypeReflection::Kind::ParameterBlock)
            {
                std::string name = std::string(pSlangLayout->getName());
                ParameterBlockReflection::SharedPtr pBlock = ParameterBlockReflection::create(name);
                pBlock->addResource(pVar);
                pBlock->finalize();
                addParameterBlock(pBlock);
            }
            else
            {
                pDefaultBlock->addResource(pVar);
            }
        }

        pDefaultBlock->finalize();
        addParameterBlock(pDefaultBlock);

        if (pDefaultBlock->isEmpty() == false)
        {            
            // Initialize the map from the default-block resources to the global resources
            for (const auto& res : mpDefaultBlock->getResourceVec())
            {
                const auto& loc = mpDefaultBlock->getResourceBinding(res.name);
                ResourceBinding bind;
                bind.regIndex = res.regIndex;
                bind.regSpace = res.regSpace;
                bind.type = getBindTypeFromSetType(res.setType);
                mResourceBindMap[bind] = loc;
            }
        }

        // Reflect per-stage parameters
        SlangUInt entryPointCount = pSlangReflector->getEntryPointCount();
        for (SlangUInt ee = 0; ee < entryPointCount; ++ee)
        {
            EntryPointReflection* pEntryPoint = pSlangReflector->getEntryPointByIndex(ee);

            switch (pEntryPoint->getStage())
            {
            case SLANG_STAGE_COMPUTE:
            {
                SlangUInt sizeAlongAxis[3];
                pEntryPoint->getComputeThreadGroupSize(3, &sizeAlongAxis[0]);
                mThreadGroupSize.x = (uint32_t)sizeAlongAxis[0];
                mThreadGroupSize.y = (uint32_t)sizeAlongAxis[1];
                mThreadGroupSize.z = (uint32_t)sizeAlongAxis[2];
            }
            break;
            case SLANG_STAGE_FRAGMENT:
                reflectShaderIO(pEntryPoint, SLANG_PARAMETER_CATEGORY_FRAGMENT_OUTPUT, mPsOut);
                break;
            case SLANG_STAGE_VERTEX:
                reflectShaderIO(pEntryPoint, SLANG_PARAMETER_CATEGORY_VERTEX_INPUT, mVertAttr, &mVertAttrBySemantic);
                break;
#ifdef FALCOR_VK
                mIsSampleFrequency = pEntryPoint->usesAnySampleRateInput();
#else
                mIsSampleFrequency = true; // #SLANG Slang reports false for DX shaders. There's an open issue, once it's fixed we should remove that
#endif
            default:
                break;
            }
        }
    }

    void ProgramReflection::addParameterBlock(const ParameterBlockReflection::SharedConstPtr& pBlock)
    {
        assert(mParameterBlocksIndices.find(pBlock->getName()) == mParameterBlocksIndices.end());
        mParameterBlocksIndices[pBlock->getName()] = mpParameterBlocks.size();
        mpParameterBlocks.push_back(pBlock);
        if (pBlock->getName().size() == 0) mpDefaultBlock = pBlock;
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

    ReflectionVar::SharedPtr ReflectionVar::create(const std::string& name, const ReflectionType::SharedConstPtr& pType, size_t offset, uint32_t descOffset, uint32_t regSpace)
    {
        return SharedPtr(new ReflectionVar(name, pType, offset, descOffset, regSpace));
    }

    ReflectionVar::ReflectionVar(const std::string& name, const ReflectionType::SharedConstPtr& pType, size_t offset, uint32_t descOffset, uint32_t regSpace) : mName(name), mpType(pType), mOffset(offset), mRegSpace(regSpace), mDescOffset(descOffset)
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

    static ParameterBlockReflection::ResourceDesc getResourceDesc(const ReflectionVar::SharedConstPtr& pVar, uint32_t descCount, const std::string& name)
    {
        ParameterBlockReflection::ResourceDesc d;
        d.descCount = descCount;
        d.descOffset = 0;
        d.regIndex = pVar->getRegisterIndex();
        d.regSpace = pVar->getRegisterSpace();

        d.pType = pVar->getType()->unwrapArray()->asResourceType()->inherit_shared_from_this::shared_from_this();
        assert(d.pType);
        auto shaderAccess = d.pType->getShaderAccess();
        switch (d.pType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
            d.setType = ParameterBlockReflection::ResourceDesc::Type::Cbv;
            break;
        case ReflectionResourceType::Type::RawBuffer:
        case ReflectionResourceType::Type::Texture:
            d.setType = shaderAccess == ReflectionResourceType::ShaderAccess::Read ? ParameterBlockReflection::ResourceDesc::Type::TextureSrv : ParameterBlockReflection::ResourceDesc::Type::TextureUav;
            break;
        case ReflectionResourceType::Type::Sampler:
            d.setType = ParameterBlockReflection::ResourceDesc::Type::Sampler;
            break;
        case ReflectionResourceType::Type::StructuredBuffer:
            d.setType = shaderAccess == ReflectionResourceType::ShaderAccess::Read ? ParameterBlockReflection::ResourceDesc::Type::StructuredBufferSrv : ParameterBlockReflection::ResourceDesc::Type::StructuredBufferUav;
            break;
        case ReflectionResourceType::Type::TypedBuffer:
            d.setType = shaderAccess == ReflectionResourceType::ShaderAccess::Read ? ParameterBlockReflection::ResourceDesc::Type::TypedBufferSrv : ParameterBlockReflection::ResourceDesc::Type::TypedBufferUav;
            break;
        default:
            should_not_get_here();
            d.setType = ParameterBlockReflection::ResourceDesc::Type::Count;
        }
        d.name = name;
        return d;
    }

    static void flattenResources(const ReflectionVar::SharedConstPtr& pVar, ParameterBlockReflection::ResourceVec& resources, uint32_t arrayElements, std::string name);
        
    static void flattenResources(const ReflectionArrayType* pArrayType, ParameterBlockReflection::ResourceVec& resources, uint32_t arrayElements, std::string name)
    {
        assert(pArrayType->asResourceType() == nullptr);
        if (pArrayType->getType()->asArrayType())
        {
            pArrayType = pArrayType->getType()->asArrayType();
            for (uint32_t i = 0; i < pArrayType->getArraySize(); i++)
            {
                flattenResources(pArrayType, resources, 1, name + '[' + std::to_string(i) + ']');
            }
        }

        const ReflectionStructType* pStructType = pArrayType->getType()->asStructType();
        if (pStructType)
        {
            for (const auto& pMember : *pStructType)
            {
                flattenResources(pMember, resources, 1, name + '.' + pMember->getName());
            }
        }
    }

    static void flattenResources(const ReflectionVar::SharedConstPtr& pVar, ParameterBlockReflection::ResourceVec& resources, uint32_t arrayElements, std::string name)
    {
        const ReflectionType* pType = pVar->getType()->unwrapArray();
        uint32_t elementCount = max(1u, pVar->getType()->getTotalArraySize()) * arrayElements;
        if (pType->asResourceType())
        {
            resources.push_back(getResourceDesc(pVar, elementCount, name));
            return;
        }

        const ReflectionArrayType* pArrayType = pVar->getType()->asArrayType();
        if (pArrayType)
        {
            for (uint32_t i = 0; i < pArrayType->getArraySize(); i++)
            {
                flattenResources(pArrayType, resources, 1, name + '[' + std::to_string(i) + ']');
            }
        }

        const ReflectionStructType* pStructType = pType->asStructType();
        if (pStructType)
        {
            for (const auto& pMember : *pStructType)
            {
                flattenResources(pMember, resources, elementCount, name + '.' + pMember->getName());
            }
        }
    }

    static void flattenResources(const ReflectionStructType* pStructType, ParameterBlockReflection::ResourceVec& resources)
    {
        for (const auto& pMember : *pStructType)
        {
            flattenResources(pMember, resources, 1, pMember->getName());
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
        uint32_t elementCount = max(1u, pVar->getType()->getTotalArraySize());
        mResources.push_back(getResourceDesc(pVar, elementCount, pVar->getName()));
        mpResourceVars->addMember(pVar);

        // If this is a constant-buffer, it might contain resources. Extract them.
        const ReflectionType* pType = pResourceType->getStructType().get();
        const ReflectionStructType* pStruct = (pType != nullptr) ? pType->asStructType() : nullptr;
        if (pStruct)
        {
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

    void ParameterBlockReflection::finalize()
    {
        struct SetIndex
        {
            SetIndex(const ResourceDesc& desc) : regSpace(desc.regSpace)
#ifdef FALCOR_D3D12
                ,isSampler(desc.setType == ResourceDesc::Type::Sampler)
#endif
            {}
            bool isSampler = false;
            uint32_t regSpace;
            bool operator<(const SetIndex& other) const { return (regSpace == other.regSpace) ? isSampler < other.isSampler : regSpace < other.regSpace; }
        };

        std::map<SetIndex, uint32_t> newSetIndices;
        std::vector<std::vector<DescriptorSet::Layout::Range>> sets;

        // Generate the descriptor sets layouts
        for (const auto& res : mResources)
        {
            if (hasSuffix(res.name, ".layers"))
            {
                std::string a = res.name;
            }
            SetIndex origIndex(res);
            uint32_t setIndex;
            if (newSetIndices.find(origIndex) == newSetIndices.end())
            {
                // New set
                setIndex = (uint32_t)sets.size();
                newSetIndices[origIndex] = setIndex;
                sets.push_back({});
            }
            else
            {
                setIndex = newSetIndices[origIndex];
            }

            // Add the current resource range. We need to check that the range doesn't already exist. If it is, we merge the ranges
            auto& ranges = sets[setIndex];
            bool found = false;
            for (uint32_t r = 0; r < ranges.size(); r++)
            {
                auto& range = ranges[r];
                if (range.baseRegIndex == res.regIndex && (res.setType == range.type))
                {
                    range.descCount = max(range.descCount, res.descCount + res.descOffset);
                    mResourceBindings[res.name] = BindLocation(setIndex, r);
                    found = true;
                    break;
                }
            }

            if (found == false)
            {
                mResourceBindings[res.name] = BindLocation(setIndex, (uint32_t)ranges.size());
                DescriptorSet::Layout::Range range;
                range.baseRegIndex = res.regIndex;
                range.descCount = res.descCount + res.descOffset;
                range.regSpace = res.regSpace;
                range.type = res.setType;
                ranges.push_back(range);
            }
        }

        mSetLayouts.resize(sets.size());
        for (uint32_t s = 0; s < sets.size(); s++)
        {
            const auto& src = sets[s];
            auto& dst = mSetLayouts[s];
            for (uint32_t r = 0; r < src.size(); r++)
            {
                const auto& range = src[r];
                dst.addRange(range.type, range.baseRegIndex, range.descCount, range.regSpace);
            }
        }
    }

    uint32_t ProgramReflection::getParameterBlockIndex(const std::string& name) const
    {
        const auto& it = mParameterBlocksIndices.find(name);
        return (it == mParameterBlocksIndices.end()) ? kInvalidLocation : (uint32_t)it->second;
    }

    const ParameterBlockReflection::SharedConstPtr& ProgramReflection::getParameterBlock(const std::string& name) const
    {
        uint32_t index = getParameterBlockIndex(name);
        if (index == kInvalidLocation)
        {
            static ParameterBlockReflection::SharedConstPtr pNull = nullptr;
            logWarning("Can't find a parameter block named " + name);
            return pNull;
        }
        return mpParameterBlocks[index];
    }

    const ParameterBlockReflection::SharedConstPtr& ProgramReflection::getParameterBlock(uint32_t index) const
    {
        if (index >= mpParameterBlocks.size())
        {
            logWarning("Can't find a parameter block at index " + std::to_string(index));
            static ParameterBlockReflection::SharedConstPtr pNull = nullptr;
            return pNull;
        }
        return mpParameterBlocks[index];
    }

    ReflectionVar::SharedConstPtr ReflectionType::findMember(const std::string& name) const
    {
        return findMemberInternal(name, 0, 0, 0, 0, 0);
    }

    ReflectionVar::SharedConstPtr ReflectionBasicType::findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace, uint32_t descOffset) const
    {
        // We shouldn't get here
        logWarning("Can't find variable + " + name);
        return nullptr;
    }

    ReflectionVar::SharedConstPtr ReflectionResourceType::findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace, uint32_t descOffset) const
    {
        if (mpStructType)
        {
            return mpStructType->findMemberInternal(name, strPos, offset, regIndex, regSpace, descOffset);
        }
        else
        {
            logWarning("Can't find variable '" + name + "'");
            return nullptr;
        }
    }

    ReflectionVar::SharedConstPtr ReflectionArrayType::findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace, uint32_t descOffset) const
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
            descOffset += index;
            ReflectionVar::SharedPtr pVar;
            if (mpType->asResourceType())
            {
                pVar = ReflectionVar::create(name, mpType, regIndex, descOffset, regSpace);
            }
            else
            {
              pVar = ReflectionVar::create(name, mpType, offset);
            }
            return pVar;
        }

        return mpType->findMemberInternal(name, endPos + 1, offset, regIndex, regSpace, descOffset + (index * max(1u, mpType->getTotalArraySize())));
    }

    static ReflectionVar::SharedConstPtr returnOrCreateVar(ReflectionVar::SharedConstPtr pVar, const std::string& name, size_t offset, uint32_t regIndex, uint32_t descOffset)
    {
        if (pVar->getType()->asResourceType())
        {
            if(regIndex || descOffset)
            {
                regIndex += pVar->getRegisterIndex();
                descOffset += pVar->getDescOffset();
                pVar = ReflectionVar::create(name, pVar->getType(), regIndex, descOffset, pVar->getRegisterSpace());
            }
        }
        else if (offset)
        {
            pVar = ReflectionVar::create(name, pVar->getType(), pVar->getOffset() + offset);
        }
        return pVar;
    }

    ReflectionVar::SharedConstPtr ReflectionStructType::findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace, uint32_t descOffset) const
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
        if (newPos == std::string::npos) return returnOrCreateVar(pVar, name, offset, regIndex, descOffset);
        const auto& pNewType = pVar->getType().get();
        regIndex = pVar->getType()->unwrapArray()->asResourceType() ? pVar->getRegisterIndex() : 0;
        regSpace = pVar->getType()->unwrapArray()->asResourceType() ? pVar->getRegisterSpace() : 0;
        descOffset += pVar->getDescOffset();
        offset += pVar->getOffset();
        return pNewType->findMemberInternal(name, newPos + 1, offset, regIndex, regSpace, descOffset);
    }

    size_t ReflectionStructType::getMemberIndex(const std::string& name) const
    {
        auto it = mNameToIndex.find(name);
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

    static void extractOffsets(const ReflectionType* pType, size_t offset, uint32_t arraySize, ReflectionResourceType::OffsetDescMap& offsetMap)
    {
        if (pType->asResourceType()) return;
        const ReflectionBasicType* pBasicType = pType->asBasicType();
        if (pBasicType)
        {
            ReflectionResourceType::OffsetDesc desc;
            desc.type = pBasicType->getType();
            desc.count = arraySize;
            offsetMap[offset] = desc;
            return;
        }

        const ReflectionArrayType* pArrayType = pType->asArrayType();
        if (pArrayType)
        {
            pType = pArrayType->getType().get();
            uint32_t arraySize = pArrayType->getArraySize();
            uint32_t count = arraySize;
            for (uint32_t i = 0; i < arraySize; i++)
            {
                extractOffsets(pType, offset + i * pArrayType->getArrayStride(), count, offsetMap);
                --count;
            }
            return;
        }

        // If we got here this has to be a struct
        const ReflectionStructType* pStructType = pType->asStructType();
        assert(pStructType);
        for (const auto &pVar : *pStructType)
        {
            size_t memberOffset = pVar->getOffset() + offset;
            extractOffsets(pVar->getType().get(), memberOffset, 0, offsetMap);
        }
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

    void ReflectionResourceType::setStructType(const ReflectionType::SharedConstPtr& pType)
    {
        mpStructType = pType;
        extractOffsets(pType.get(), 0, 0, mOffsetDescMap);
    }

    ReflectionBasicType::SharedPtr ReflectionBasicType::create(size_t offset, Type type, bool isRowMajor, size_t size)
    {
        return SharedPtr(new ReflectionBasicType(offset, type, isRowMajor, size));
    }

    ReflectionBasicType::ReflectionBasicType(size_t offset, Type type, bool isRowMajor, size_t size) :
        ReflectionType(offset), mType(type), mIsRowMajor(isRowMajor), mSize(size) {}

    ReflectionStructType::SharedPtr ReflectionStructType::create(size_t offset, size_t size, const std::string& name)
    {
        return SharedPtr(new ReflectionStructType(offset, size, name));
    }

    ReflectionStructType::ReflectionStructType(size_t offset, size_t size, const std::string& name) :
        ReflectionType(offset), mSize(size), mName(name) {}

    ParameterBlockReflection::BindLocation ParameterBlockReflection::getResourceBinding(const std::string& name) const
    {
        const auto it = mResourceBindings.find(name);
        return (it == mResourceBindings.end()) ? BindLocation() : it->second;
    }

    const ReflectionVar::SharedConstPtr ProgramReflection::getResource(const std::string& name) const
    {
        return mpDefaultBlock->getResource(name);
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

    const ProgramReflection::ShaderVariable* getShaderAttribute(const std::string& name, const ProgramReflection::VariableMap& varMap, const std::string& funcName)
    {
        const auto& it = varMap.find(name);
        return (it == varMap.end()) ? nullptr : &(it->second);
    }

    const ProgramReflection::ShaderVariable* ProgramReflection::getVertexAttributeBySemantic(const std::string& semantic) const
    {
        return getShaderAttribute(semantic, mVertAttrBySemantic, "getVertexAttributeBySemantic()");
    }

    const ProgramReflection::ShaderVariable* ProgramReflection::getVertexAttribute(const std::string& name) const
    {
        return getShaderAttribute(name, mVertAttr, "getVertexAttribute()");
    }

    const ProgramReflection::ShaderVariable* ProgramReflection::getPixelShaderOutput(const std::string& name) const
    {
        return getShaderAttribute(name, mPsOut, "getPixelShaderOutput()");
    }

    const ReflectionResourceType::OffsetDesc& ReflectionResourceType::getOffsetDesc(size_t offset) const
    {
        static const ReflectionResourceType::OffsetDesc empty;
        const auto& offsetIt = mOffsetDescMap.find(offset);
        return (offsetIt == mOffsetDescMap.end()) ? empty : offsetIt->second;
    }
}
