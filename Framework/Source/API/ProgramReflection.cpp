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
    ProgramReflection::SharedPtr ProgramReflection::create(ShaderReflection* pSlangReflector, std::string& log)
    {
        SharedPtr pReflection = SharedPtr(new ProgramReflection);
        return pReflection->init(pSlangReflector, log) ? pReflection : nullptr;
    }

    ProgramReflection::BindLocation ProgramReflection::getBufferBinding(const std::string& name) const
    {
        // Names are unique regardless of buffer type. Search in each map
        for (const auto& desc : mBuffers)
        {
            const auto& it = desc.nameMap.find(name);
            if (it != desc.nameMap.end())
            {
                return it->second;
            }
        }

        static const BindLocation invalidBind;
        return invalidBind;
    }

    bool ProgramReflection::init(ShaderReflection* pSlangReflector, std::string& log)
    {
        bool b = true;
        b = b && reflectResources(          pSlangReflector, log);
        b = b && reflectVertexAttributes(   pSlangReflector, log);
        b = b && reflectPixelShaderOutputs( pSlangReflector, log);
        return b;
    }

    const ProgramReflection::Variable* ProgramReflection::BufferReflection::getVariableData(const std::string& name, size_t& offset) const
    {
        const std::string msg = "Error when getting variable data \"" + name + "\" from buffer \"" + mName + "\".\n";
        uint32_t arrayIndex = 0;
        offset = kInvalidLocation;

        // Look for the variable
        auto&& var = mVariables.find(name);

#ifdef FALCOR_DX11
        if (var == mVariables.end())
        {
            // Textures might come from our struct. Try again.
            std::string texName = name + ".t";
            var = mVariables.find(texName);
        }
#endif
        if (var == mVariables.end())
        {
            // The name might contain an array index. Remove the last array index and search again
            std::string nameV2;
            uint32_t arrayIndex;
            if(parseArrayIndex(name, nameV2, arrayIndex))
            {
                var = mVariables.find(nameV2);
                if (var == mVariables.end())
                {
                    logWarning(msg + "Variable " + name + "not found");
                    return nullptr;
                }

                const auto& data = var->second;
                if (data.arraySize == 0)
                {
                    // Not an array, so can't have an array index
                    logError(msg + "Variable is not an array, so name can't include an array index.");
                    return nullptr;
                }

                if (arrayIndex >= data.arraySize)
                {
                    logError(msg + "Array index (" + std::to_string(arrayIndex) + ") out-of-range. Array size == " + std::to_string(data.arraySize) + ".");
                    return nullptr;
                }
            }
        }

        const auto* pData = &var->second;
        offset = pData->location + pData->arrayStride * arrayIndex;
        return pData;
    }

    const ProgramReflection::Variable* ProgramReflection::BufferReflection::getVariableData(const std::string& name) const
    {
        size_t t;
        return getVariableData(name, t);
    }

    ProgramReflection::BufferReflection::SharedConstPtr ProgramReflection::getBufferDesc(uint32_t regSpace, uint32_t regIndex, ShaderAccess shaderAccess, BufferReflection::Type bufferType) const
    {
        const auto& descMap = mBuffers[uint32_t(bufferType)].descMap;
        const auto& desc = descMap.find({ regSpace, regIndex, shaderAccess });
        if (desc == descMap.end())
        {
            return nullptr;
        }
        return desc->second;
    }

    ProgramReflection::BufferReflection::SharedConstPtr ProgramReflection::getBufferDesc(const std::string& name, BufferReflection::Type bufferType) const
    {
        BindLocation bindLoc = getBufferBinding(name);
        if (bindLoc.baseRegIndex != kInvalidLocation)
        {
            return getBufferDesc(bindLoc.regSpace, bindLoc.baseRegIndex, bindLoc.shaderAccess, bufferType);
        }
        return nullptr;
    }

    const ProgramReflection::Resource* ProgramReflection::BufferReflection::getResourceData(const std::string& name) const
    {
        const auto& it = mResources.find(name);
        return it == mResources.end() ? nullptr : &(it->second);
    }

    ProgramReflection::BufferReflection::BufferReflection(const std::string& name, uint32_t regSpace, uint32_t baseRegIndex, uint32_t arraySize, Type type, StructuredType structuredType, size_t size, const VariableMap& varMap, const ResourceMap& resourceMap, ShaderAccess shaderAccess) :
        mName(name),
        mType(type),
        mStructuredType(structuredType),
        mSizeInBytes(size),
        mVariables(varMap),
        mResources(resourceMap),
        mRegIndex(baseRegIndex),
        mRegSpace(regSpace),
        mArraySize(arraySize),
        mShaderAccess(shaderAccess)
    {
    }

    ProgramReflection::BufferReflection::SharedPtr ProgramReflection::BufferReflection::create(const std::string& name, uint32_t regSpace, uint32_t baseRegIndex, uint32_t arraySize, Type type, StructuredType structuredType, size_t size, const VariableMap& varMap, const ResourceMap& resourceMap, ShaderAccess shaderAccess)
    {
        return SharedPtr(new BufferReflection(name, regSpace, baseRegIndex, arraySize, type, structuredType, size, varMap, resourceMap, shaderAccess));
    }

    const ProgramReflection::Variable* ProgramReflection::getVertexAttribute(const std::string& name) const
    {
        const auto& it = mVertAttr.find(name);
        return (it == mVertAttr.end()) ? nullptr : &(it->second);
    }

    const ProgramReflection::Variable* ProgramReflection::getVertexAttributeBySemantic(const std::string& semantic) const
    {
        const auto& it = mVertAttrBySemantic.find(semantic);
        return (it == mVertAttrBySemantic.end()) ? nullptr : &(it->second);
    }

    const ProgramReflection::Variable* ProgramReflection::getFragmentOutput(const std::string& name) const
    {
        const auto& it = mFragOut.find(name);
        return (it == mFragOut.end()) ? nullptr : &(it->second);
    }

    const ProgramReflection::Resource* ProgramReflection::getResourceDesc(const std::string& name) const
    {
        const auto& it = mResources.find(name);
        const ProgramReflection::Resource* pRes = (it == mResources.end()) ? nullptr : &(it->second);

        if (pRes == nullptr)
        {
            logWarning("Can't find resource '" + name + "' in program");
        }
        return pRes;
    }

    /************************************************************************/
    /*  SPIRE Reflection                                                    */
    /************************************************************************/
    ProgramReflection::Variable::Type getVariableType(TypeReflection::ScalarType slangScalarType, uint32_t rows, uint32_t columns)
    {
        switch (slangScalarType)
        {
        case TypeReflection::ScalarType::None:
            // This isn't a scalar/matrix/vector, so it can't
            // be encoded in the `enum` that Falcor provides.
            return ProgramReflection::Variable::Type::Unknown;

        case TypeReflection::ScalarType::Bool:
            assert(rows == 1);
            switch (columns)
            {
            case 1:
                return ProgramReflection::Variable::Type::Bool;
            case 2:
                return ProgramReflection::Variable::Type::Bool2;
            case 3:
                return ProgramReflection::Variable::Type::Bool3;
            case 4:
                return ProgramReflection::Variable::Type::Bool4;
            }
        case TypeReflection::ScalarType::UInt32:
            assert(rows == 1);
            switch (columns)
            {
            case 1:
                return ProgramReflection::Variable::Type::Uint;
            case 2:
                return ProgramReflection::Variable::Type::Uint2;
            case 3:
                return ProgramReflection::Variable::Type::Uint3;
            case 4:
                return ProgramReflection::Variable::Type::Uint4;
            }
        case TypeReflection::ScalarType::Int32:
            assert(rows == 1);
            switch (columns)
            {
            case 1:
                return ProgramReflection::Variable::Type::Int;
            case 2:
                return ProgramReflection::Variable::Type::Int2;
            case 3:
                return ProgramReflection::Variable::Type::Int3;
            case 4:
                return ProgramReflection::Variable::Type::Int4;
            }
        case TypeReflection::ScalarType::Float32:
            switch (rows)
            {
            case 1:
                switch (columns)
                {
                case 1:
                    return ProgramReflection::Variable::Type::Float;
                case 2:
                    return ProgramReflection::Variable::Type::Float2;
                case 3:
                    return ProgramReflection::Variable::Type::Float3;
                case 4:
                    return ProgramReflection::Variable::Type::Float4;
                }
                break;
            case 2:
                switch (columns)
                {
                case 2:
                    return ProgramReflection::Variable::Type::Float2x2;
                case 3:
                    return ProgramReflection::Variable::Type::Float2x3;
                case 4:
                    return ProgramReflection::Variable::Type::Float2x4;
                }
                break;
            case 3:
                switch (columns)
                {
                case 2:
                    return ProgramReflection::Variable::Type::Float3x2;
                case 3:
                    return ProgramReflection::Variable::Type::Float3x3;
                case 4:
                    return ProgramReflection::Variable::Type::Float3x4;
                }
                break;
            case 4:
                switch (columns)
                {
                case 2:
                    return ProgramReflection::Variable::Type::Float4x2;
                case 3:
                    return ProgramReflection::Variable::Type::Float4x3;
                case 4:
                    return ProgramReflection::Variable::Type::Float4x4;
                }
                break;
            }
        }

        should_not_get_here();
        return ProgramReflection::Variable::Type::Unknown;
    }

    size_t getRowCountFromType(ProgramReflection::Variable::Type type)
    {
        switch (type)
        {
        case ProgramReflection::Variable::Type::Unknown:
            return 0;
        case ProgramReflection::Variable::Type::Bool:
        case ProgramReflection::Variable::Type::Bool2:
        case ProgramReflection::Variable::Type::Bool3:
        case ProgramReflection::Variable::Type::Bool4:
        case ProgramReflection::Variable::Type::Uint:
        case ProgramReflection::Variable::Type::Uint2:
        case ProgramReflection::Variable::Type::Uint3:
        case ProgramReflection::Variable::Type::Uint4:
        case ProgramReflection::Variable::Type::Int:
        case ProgramReflection::Variable::Type::Int2:
        case ProgramReflection::Variable::Type::Int3:
        case ProgramReflection::Variable::Type::Int4:
        case ProgramReflection::Variable::Type::Float:
        case ProgramReflection::Variable::Type::Float2:
        case ProgramReflection::Variable::Type::Float3:
        case ProgramReflection::Variable::Type::Float4:
            return 1;
        case ProgramReflection::Variable::Type::Float2x2:
        case ProgramReflection::Variable::Type::Float2x3:
        case ProgramReflection::Variable::Type::Float2x4:
            return 2;
        case ProgramReflection::Variable::Type::Float3x2:
        case ProgramReflection::Variable::Type::Float3x3:
        case ProgramReflection::Variable::Type::Float3x4:
            return 3;
        case ProgramReflection::Variable::Type::Float4x2:
        case ProgramReflection::Variable::Type::Float4x3:
        case ProgramReflection::Variable::Type::Float4x4:
            return 4;
        default:
            should_not_get_here();
            return 0;
        }
    }

    size_t getBytesPerVarType(ProgramReflection::Variable::Type type)
    {
        switch (type)
        {
        case ProgramReflection::Variable::Type::Unknown:
            return 0;
        case ProgramReflection::Variable::Type::Bool:
        case ProgramReflection::Variable::Type::Uint:
        case ProgramReflection::Variable::Type::Int:
        case ProgramReflection::Variable::Type::Float:
            return 4;
        case ProgramReflection::Variable::Type::Bool2:
        case ProgramReflection::Variable::Type::Uint2:
        case ProgramReflection::Variable::Type::Int2:
        case ProgramReflection::Variable::Type::Float2:
            return 8;
        case ProgramReflection::Variable::Type::Bool3:
        case ProgramReflection::Variable::Type::Uint3:
        case ProgramReflection::Variable::Type::Int3:
        case ProgramReflection::Variable::Type::Float3:
            return 12;
        case ProgramReflection::Variable::Type::Bool4:
        case ProgramReflection::Variable::Type::Uint4:
        case ProgramReflection::Variable::Type::Int4:
        case ProgramReflection::Variable::Type::Float4:
        case ProgramReflection::Variable::Type::Float2x2:
            return 16;
        case ProgramReflection::Variable::Type::Float2x3:
        case ProgramReflection::Variable::Type::Float3x2:
            return 24;
        case ProgramReflection::Variable::Type::Float2x4:
        case ProgramReflection::Variable::Type::Float4x2:
            return 32;
        case ProgramReflection::Variable::Type::Float3x3:
            return 36;
        case ProgramReflection::Variable::Type::Float3x4:
        case ProgramReflection::Variable::Type::Float4x3:
            return 48;
        case ProgramReflection::Variable::Type::Float4x4:
            return 64;
        default:
            should_not_get_here();
            return 0;
        }

    }

    // Information we need to track when converting Slang reflection
    // information over to the Falcor equivalent
    struct ReflectionGenerationContext
    {
        SlangStage stage = SLANG_STAGE_NONE;
        ProgramReflection*  pReflector = nullptr;
        ProgramReflection::VariableMap* pVariables = nullptr;
        ProgramReflection::ResourceMap* pResourceMap = nullptr;
        std::string* pLog = nullptr;

        ProgramReflection::VariableMap& getVariableMap() { return *pVariables; }
        ProgramReflection::ResourceMap& getResourceMap() { return *pResourceMap; }
        std::string& getLog() { return *pLog; }
    };

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
        ReflectionPath*                     parent = nullptr;
        VariableLayoutReflection*    var = nullptr;
        TypeLayoutReflection*        typeLayout = nullptr;
        uint32_t                            childIndex = 0;
    };

    // Once we've found the path from the root down to a particular leaf
    // variable, `getBindingIndex` can be used to find the final summed-up
    // index (or byte offset, in the case of a uniform).
    uint32_t getBindingIndex(ReflectionPath* path, SlangParameterCategory category)
    {
        uint32_t offset = 0;
        for (auto pp = path; pp; pp = pp->parent)
        {
            if (pp->var)
            {
                offset += (uint32_t)pp->var->getOffset(category);
                continue;
            }
            else if (pp->typeLayout)
            {
                switch (pp->typeLayout->getKind())
                {
                case TypeReflection::Kind::Array:
                    offset += (uint32_t)pp->typeLayout->getElementStride(category) * pp->childIndex;
                    continue;

                case TypeReflection::Kind::Struct:
                    offset += (uint32_t)pp->typeLayout->getFieldByIndex(int(pp->childIndex))->getOffset(category);
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

	// Once we've found the path from the root down to a particular leaf
	// variable, `getDescOffset` can be used to find the final summed-up descriptor offset of the element
	uint32_t getDescOffset(ReflectionPath* path, uint32_t arraySize, SlangParameterCategory category)
	{
#ifndef FALCOR_VK
		return 0;
#else
		uint32_t offset = 0;
		bool first = true;
		for (auto pp = path; pp; pp = pp->parent)
		{
			if ((pp->typeLayout) && (pp->typeLayout->getKind() == TypeReflection::Kind::Array))
			{
				offset += (uint32_t)pp->childIndex * arraySize;
				arraySize *= (uint32_t)pp->typeLayout->getElementCount();
			}
		}
		return offset;
#endif
	}

    size_t getUniformOffset(ReflectionPath* path)
    {
        return getBindingIndex(path, SLANG_PARAMETER_CATEGORY_UNIFORM);
    }

    uint32_t getBindingSpace(ReflectionPath* path, SlangParameterCategory category)
    {
        return (uint32_t)path->var->getBindingSpace(category);
    }

    static ProgramReflection::Resource::ResourceType getResourceType(TypeReflection* pSlangType)
    {
        switch (pSlangType->unwrapArray()->getKind())
        {
        case TypeReflection::Kind::SamplerState:
            return ProgramReflection::Resource::ResourceType::Sampler;
        case TypeReflection::Kind::ShaderStorageBuffer:
            return ProgramReflection::Resource::ResourceType::StructuredBuffer;
        case TypeReflection::Kind::Resource:
            switch (pSlangType->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK)
            {
            case SLANG_STRUCTURED_BUFFER:
                return ProgramReflection::Resource::ResourceType::StructuredBuffer;

            case SLANG_BYTE_ADDRESS_BUFFER:
                return ProgramReflection::Resource::ResourceType::RawBuffer;
            case SLANG_TEXTURE_BUFFER:
                return ProgramReflection::Resource::ResourceType::TypedBuffer;
            default:
                return ProgramReflection::Resource::ResourceType::Texture;
            }
            break;

        default:
            break;
        }
        should_not_get_here();
        return ProgramReflection::Resource::ResourceType::Unknown;
    }

    static ProgramReflection::ShaderAccess getShaderAccess(TypeReflection* pSlangType)
    {
        // Compute access for an array using the underlying type...
        pSlangType = pSlangType->unwrapArray();

        switch (pSlangType->getKind())
        {
        case TypeReflection::Kind::SamplerState:
        case TypeReflection::Kind::ConstantBuffer:
            return ProgramReflection::ShaderAccess::Read;
            break;

        case TypeReflection::Kind::Resource:
        case TypeReflection::Kind::ShaderStorageBuffer:
            switch (pSlangType->getResourceAccess())
            {
            case SLANG_RESOURCE_ACCESS_NONE:
                return ProgramReflection::ShaderAccess::Undefined;

            case SLANG_RESOURCE_ACCESS_READ:
                return ProgramReflection::ShaderAccess::Read;

            default:
                return ProgramReflection::ShaderAccess::ReadWrite;
            }
            break;

        default:
            should_not_get_here();
            return ProgramReflection::ShaderAccess::Undefined;
        }
    }

    static ProgramReflection::Resource::ReturnType getReturnType(TypeReflection* pType)
    {
        // Could be a resource that doesn't have a specific element type (e.g., a raw buffer)
        if (!pType)
            return ProgramReflection::Resource::ReturnType::Unknown;

        switch (pType->getScalarType())
        {
        case TypeReflection::ScalarType::Float32:
            return ProgramReflection::Resource::ReturnType::Float;
        case TypeReflection::ScalarType::Int32:
            return ProgramReflection::Resource::ReturnType::Int;
        case TypeReflection::ScalarType::UInt32:
            return ProgramReflection::Resource::ReturnType::Uint;
        case TypeReflection::ScalarType::Float64:
            return ProgramReflection::Resource::ReturnType::Double;

            // Could be a resource that uses an aggregate element type (e.g., a structured buffer)
        case TypeReflection::ScalarType::None:
            return ProgramReflection::Resource::ReturnType::Unknown;

        default:
            should_not_get_here();
            return ProgramReflection::Resource::ReturnType::Unknown;
        }
    }

    static ProgramReflection::Resource::Dimensions getResourceDimensions(SlangResourceShape shape)
    {
        switch (shape)
        {
        case SLANG_TEXTURE_1D:
            return ProgramReflection::Resource::Dimensions::Texture1D;
        case SLANG_TEXTURE_1D_ARRAY:
            return ProgramReflection::Resource::Dimensions::Texture1DArray;
        case SLANG_TEXTURE_2D:
            return ProgramReflection::Resource::Dimensions::Texture2D;
        case SLANG_TEXTURE_2D_ARRAY:
            return ProgramReflection::Resource::Dimensions::Texture2DArray;
        case SLANG_TEXTURE_2D_MULTISAMPLE:
            return ProgramReflection::Resource::Dimensions::Texture2DMS;
        case SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY:
            return ProgramReflection::Resource::Dimensions::Texture2DMSArray;
        case SLANG_TEXTURE_3D:
            return ProgramReflection::Resource::Dimensions::Texture3D;
        case SLANG_TEXTURE_CUBE:
            return ProgramReflection::Resource::Dimensions::TextureCube;
        case SLANG_TEXTURE_CUBE_ARRAY:
            return ProgramReflection::Resource::Dimensions::TextureCubeArray;

        case SLANG_TEXTURE_BUFFER:
        case SLANG_STRUCTURED_BUFFER:
        case SLANG_BYTE_ADDRESS_BUFFER:
            return ProgramReflection::Resource::Dimensions::Buffer;

        default:
            should_not_get_here();
            return ProgramReflection::Resource::Dimensions::Unknown;
        }
    }

    static bool verifyResourceDefinition(const ProgramReflection::Resource& prev, ProgramReflection::Resource& current, std::string& log)
    {
        bool match = true;
#define error_msg(msg_) std::string(msg_) + " mismatch.\n";
#define test_field(field_)                                           \
            if(prev.field_ != current.field_)                        \
            {                                                        \
                log += error_msg(#field_)                            \
                match = false;                                       \
            }

        test_field(type);
        test_field(dims);
        test_field(retType);
        test_field(regIndex);
        test_field(regSpace);
        test_field(arraySize);
#undef test_field
#undef error_msg

        return match;
    }

    static bool reflectStructuredBuffer(
        ReflectionGenerationContext*    pContext,
        TypeLayoutReflection*    pSlangType,
        const std::string&              name,
        ReflectionPath*                 path);

    static bool reflectConstantBuffer(
        ReflectionGenerationContext*    pContext,
        TypeLayoutReflection*    pSlangType,
        const std::string&              name,
        ReflectionPath*                 path);


    // Generate reflection data for a single variable
    static bool reflectResource(
        ReflectionGenerationContext*    pContext,
        TypeLayoutReflection*    pSlangType,
        const std::string&              name,
        ReflectionPath*                 path)
    {
        auto resourceType = getResourceType(pSlangType->getType());
        if (resourceType == ProgramReflection::Resource::ResourceType::StructuredBuffer)
        {
            // reflect this parameter as a buffer
            return reflectStructuredBuffer(
                pContext,
                pSlangType,
                name,
                path);
        }

        ProgramReflection::Resource falcorDesc;
        falcorDesc.type = resourceType;
        falcorDesc.shaderAccess = getShaderAccess(pSlangType->getType());
        if (resourceType != ProgramReflection::Resource::ResourceType::Sampler)
        {
            falcorDesc.retType = getReturnType(pSlangType->getResourceResultType());
            falcorDesc.dims = getResourceDimensions(pSlangType->getResourceShape());
        }
        bool isArray = pSlangType->isArray();
        falcorDesc.regIndex = (uint32_t)getBindingIndex(path, pSlangType->getParameterCategory());
        falcorDesc.regSpace = (uint32_t)getBindingSpace(path, pSlangType->getParameterCategory());
        falcorDesc.arraySize = isArray ? (uint32_t)pSlangType->getTotalArrayElementCount() : 0;
		falcorDesc.descOffset = (uint32_t)getDescOffset(path, max(1u, falcorDesc.arraySize), pSlangType->getParameterCategory());

        // If this already exists, definitions should match
        auto& resourceMap = *pContext->pResourceMap;
        const auto& prevDef = resourceMap.find(name);
        if (prevDef == resourceMap.end())
        {
            resourceMap[name] = falcorDesc;
        }
        else
        {
            std::string varLog;
            if (verifyResourceDefinition(prevDef->second, falcorDesc, varLog) == false)
            {
                pContext->getLog() += "Shader resource '" + std::string(name) + "' has different definitions between different shader stages. " + varLog;
                return false;
            }
        }

        // For now, we expose all resources as visible to all stages
        resourceMap[name].shaderMask = 0xFFFFFFFF;

        return true;
    }

    // Given the path to a reflection variable, extract its semantic name and index.
    static char const* getSemantic(
        ReflectionPath* pPath,
        uint32_t*       outSemanticIndex)
    {
        // For the most part we just want to walk up the path until we find
        // a concrete variable and then return its name/index.
        //
        // TODO: If the path references an element of a varying array, then
        // we should technically apply a suitable offset to the semantic index,
        // based on the chosen array element. E.g., given:
        //
        //     float4 uvs[4] : TEXCOORD;
        //
        // we should report `uvs` as having semantic `TEXCOORD` with index `0`,
        // while `uvs[2]` should report semantic `TEXCOORD` with index `2`.
        //
        auto pp = pPath;
        while (pp)
        {
            if (auto var = pp->var)
            {
                auto semanticName = var->getSemanticName();
                if (outSemanticIndex)
                    *outSemanticIndex = (uint32_t) var->getSemanticIndex();

                return semanticName;
            }

            pp = pp->parent;
        }

        return nullptr;
    }

    static void maybeReflectVaryingParameter(
        ReflectionGenerationContext*    pContext,
        TypeLayoutReflection*           pSlangType,
        const std::string&              name,
        ReflectionPath*                 pPath,
        slang::ParameterCategory        category,
        ProgramReflection::VariableMap* ioVarMap,
        ProgramReflection::VariableMap* ioVarBySemanticMap = nullptr)
    {
        // Skip parameters that don't consume space in the given category
        if (pSlangType->getSize(category) == 0)
            return;

        ProgramReflection::Variable desc;
        desc.location = getBindingIndex(pPath, category);
        desc.type = getVariableType(
            pSlangType->getScalarType(),
            pSlangType->getRowCount(),
            pSlangType->getColumnCount());

        switch (pSlangType->getKind())
        {
        default:
            break;

        case TypeReflection::Kind::Array:
            desc.arraySize = (uint32_t)pSlangType->getElementCount();
            desc.arrayStride = (uint32_t)pSlangType->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
            break;

        case TypeReflection::Kind::Matrix:
            // TODO(tfoley): Slang needs to report this information!
            //                desc.isRowMajor = (typeDesc.Class == D3D_SVC_MATRIX_ROWS);
            break;
        }

        // Register this varying parameter in the appropriate map
        // for lookup by name.
        if (ioVarMap)
        {
            (*ioVarMap)[name] = desc;
        }

        // Does this variable have a semantic attached?
        uint32_t semanticIndex = 0;
        char const* semanticName = getSemantic(pPath, &semanticIndex);
        if (semanticName && ioVarBySemanticMap)
        {
            if (semanticIndex == 0)
            {
                (*ioVarBySemanticMap)[semanticName] = desc;
            }

            std::string fullSemanticName = std::string(semanticName) + std::to_string(semanticIndex);
            (*ioVarBySemanticMap)[fullSemanticName] = desc;
        }
    }

    static void reflectType(
        ReflectionGenerationContext*    pContext,
        TypeLayoutReflection*    pSlangType,
        const std::string&              name,
        ReflectionPath*                 pPath)
    {
        size_t uniformSize = pSlangType->getSize();

        // For any variable that actually occupies space in
        // uniform data, we want to add an entry to
        // the variables map that can be directly queried:
        if (uniformSize != 0)
        {
            ProgramReflection::Variable desc;

            desc.location = getUniformOffset(pPath);
            desc.type = getVariableType(
                pSlangType->getScalarType(),
                pSlangType->getRowCount(),
                pSlangType->getColumnCount());

            switch (pSlangType->getKind())
            {
            default:
                break;

            case TypeReflection::Kind::Array:
                desc.arraySize = (uint32_t)pSlangType->getElementCount();
                desc.arrayStride = (uint32_t)pSlangType->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
                break;

            case TypeReflection::Kind::Matrix:
                // TODO(tfoley): Slang needs to report this information!
                //                desc.isRowMajor = (typeDesc.Class == D3D_SVC_MATRIX_ROWS);
                break;
            }

            if (!pContext->pVariables)
            {
                logError("unimplemented: global-scope uniforms");
            }

            (*pContext->pVariables)[name] = desc;
        }

        // We might be looking at a varying input/output parameter,
        // so let's try to detect that here. We only want to concern
        // ourselves with varying parameters once they have been
        // broken down to leaf parameters, though.
        //
        // Are we looking at a leaf parameter?
        switch (pSlangType->unwrapArray()->getKind())
        {
        case TypeReflection::Kind::Scalar:
        case TypeReflection::Kind::Vector:
        case TypeReflection::Kind::Matrix:
            // Are we looking at a vertex or fragment entry-point parameter?
            //
            // TODO: This logic will fail to catch GLSL varying parameters,
            // since they aren't declared under a corresponding entry-point.
            // Fixing this would require the Slang API to be able to identify
            // the stage corresponding to any global-scope parameter in GLSL.
            switch (pContext->stage)
            {
            case SLANG_STAGE_VERTEX:
                maybeReflectVaryingParameter(
                    pContext,
                    pSlangType,
                    name,
                    pPath,
                    slang::ParameterCategory::VertexInput,
                    &pContext->pReflector->mVertAttr,
                    &pContext->pReflector->mVertAttrBySemantic);
                break;

            case SLANG_STAGE_FRAGMENT:
                maybeReflectVaryingParameter(
                    pContext,
                    pSlangType,
                    name,
                    pPath,
                    slang::ParameterCategory::FragmentOutput,
                    &pContext->pReflector->mFragOut);
                break;

            default:
                break;
            }
            break;

        default:
            break;
        }


        // We want to reflect resource parameters as soon as we find a
        // type that is an (array of)* (sampler|texture|...)
        //
        // That is, we will look through any number of levels of array-ness
        // to see the type underneath:
        switch (pSlangType->unwrapArray()->getKind())
        {
        case TypeReflection::Kind::Struct:
            // A `struct` type obviously isn't a resource
            break;

            // TODO: If we ever start using arrays of constant buffers,
            // we'd probably want to handle them here too...

        // Explicitly skip constant buffers at this step, because
        // otherwise the test below for resources would catch then
        // when using Vulkan (because constant buffers use the same
        // binding space as texture/sampler parameters).
        case TypeReflection::Kind::ConstantBuffer:
            break;

        default:
            // This might be a resource, or an array of resources.
            // To find out, let's ask what category of resource
            // it consumes.
            switch (pSlangType->getParameterCategory())
            {
            case ParameterCategory::ShaderResource:
            case ParameterCategory::UnorderedAccess:
            case ParameterCategory::SamplerState:
            case ParameterCategory::DescriptorTableSlot:
                // This is a resource, or an array of them (or an array of arrays ...)
                reflectResource(
                    pContext,
                    pSlangType,
                    name,
                    pPath);

                // We don't want to enumerate each individual field
                // as a separate entry in the resources map, so bail out here
                return;

            default:
                break;
            }
            break;
        }

        // If we didn't early exit in the resource case above, then we
        // will go ahead and recurse into the sub-elements of the type
        // (fields of a struct, elements of an array, etc.)
        switch (pSlangType->getKind())
        {
        default:
            // All the interesting cases for non-aggregate values
            // have been handled above.
            break;

        case TypeReflection::Kind::ConstantBuffer:
            // We've found a constant buffer, so reflect it as a top-level buffer
            reflectConstantBuffer(
                pContext,
                pSlangType,
                name,
                pPath);
            break;

        case TypeReflection::Kind::Array:
        {
            // For a variable with array type, we are going to create
            // entries for each element of the array.
            //
            // TODO: This probably isn't a good idea for very large
            // arrays, and obviously doesn't work for arrays that
            // aren't statically sized.
            //
            // TODO: we should probably also handle arrays-of-textures
            // and arrays-of-samplers specially here.

            auto elementCount = (uint32_t)pSlangType->getElementCount();
            TypeLayoutReflection* elementType = pSlangType->getElementTypeLayout();

            assert(name.size());

            for (uint32_t ee = 0; ee < elementCount; ++ee)
            {
                ReflectionPath elementPath;
                elementPath.parent = pPath;
                elementPath.typeLayout = pSlangType;
                elementPath.childIndex = ee;

                reflectType(pContext, elementType, name + '[' + std::to_string(ee) + "]", &elementPath);
            }
        }
        break;

        case TypeReflection::Kind::Struct:
        {
            // For a variable with structure type, we are going
            // to create entries for each field of the structure.
            //
            // TODO: it isn't strictly necessary to do this, but
            // doing something more clever involves additional
            // parsing work during lookup, to deal with `.`
            // operations in the path to a variable.

            uint32_t fieldCount = pSlangType->getFieldCount();
            for (uint32_t ff = 0; ff < fieldCount; ++ff)
            {
                VariableLayoutReflection* field = pSlangType->getFieldByIndex(ff);
                std::string memberName(field->getName());
                std::string fullName = name.size() ? name + '.' + memberName : memberName;

                ReflectionPath fieldPath;
                fieldPath.parent = pPath;
                fieldPath.typeLayout = pSlangType;
                fieldPath.childIndex = ff;
                fieldPath.var = field;

                reflectType(pContext, field->getTypeLayout(), fullName, &fieldPath);
            }
        }
        break;
        }
    }

    static void reflectVariable(
        ReflectionGenerationContext*        pContext,
        VariableLayoutReflection*    pSlangVar,
        ReflectionPath*                     pParentPath)
    {
        // Get the variable name
        std::string name(pSlangVar->getName());

        // Create a path element for the variable
        ReflectionPath varPath;
        varPath.parent = pParentPath;
        varPath.var = pSlangVar;

        // Reflect the Type
        reflectType(pContext, pSlangVar->getTypeLayout(), name, &varPath);
    }

    static void initializeBufferVariables(
        ReflectionGenerationContext*    pContext,
        ReflectionPath*                 pBufferPath,
        TypeLayoutReflection*    pSlangElementType)
    {
        // Element type of a structured buffer need not be a structured type,
        // so don't recurse unless needed...
        if (pSlangElementType->getKind() != TypeReflection::Kind::Struct)
            return;

        uint32_t fieldCount = pSlangElementType->getFieldCount();

        for (uint32_t ff = 0; ff < fieldCount; ff++)
        {
            auto var = pSlangElementType->getFieldByIndex(ff);

            reflectVariable(pContext, var, pBufferPath);
        }
    }

    bool validateBufferDeclaration(const ProgramReflection::BufferReflection* pPrevDesc, const ProgramReflection::VariableMap& varMap, std::string& log)
    {
        bool match = true;
#define error_msg(msg_) std::string(msg_) + " mismatch.\n";
        if (pPrevDesc->getVariableCount() != varMap.size())
        {
            log += error_msg("Variable count");
            match = false;
        }

        for (auto& prevVar = pPrevDesc->varBegin(); prevVar != pPrevDesc->varEnd(); prevVar++)
        {
            const std::string& name = prevVar->first;
            const auto& curVar = varMap.find(name);
            if (curVar == varMap.end())
            {
                log += "Can't find variable '" + name + "' in the new definitions";
            }
            else
            {
#define test_field(field_, msg_)                                      \
            if(prevVar->second.field_ != curVar->second.field_)       \
            {                                                         \
                log += error_msg(prevVar->first + " " + msg_)         \
                match = false;                                       \
            }

                test_field(location, "offset");
                test_field(arraySize, "array size");
                test_field(arrayStride, "array stride");
                test_field(isRowMajor, "row major");
                test_field(type, "Type");
#undef test_field
            }
        }

#undef error_msg

        return match;
    }

    static ProgramReflection::BufferReflection::StructuredType getStructuredBufferType(
        TypeReflection* pSlangType)
    {
        auto invalid = ProgramReflection::BufferReflection::StructuredType::Invalid;

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
            return ProgramReflection::BufferReflection::StructuredType::Default;

        case SLANG_RESOURCE_ACCESS_READ_WRITE:
        case SLANG_RESOURCE_ACCESS_RASTER_ORDERED:
            return ProgramReflection::BufferReflection::StructuredType::Counter;
        case SLANG_RESOURCE_ACCESS_APPEND:
            return ProgramReflection::BufferReflection::StructuredType::Append;
        case SLANG_RESOURCE_ACCESS_CONSUME:
            return ProgramReflection::BufferReflection::StructuredType::Consume;
        }
    }

    static bool reflectBuffer(
        ReflectionGenerationContext*                pContext,
        TypeLayoutReflection*                pSlangType,
        const std::string&                          name,
        ReflectionPath*                             pPath,
        ProgramReflection::BufferData&              bufferDesc,
        ProgramReflection::BufferReflection::Type   bufferType,
        ProgramReflection::ShaderAccess             shaderAccess)
    {
        auto pSlangElementType = pSlangType->unwrapArray()->getElementTypeLayout();

        ProgramReflection::VariableMap varMap;

        ReflectionGenerationContext context = *pContext;
        context.pVariables = &varMap;

        initializeBufferVariables(
            &context,
            pPath,
            pSlangElementType);

        // TODO(tfoley): This is a bit of an ugly workaround, and it would
        // be good for the Slang API to make it unnecessary...
        auto category = pSlangType->getParameterCategory();
        if (category == ParameterCategory::Mixed)
        {
            if (pSlangType->getKind() == TypeReflection::Kind::ConstantBuffer)
            {
                category = ParameterCategory::ConstantBuffer;
            }
        }

        auto bindingIndex = getBindingIndex(pPath, category);
        auto bindingSpace = getBindingSpace(pPath, category);
        ProgramReflection::BindLocation bindLocation(
            bindingSpace,
            bindingIndex,
            shaderAccess);
        // If the buffer already exists in the program, make sure the definitions match
        const auto& prevDef = bufferDesc.nameMap.find(name);

        if (prevDef != bufferDesc.nameMap.end())
        {
            if (bindLocation != prevDef->second)
            {
                pContext->getLog() += to_string(bufferType) + " buffer '" + name + "' has different bind locations between different shader stages. Falcor do not support that. Use explicit bind locations to avoid this error";
                return false;
            }
            ProgramReflection::BufferReflection* pPrevBuffer = bufferDesc.descMap[bindLocation].get();
            std::string bufLog;
            if (validateBufferDeclaration(pPrevBuffer, varMap, bufLog) == false)
            {
                pContext->getLog() += to_string(bufferType) + " buffer '" + name + "' has different definitions between different shader stages. " + bufLog;
                return false;
            }
        }
        else
        {
            bool isArray = pSlangType->isArray();

            // Create the buffer reflection
            bufferDesc.nameMap[name] = bindLocation;
            bufferDesc.descMap[bindLocation] = ProgramReflection::BufferReflection::create(
                name,
                bindingSpace,
                bindingIndex,
                isArray ? (uint32_t)pSlangType->getTotalArrayElementCount() : 0,
                bufferType,
                getStructuredBufferType(pSlangType->getType()),
                (uint32_t)pSlangElementType->getSize(),
                varMap,
                ProgramReflection::ResourceMap(),
                shaderAccess);
        }

        // For now we expose all buffers as visible to every stage
        uint32_t mask = 0xFFFFFFFF;
        bufferDesc.descMap[bindLocation]->setShaderMask(mask);

        return true;
    }

    bool ProgramReflection::reflectVertexAttributes(
        ShaderReflection*   pSlangReflector,
        std::string&        log)
    {
        // TODO(tfoley): Add vertex input reflection capability to Slang
        return true;
    }

    bool ProgramReflection::reflectPixelShaderOutputs(
        ShaderReflection*    pSlangReflector,
        std::string&                log)
    {
        // TODO(tfoley): Add fragment output reflection capability to Slang
        return true;
    }

    // TODO(tfoley): Should try to strictly use type...
    static ProgramReflection::Resource::ResourceType getResourceType(VariableLayoutReflection* pParameter)
    {
        switch (pParameter->getCategory())
        {
        case ParameterCategory::SamplerState:
            return ProgramReflection::Resource::ResourceType::Sampler;
        case ParameterCategory::ShaderResource:
        case ParameterCategory::UnorderedAccess:
            switch (pParameter->getType()->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK)
            {
            case SLANG_BYTE_ADDRESS_BUFFER:
                return ProgramReflection::Resource::ResourceType::RawBuffer;

            case SLANG_STRUCTURED_BUFFER:
                return ProgramReflection::Resource::ResourceType::StructuredBuffer;

            default:
                return ProgramReflection::Resource::ResourceType::Texture;

            case SLANG_RESOURCE_NONE:
                break;
            }
            break;
        case ParameterCategory::Mixed:
            // TODO: propagate this information up the Falcor level
            return ProgramReflection::Resource::ResourceType::Unknown;
        default:
            break;
        }
        should_not_get_here();
        return ProgramReflection::Resource::ResourceType::Unknown;
    }

    static ProgramReflection::ShaderAccess getShaderAccess(ParameterCategory category)
    {
        switch (category)
        {
        case ParameterCategory::ShaderResource:
        case ParameterCategory::SamplerState:
            return ProgramReflection::ShaderAccess::Read;
        case ParameterCategory::UnorderedAccess:
            return ProgramReflection::ShaderAccess::ReadWrite;
        case ParameterCategory::Mixed:
            return ProgramReflection::ShaderAccess::Undefined;
        default:
            should_not_get_here();
            return ProgramReflection::ShaderAccess::Undefined;
        }
    }

#if 0
    bool reflectResource(
        ReflectionGenerationContext*    pContext,
        ParameterReflection*     pParameter)
    {
        ProgramReflection::Resource falcorDesc;
        std::string name(pParameter->getName());

        falcorDesc.type = getResourceType(pParameter);
        falcorDesc.shaderAccess = getShaderAccess(pParameter->getCategory());
        if (falcorDesc.type == ProgramReflection::Resource::ResourceType::Texture)
        {
            falcorDesc.retType = getReturnType(pParameter->getType()->getTextureResultType());
            falcorDesc.dims = getResourceDimensions(pParameter->getType()->getTextureShape());
        }
        bool isArray = pParameter->getType()->isArray();
        falcorDesc.regIndex = pParameter->getBindingIndex();
        falcorDesc.registerSpace = pParameter->getBindingSpace();
        assert(falcorDesc.registerSpace == 0);
        falcorDesc.arraySize = isArray ? (uint32_t)pParameter->getType()->getTotalArrayElementCount() : 0;

        // If this already exists, definitions should match
        auto& resourceMap = pContext->getResourceMap();
        const auto& prevDef = resourceMap.find(name);
        if (prevDef == resourceMap.end())
        {
            resourceMap[name] = falcorDesc;
        }
        else
        {
            std::string varLog;
            if (verifyResourceDefinition(prevDef->second, falcorDesc, varLog) == false)
            {
                pContext->getLog() += "Shader resource '" + std::string(name) + "' has different definitions between different shader stages. " + varLog;
                return false;
            }
        }

        // Update the mask
        resourceMap[name].shaderMask |= (1 << pContext->shaderIndex);

        return true;
    }
#endif

    static bool reflectStructuredBuffer(
        ReflectionGenerationContext*    pContext,
        TypeLayoutReflection*    pSlangType,
        const std::string&              name,
        ReflectionPath*                 path)
    {
        auto shaderAccess = getShaderAccess(pSlangType->getType());
        return reflectBuffer(
            pContext,
            pSlangType,
            name,
            path,
            pContext->pReflector->mBuffers[(uint32_t)ProgramReflection::BufferReflection::Type::Structured],
            ProgramReflection::BufferReflection::Type::Structured,
            shaderAccess);
    }

    static bool reflectConstantBuffer(
        ReflectionGenerationContext*    pContext,
        TypeLayoutReflection*    pSlangType,
        const std::string&              name,
        ReflectionPath*                 path)
    {
        return reflectBuffer(
            pContext,
            pSlangType,
            name,
            path,
            pContext->pReflector->mBuffers[(uint32_t)ProgramReflection::BufferReflection::Type::Constant],
            ProgramReflection::BufferReflection::Type::Constant,
            ProgramReflection::ShaderAccess::Read);
    }

#if 0
    static bool reflectStructuredBuffer(
        ReflectionGenerationContext*    pContext,
        BufferReflection*        slangParam)
    {
        auto shaderAccess = getShaderAccess(slangParam->getCategory());
        return reflectBuffer(
            pContext,
            slangParam,
            pContext->pReflector->mBuffers[(uint32_t)ProgramReflection::BufferReflection::Type::Structured],
            ProgramReflection::BufferReflection::Type::Structured,
            ProgramReflection::ShaderAccess::Read);
    }

    static bool reflectConstantBuffer(
        ReflectionGenerationContext*    pContext,
        BufferReflection*        slangParam)
    {
        return reflectBuffer(
            pContext,
            slangParam,
            pContext->pReflector->mBuffers[(uint32_t)ProgramReflection::BufferReflection::Type::Constant],
            ProgramReflection::BufferReflection::Type::Constant,
            ProgramReflection::ShaderAccess::Read);
    }

    static bool reflectVariable(
        ReflectionGenerationContext*        pContext,
        VariableLayoutReflection*    slangVar)
    {
        switch (slangVar->getCategory())
        {
        case ParameterCategory::ConstantBuffer:
            return reflectConstantBuffer(pContext, slangVar->asBuffer());

        case ParameterCategory::ShaderResource:
        case ParameterCategory::UnorderedAccess:
        case ParameterCategory::SamplerState:
            return reflectResource(pContext, slangVar);

        case ParameterCategory::Mixed:
        {
            // The parameter spans multiple binding kinds (e.g., both texture and uniform).
            // We need to recursively split it into sub-parameters, each using a single
            // kind of bindable resource.
            //
            // Also, the parameter may have been declared as a constant buffer, so
            // we need to reflect it directly in that case:
            //
            switch (slangVar->getType()->getKind())
            {
            case TypeReflection::Kind::ConstantBuffer:
                return reflectConstantBuffer(pContext, slangVar->asBuffer());

            default:
                //
                // Okay, let's walk it recursively to bind the sub-pieces...
                //
                logError("unimplemented: global-scope uniforms");
                break;
            }
        }
        break;


        case ParameterCategory::Uniform:
            // Ignore uniform parameters during this pass...
            logError("unimplemented: global-scope uniforms");
            return true;

        default:
            break;
        }
    }
#endif

    static bool reflectParameter(
        ReflectionGenerationContext*        pContext,
        VariableLayoutReflection*    slangParam)
    {
        reflectVariable(pContext, slangParam, nullptr);
        return true;
    }

    bool ProgramReflection::reflectResources(
        ShaderReflection*    pSlangReflector,
        std::string&                log)
    {
        ReflectionGenerationContext context;
        context.pReflector = this;
        context.pResourceMap = &mResources;
        context.pLog = &log;

        bool res = true;

        uint32_t paramCount = pSlangReflector->getParameterCount();
        for (uint32_t pp = 0; pp < paramCount; ++pp)
        {
            VariableLayoutReflection* param = pSlangReflector->getParameterByIndex(pp);
            res = reflectParameter(&context, param);
        }

        // Also extract entry-point stuff

        SlangUInt entryPointCount = pSlangReflector->getEntryPointCount();
        for (SlangUInt ee = 0; ee < entryPointCount; ++ee)
        {
            EntryPointReflection* entryPoint = pSlangReflector->getEntryPointByIndex(ee);

            // We need to reflect entry-point parameters just like any others
            context.stage = entryPoint->getStage();

            uint32_t entryPointParamCount = entryPoint->getParameterCount();
            for (uint32_t pp = 0; pp < entryPointParamCount; ++pp)
            {
                VariableLayoutReflection* param = entryPoint->getParameterByIndex(pp);
                res = reflectParameter(&context, param);
            }


            switch (entryPoint->getStage())
            {
            case SLANG_STAGE_COMPUTE:
            {
                SlangUInt sizeAlongAxis[3];
                entryPoint->getComputeThreadGroupSize(3, &sizeAlongAxis[0]);
                mThreadGroupSize.x = (uint32_t)sizeAlongAxis[0];
                mThreadGroupSize.y = (uint32_t)sizeAlongAxis[1];
                mThreadGroupSize.z = (uint32_t)sizeAlongAxis[2];
            }
            break;
            case SLANG_STAGE_PIXEL:
#ifdef FALCOR_VK
                mIsSampleFrequency = entryPoint->usesAnySampleRateInput();
#else
                mIsSampleFrequency = true; // #SLANG Slang reports false for DX shaders. There's an open issue, once it's fixed we should remove that
#endif            default:
                break;
            }
        }
        return res;
    }
}