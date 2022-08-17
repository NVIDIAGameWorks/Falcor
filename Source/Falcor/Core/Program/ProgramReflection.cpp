/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "ProgramReflection.h"
#include "Program.h"
#include "ProgramVersion.h"
#include "Utils/StringUtils.h"

#include <slang.h>

#include <map>

using namespace slang;

namespace Falcor
{
    namespace
    {
        const char* kRootDescriptorAttribute = "root";
    }

    TypedShaderVarOffset::TypedShaderVarOffset(
        const ReflectionType* pType,
        ShaderVarOffset       offset)
        : ShaderVarOffset(offset)
        , mpType(pType->shared_from_this())
    {}

    TypedShaderVarOffset TypedShaderVarOffset::operator[](const std::string& name) const
    {
        if (!isValid()) return *this;

        auto pType = getType();

        if (auto pStructType = pType->asStructType())
        {
            if (auto pMember = pStructType->findMember(name))
            {
                return TypedShaderVarOffset(
                    pMember->getType().get(),
                    (*this) + pMember->getBindLocation());
            }
        }

        reportError(fmt::format("No member named '{}' found.", name));
        return TypedShaderVarOffset();
    }

    TypedShaderVarOffset TypedShaderVarOffset::operator[](const char* name) const
    {
        return (*this)[std::string(name)];
    }

    TypedShaderVarOffset TypedShaderVarOffset::operator[](size_t index) const
    {
        throw 99;
    }

    TypedShaderVarOffset ReflectionType::getZeroOffset() const
    {
        return TypedShaderVarOffset(this, ShaderVarOffset::kZero);
    }

    TypedShaderVarOffset ReflectionType::getMemberOffset(const std::string& name) const
    {
        return getZeroOffset()[name];
    }

    // Represents one link in a "breadcrumb trail" leading from a particular variable
    // back through the path of member-access operations that led to it.
    // E.g., when trying to construct information for `foo.bar.baz`
    // we might have a path that consists of:
    //
    // - An link for the field `baz` in type `Bar` (which knows its offset)
    // - An link for the field `bar` in type `Foo`
    // - An link for the top-level shader parameter `foo`
    //
    // To compute the correct offset for `baz` we can walk up this chain
    // and add up offsets.
    //
    // In simple cases, one can track this info top-down, by simply keeping
    // a "running total" offset, but that doesn't account for the fact that
    // `baz` might be a texture, UAV, sampler, or uniform, and the offset
    // we'd need to track for each case is different.
    //
    struct ReflectionPathLink
    {
        const ReflectionPathLink* pParent = nullptr;
        VariableLayoutReflection* pVar = nullptr;
    };

    // Represents a full"breadcrumb trail" leading from a particular variable
    // back through the path of member-access operations that led to it.
    //
    // The `pPrimary` field represents the main path that gets used for
    // ordinary uniform, texture, buffer, etc. variables. In the 99% case
    // this is all that ever gets used.
    //
    // The `pDeferred` field represents a secondary path that describes
    // where the data that arose due to specialization ended up.
    //
    // E.g., if we have a program like:
    //
    //     struct MyStuff { IFoo f; Texture2D t; }
    //     MyStuff gStuff;
    //     Texture2D gOther;
    //
    // Then `gStuff` will be assigned a starting `t` register of `t0`,
    // and the *primary* path for `gStuff.t` will show that offset.
    //
    // However, if `gStuff.f` gets specialized to some type `Bar`:
    //
    //     struct Bar { Texture2D b; }
    //
    // Then the `gStuff.f.b` field also needs a texture register to be
    // assigned. It can't use registers `t0` or `t1` since those were
    // already allocated in the unspecialized program (to `gStuff.t`
    // and `gOther`, respectively), so it needs to use `t2`.
    //
    // But that means that the allocation for `gStuff` is split into two
    // pieces: a "primary" allocation for `gStuff.t`, and then a secondary
    // allocation for `gStuff.f` that got "deferred" until after specialization
    // (which means it comes after all the un-specialized parameters).
    //
    // The Slang reflection information lets us query both the primary
    // and deferred allocation/layout for a shader parameter, and we
    // need to handle both in order to support specialization.
    //
    struct ReflectionPath
    {
        ReflectionPathLink* pPrimary = nullptr;
        ReflectionPathLink* pDeferred = nullptr;
    };

    // A helper RAII type to extend a `ReflectionPath` with
    // additional links as needed based on the reflection
    // information from a Slang `VariableLayoutReflection`.
    //
    struct ExtendedReflectionPath : ReflectionPath
    {
        ExtendedReflectionPath(
            ReflectionPath const*       pParent,
            VariableLayoutReflection*   pVar)
        {
            // If there is any path stored in `pParent`,
            // then that will be our starting point.
            //
            if(pParent)
            {
                pPrimary = pParent->pPrimary;
                pDeferred = pParent->pDeferred;
            }

            // Next, if `pVar` has a primary layout (and/or
            // an optional pending/deferred layout), then
            // we will extend the appropriate breadcrumb
            // trail with its information.
            //
            if(pVar)
            {
                primaryLinkStorage.pParent = pPrimary;
                primaryLinkStorage.pVar = pVar;
                pPrimary = &primaryLinkStorage;

                if( auto pDeferredVar = pVar->getPendingDataLayout() )
                {
                    deferredLinkStorage.pParent = pDeferred;
                    deferredLinkStorage.pVar = pDeferredVar;
                    pDeferred = &deferredLinkStorage;
                }
            }
        }

        // These "storage" fields are used in the constructor
        // when it needs to allocate additional links. By pre-allocating
        // them here in the body of the type we avoid having to do
        // heap allocation when constructing an extended path.
        //
        ReflectionPathLink primaryLinkStorage;
        ReflectionPathLink deferredLinkStorage;
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
        case TypeReflection::Kind::TextureBuffer:
            return ReflectionResourceType::Type::TypedBuffer;
        case TypeReflection::Kind::Resource:
            switch (pSlangType->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK)
            {
            case SLANG_STRUCTURED_BUFFER:
                return ReflectionResourceType::Type::StructuredBuffer;
            case SLANG_BYTE_ADDRESS_BUFFER:
                return ReflectionResourceType::Type::RawBuffer;
            case SLANG_TEXTURE_BUFFER:
                return ReflectionResourceType::Type::TypedBuffer;
            case SLANG_ACCELERATION_STRUCTURE:
                return ReflectionResourceType::Type::AccelerationStructure;
            case SLANG_TEXTURE_1D:
            case SLANG_TEXTURE_2D:
            case SLANG_TEXTURE_3D:
            case SLANG_TEXTURE_CUBE:
                return ReflectionResourceType::Type::Texture;
            default:
                FALCOR_UNREACHABLE();
                return ReflectionResourceType::Type(-1);
            }
        default:
            FALCOR_UNREACHABLE();
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
        case SLANG_ACCELERATION_STRUCTURE:
            return ReflectionResourceType::Dimensions::AccelerationStructure;

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
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Bool;
            case 2: return ReflectionBasicType::Type::Bool2;
            case 3: return ReflectionBasicType::Type::Bool3;
            case 4: return ReflectionBasicType::Type::Bool4;
            }
            break;
        case TypeReflection::ScalarType::UInt8:
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Uint8;
            case 2: return ReflectionBasicType::Type::Uint8_2;
            case 3: return ReflectionBasicType::Type::Uint8_3;
            case 4: return ReflectionBasicType::Type::Uint8_4;
            }
            break;
        case TypeReflection::ScalarType::UInt16:
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Uint16;
            case 2: return ReflectionBasicType::Type::Uint16_2;
            case 3: return ReflectionBasicType::Type::Uint16_3;
            case 4: return ReflectionBasicType::Type::Uint16_4;
            }
            break;
        case TypeReflection::ScalarType::UInt32:
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Uint;
            case 2: return ReflectionBasicType::Type::Uint2;
            case 3: return ReflectionBasicType::Type::Uint3;
            case 4: return ReflectionBasicType::Type::Uint4;
            }
            break;
        case TypeReflection::ScalarType::UInt64:
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Uint64;
            case 2: return ReflectionBasicType::Type::Uint64_2;
            case 3: return ReflectionBasicType::Type::Uint64_3;
            case 4: return ReflectionBasicType::Type::Uint64_4;
            }
            break;
        case TypeReflection::ScalarType::Int8:
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Int8;
            case 2: return ReflectionBasicType::Type::Int8_2;
            case 3: return ReflectionBasicType::Type::Int8_3;
            case 4: return ReflectionBasicType::Type::Int8_4;
            }
            break;
        case TypeReflection::ScalarType::Int16:
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Int16;
            case 2: return ReflectionBasicType::Type::Int16_2;
            case 3: return ReflectionBasicType::Type::Int16_3;
            case 4: return ReflectionBasicType::Type::Int16_4;
            }
            break;
        case TypeReflection::ScalarType::Int32:
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Int;
            case 2: return ReflectionBasicType::Type::Int2;
            case 3: return ReflectionBasicType::Type::Int3;
            case 4: return ReflectionBasicType::Type::Int4;
            }
            break;
        case TypeReflection::ScalarType::Int64:
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Int64;
            case 2: return ReflectionBasicType::Type::Int64_2;
            case 3: return ReflectionBasicType::Type::Int64_3;
            case 4: return ReflectionBasicType::Type::Int64_4;
            }
            break;
        case TypeReflection::ScalarType::Float16:
            switch (rows)
            {
            case 1:
                switch (columns)
                {
                case 1: return ReflectionBasicType::Type::Float16;
                case 2: return ReflectionBasicType::Type::Float16_2;
                case 3: return ReflectionBasicType::Type::Float16_3;
                case 4: return ReflectionBasicType::Type::Float16_4;
                }
                break;
            case 2:
                switch (columns)
                {
                case 2: return ReflectionBasicType::Type::Float16_2x2;
                case 3: return ReflectionBasicType::Type::Float16_2x3;
                case 4: return ReflectionBasicType::Type::Float16_2x4;
                }
                break;
            case 3:
                switch (columns)
                {
                case 2: return ReflectionBasicType::Type::Float16_3x2;
                case 3: return ReflectionBasicType::Type::Float16_3x3;
                case 4: return ReflectionBasicType::Type::Float16_3x4;
                }
                break;
            case 4:
                switch (columns)
                {
                case 2: return ReflectionBasicType::Type::Float16_4x2;
                case 3: return ReflectionBasicType::Type::Float16_4x3;
                case 4: return ReflectionBasicType::Type::Float16_4x4;
                }
                break;
            }
            break;
        case TypeReflection::ScalarType::Float32:
            switch (rows)
            {
            case 1:
                switch (columns)
                {
                case 1: return ReflectionBasicType::Type::Float;
                case 2: return ReflectionBasicType::Type::Float2;
                case 3: return ReflectionBasicType::Type::Float3;
                case 4: return ReflectionBasicType::Type::Float4;
                }
                break;
            case 2:
                switch (columns)
                {
                case 2: return ReflectionBasicType::Type::Float2x2;
                case 3: return ReflectionBasicType::Type::Float2x3;
                case 4: return ReflectionBasicType::Type::Float2x4;
                }
                break;
            case 3:
                switch (columns)
                {
                case 2: return ReflectionBasicType::Type::Float3x2;
                case 3: return ReflectionBasicType::Type::Float3x3;
                case 4: return ReflectionBasicType::Type::Float3x4;
                }
                break;
            case 4:
                switch (columns)
                {
                case 2: return ReflectionBasicType::Type::Float4x2;
                case 3: return ReflectionBasicType::Type::Float4x3;
                case 4: return ReflectionBasicType::Type::Float4x4;
                }
                break;
            }
            break;
        case TypeReflection::ScalarType::Float64:
            FALCOR_ASSERT(rows == 1);
            switch (columns)
            {
            case 1: return ReflectionBasicType::Type::Float64;
            case 2: return ReflectionBasicType::Type::Float64_2;
            case 3: return ReflectionBasicType::Type::Float64_3;
            case 4: return ReflectionBasicType::Type::Float64_4;
            }
            break;
        }

        FALCOR_UNREACHABLE();
        return ReflectionBasicType::Type::Unknown;
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
            FALCOR_UNREACHABLE();
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

    ReflectionVar::SharedPtr reflectVariable(
        VariableLayoutReflection*   pSlangLayout,
        ShaderVarOffset::RangeIndex rangeIndex,
        ParameterBlockReflection*   pBlock,
        ReflectionPath*             pPath,
        ProgramVersion const*       pProgramVersion);

    ReflectionType::SharedPtr reflectType(
        TypeLayoutReflection*       pSlangType,
        ParameterBlockReflection*   pBlock,
        ReflectionPath*             pPath,
        ProgramVersion const*       pProgramVersion);

        // Determine if a Slang type layout consumes any storage/resources of the given kind
    static bool hasUsage(slang::TypeLayoutReflection* pSlangTypeLayout, SlangParameterCategory resourceKind)
    {
        auto kindCount = pSlangTypeLayout->getCategoryCount();
        for(unsigned int ii = 0; ii < kindCount; ++ii)
        {
            if(pSlangTypeLayout->getCategoryByIndex(ii) == resourceKind)
                return true;
        }
        return false;
    }

        // Given a "breadcrumb trail" (reflection path), determine
        // the actual register/binding that will be used by a leaf
        // parameter for the given resource kind.
    static size_t getRegisterIndexFromPath(const ReflectionPathLink* pPath, SlangParameterCategory category)
    {
        uint32_t offset = 0;
        for (auto pp = pPath; pp; pp = pp->pParent)
        {
            if (pp->pVar)
            {
                // We are in the process of walking up from a leaf
                // shader variable to the root (some global shader
                // parameter).
                //
                // If along the way we run into a parameter block,
                // *and* that parameter block has been allocated
                // into its own register space, then we should stop
                // adding contributions to the register/binding of
                // the leaf parameter, since any register offsets
                // coming from "above" this point shouldn't affect
                // the register/binding of a parameter inside of
                // the parameter block.
                //
                // TODO: This logic is really fiddly and doesn't
                // seem like something Falcor should have to do.
                // The Slang library should be provided utility
                // functions to handle this stuff.
                //
                if(pp->pVar->getType()->getKind() == slang::TypeReflection::Kind::ParameterBlock
                    && hasUsage(pp->pVar->getTypeLayout(), SLANG_PARAMETER_CATEGORY_REGISTER_SPACE)
                    && category != SLANG_PARAMETER_CATEGORY_REGISTER_SPACE)
                {
                    return offset;
                }

                offset += (uint32_t)pp->pVar->getOffset(category);
                continue;
            }
            throw RuntimeError("Invalid reflection path");
        }
        return offset;
    }

    static uint32_t getRegisterSpaceFromPath(const ReflectionPathLink* pPath, SlangParameterCategory category)
    {
        uint32_t offset = 0;
        for (auto pp = pPath; pp; pp = pp->pParent)
        {
            if (pp->pVar)
            {
                // Similar to the case above in `getRegisterIndexFromPath`,
                // if we are walking from a member in a parameter block
                // up to the block itself, then the space for our parameter
                // should be offset by the register space assigned to
                // the block itself, and we should stop walking up
                // the breadcrumb trail.
                //
                // TODO: Just as in `getRegisterIndexFromPath` this is way
                // too subtle, and Slang should be providing a service
                // to compute this.
                //
                if(pp->pVar->getTypeLayout()->getKind() == slang::TypeReflection::Kind::ParameterBlock)
                {
                    return offset + (uint32_t) getRegisterIndexFromPath(pp, SLANG_PARAMETER_CATEGORY_REGISTER_SPACE);
                }
                offset += (uint32_t)pp->pVar->getBindingSpace(category);
                continue;
            }

            throw RuntimeError("Invalid reflection path");
        }
        return offset;
    }

    static ShaderResourceType getShaderResourceType(
        const ReflectionResourceType* pType);

    static ParameterCategory getParameterCategory(TypeLayoutReflection* pTypeLayout);

    static void extractDefaultConstantBufferBinding(
        slang::TypeLayoutReflection*    pSlangType,
        ReflectionPath*                 pPath,
        ParameterBlockReflection*       pBlock,
        bool                            shouldUseRootConstants)
    {
        auto pContainerLayout = pSlangType->getContainerVarLayout();
        FALCOR_ASSERT(pContainerLayout);

        ExtendedReflectionPath containerPath(pPath, pContainerLayout);
        int32_t containerCategoryCount = pContainerLayout->getCategoryCount();
        for (int32_t containerCategoryIndex = 0; containerCategoryIndex < containerCategoryCount; ++containerCategoryIndex)
        {
            auto containerCategory = pContainerLayout->getCategoryByIndex(containerCategoryIndex);
            switch (containerCategory)
            {
            case slang::ParameterCategory::DescriptorTableSlot:
            case slang::ParameterCategory::ConstantBuffer:
            {
                ParameterBlockReflection::DefaultConstantBufferBindingInfo defaultConstantBufferInfo;
                defaultConstantBufferInfo.regIndex = (uint32_t)getRegisterIndexFromPath(containerPath.pPrimary, containerCategory);
                defaultConstantBufferInfo.regSpace = getRegisterSpaceFromPath(containerPath.pPrimary, containerCategory);
                defaultConstantBufferInfo.useRootConstants = shouldUseRootConstants;
                pBlock->setDefaultConstantBufferBindingInfo(defaultConstantBufferInfo);
            }
            break;

            default:
                break;
            }
        }
    }

    ReflectionType::SharedPtr reflectResourceType(
        TypeLayoutReflection*       pSlangType,
        ParameterBlockReflection*   pBlock,
        ReflectionPath*             pPath,
        ProgramVersion const*       pProgramVersion)
    {
        ReflectionResourceType::Type type = getResourceType(pSlangType->getType());
        ReflectionResourceType::Dimensions dims = getResourceDimensions(pSlangType->getResourceShape());;
        ReflectionResourceType::ShaderAccess shaderAccess = getShaderAccess(pSlangType->getType());
        ReflectionResourceType::ReturnType retType = getReturnType(pSlangType->getType());
        ReflectionResourceType::StructuredType structuredType = getStructuredBufferType(pSlangType->getType());

        FALCOR_ASSERT(pPath->pPrimary && pPath->pPrimary->pVar);
        std::string name = pPath->pPrimary->pVar->getName();

        // Check if resource type represents a root descriptor.
        // In the shader we use a custom [root] attribute to flag resources to map to root descriptors.
        auto pVar = pPath->pPrimary->pVar->getVariable();
        bool isRootDescriptor = pVar->findUserAttributeByName(pProgramVersion->getSlangSession()->getGlobalSession(), kRootDescriptorAttribute) != nullptr;

        // Check that the root descriptor type is supported.
        if (isRootDescriptor)
        {
            // Check the resource type and shader access.
            if (type != ReflectionResourceType::Type::RawBuffer && type != ReflectionResourceType::Type::StructuredBuffer &&
                type != ReflectionResourceType::Type::AccelerationStructure)
            {
                throw RuntimeError("Resource '{}' cannot be bound as root descriptor. Only raw buffers, structured buffers, and acceleration structures are supported.", name);
            }
            if (shaderAccess != ReflectionResourceType::ShaderAccess::Read &&
                shaderAccess != ReflectionResourceType::ShaderAccess::ReadWrite)
            {
                throw RuntimeError("Buffer '{}' cannot be bound as root descriptor. Only SRV/UAVs are supported.", name);
            }
            FALCOR_ASSERT(type != ReflectionResourceType::Type::AccelerationStructure || shaderAccess == ReflectionResourceType::ShaderAccess::Read);

            // Check that it's not an append/consume structured buffer, which is unsupported for root descriptors.
            // RWStructuredBuffer with counter is also not supported, but we cannot see that on the type declaration.
            // At bind time, we'll validate that the buffer has not been created with a UAV counter.
            if (type == ReflectionResourceType::Type::StructuredBuffer)
            {
                FALCOR_ASSERT(structuredType != ReflectionResourceType::StructuredType::Invalid);
                if (structuredType == ReflectionResourceType::StructuredType::Append || structuredType == ReflectionResourceType::StructuredType::Consume)
                {
                    throw RuntimeError("StructuredBuffer '{}' cannot be bound as root descriptor. Only regular structured buffers are supported, not append/consume buffers.", name);
                }
            }
            FALCOR_ASSERT(dims == ReflectionResourceType::Dimensions::Buffer || dims == ReflectionResourceType::Dimensions::AccelerationStructure); // We shouldn't get here otherwise
        }

        ReflectionResourceType::SharedPtr pType = ReflectionResourceType::create(type, dims, structuredType, retType, shaderAccess, pSlangType);

        ParameterCategory category = getParameterCategory(pSlangType);
        ParameterBlockReflection::ResourceRangeBindingInfo bindingInfo;
        bindingInfo.regIndex = (uint32_t)getRegisterIndexFromPath(pPath->pPrimary, category);
        bindingInfo.regSpace = getRegisterSpaceFromPath(pPath->pPrimary, category);
        bindingInfo.dimension = dims;

        if (isRootDescriptor) bindingInfo.flavor = ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::RootDescriptor;

        switch (type)
        {
        default:
            break;

        case ReflectionResourceType::Type::StructuredBuffer:
        {
            const auto& pElementLayout = pSlangType->getElementTypeLayout();
            auto pBufferType = reflectType(
                pElementLayout,
                pBlock,
                pPath,
                pProgramVersion);
            pType->setStructType(pBufferType);
        }
        break;

        // TODO: The fact that constant buffers (and parameter blocks, since Falcor currently
        // pretends that parameter blocks are constant buffers in its reflection types) are
        // treated so differently from other resource types is a huge sign that they should
        // *not* be resource types to begin with (and they *aren't* resource types in Slang).
        //
        case ReflectionResourceType::Type::ConstantBuffer:
        {
            // We have a sub-parameter-block (whether a true parameter block, or just a constant buffer)
            auto pSubBlock = ParameterBlockReflection::createEmpty(pProgramVersion);
            const auto& pElementLayout = pSlangType->getElementTypeLayout();
            auto pElementType = reflectType(
                pElementLayout,
                pSubBlock.get(),
                pPath,
                pProgramVersion);
            pSubBlock->setElementType(pElementType);

            extractDefaultConstantBufferBinding(pSlangType, pPath, pSubBlock.get(), /*shouldUseRootConstants:*/false);

            pSubBlock->finalize();

            pType->setStructType(pElementType);
            pType->setParameterBlockReflector(pSubBlock);

            // TODO: `pSubBlock` should probably get stored on the
            // `ReflectionResourceType` somewhere, so that we can
            // retrieve it later without having to use a parent
            // `ParameterBlockReflection` to look it up.

            if (pSlangType->getKind() == slang::TypeReflection::Kind::ParameterBlock)
            {
                bindingInfo.flavor = ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ParameterBlock;
            }
            else
            {
                bindingInfo.flavor = ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ConstantBuffer;
            }
            bindingInfo.pSubObjectReflector = pSubBlock;
        }
        break;
        }

        if (pBlock)
        {
            pBlock->addResourceRange(bindingInfo);
        }

        return pType;
    }

    ReflectionType::SharedPtr reflectStructType(
        TypeLayoutReflection*       pSlangType,
        ParameterBlockReflection*   pBlock,
        ReflectionPath*             pPath,
        ProgramVersion const*       pProgramVersion)
    {
        // Note: not all types have names. In particular, the "element type" of
        // a `cbuffer` declaration is an anonymous `struct` type, and Slang
        // returns `nullptr` from `getName().
        //
        auto pSlangName = pSlangType->getName();
        auto name = pSlangName ? std::string(pSlangName) : std::string();

         ReflectionStructType::SharedPtr pType = ReflectionStructType::create(
            pSlangType->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM),
            name,
            pSlangType);

        ReflectionStructType::BuildState buildState;

        for (uint32_t i = 0; i < pSlangType->getFieldCount(); i++)
        {
            auto pSlangField = pSlangType->getFieldByIndex(i);
            ExtendedReflectionPath fieldPath(pPath, pSlangField);

            ReflectionVar::SharedPtr pVar = reflectVariable(
                pSlangField,
                pType->getResourceRangeCount(),
                pBlock,
                &fieldPath,
                pProgramVersion);
            if(pVar) pType->addMember(pVar, buildState);
        }
        return pType;
    }

    static ReflectionType::ByteSize getByteSize(TypeLayoutReflection* pSlangType)
    {
        return pSlangType->getSize(SLANG_PARAMETER_CATEGORY_UNIFORM);
    }

    ReflectionType::SharedPtr reflectArrayType(
        TypeLayoutReflection*       pSlangType,
        ParameterBlockReflection*   pBlock,
        ReflectionPath*             pPath,
        ProgramVersion const*       pProgramVersion)
    {
        uint32_t elementCount = (uint32_t)pSlangType->getElementCount();
        uint32_t elementByteStride = (uint32_t)pSlangType->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);

        ReflectionType::SharedPtr pElementType = reflectType(
            pSlangType->getElementTypeLayout(),
            pBlock,
            pPath,
            pProgramVersion);
        ReflectionArrayType::SharedPtr pArrayType = ReflectionArrayType::create(
            elementCount, elementByteStride, pElementType,
            getByteSize(pSlangType),
            pSlangType);
        return pArrayType;
    }

    ReflectionType::SharedPtr reflectBasicType(TypeLayoutReflection* pSlangType)
    {
        const bool isRowMajor = pSlangType->getMatrixLayoutMode() == SLANG_MATRIX_LAYOUT_ROW_MAJOR;
        ReflectionBasicType::Type type = getVariableType(pSlangType->getScalarType(), pSlangType->getRowCount(), pSlangType->getColumnCount());
        ReflectionType::SharedPtr pType = ReflectionBasicType::create(type, isRowMajor, pSlangType->getSize(), pSlangType);
        return pType;
    }

    ReflectionType::SharedPtr reflectInterfaceType(
        TypeLayoutReflection*       pSlangType,
        ParameterBlockReflection*   pBlock,
        ReflectionPath*             pPath,
        ProgramVersion const*       pProgramVersion)
    {
        auto pType = ReflectionInterfaceType::create(pSlangType);

        ParameterCategory category = getParameterCategory(pSlangType);
        ParameterBlockReflection::ResourceRangeBindingInfo bindingInfo;
        bindingInfo.regIndex = (uint32_t)getRegisterIndexFromPath(pPath->pPrimary, category);
        bindingInfo.regSpace = getRegisterSpaceFromPath(pPath->pPrimary, category);

        bindingInfo.flavor = ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface;

        if(auto pSlangPendingTypeLayout = pSlangType->getPendingDataTypeLayout())
        {
            ReflectionPath subPath;
            subPath.pPrimary = pPath->pDeferred;
            subPath.pDeferred = nullptr;

            auto pPendingBlock = ParameterBlockReflection::createEmpty(pProgramVersion);
            auto pPendingType = reflectType(
                pSlangPendingTypeLayout,
                pPendingBlock.get(),
                &subPath,
                pProgramVersion);
            pPendingBlock->setElementType(pPendingType);

            // TODO: What to do if `pPendingType->getByteSize()` is non-zero?

            pPendingBlock->finalize();

            pType->setParameterBlockReflector(pPendingBlock);

            bindingInfo.pSubObjectReflector = pPendingBlock;

            category = slang::ParameterCategory::Uniform;
            bindingInfo.regIndex = (uint32_t) getRegisterIndexFromPath(pPath->pDeferred, category);
            bindingInfo.regSpace = getRegisterSpaceFromPath(pPath->pPrimary, category);
        }

        if(pBlock)
        {
            pBlock->addResourceRange(bindingInfo);
        }

        return pType;
    }

    ReflectionType::SharedPtr reflectSpecializedType(
        TypeLayoutReflection*       pSlangType,
        ParameterBlockReflection*   pBlock,
        ReflectionPath*             pPath,
        ProgramVersion const*       pProgramVersion)
    {
        auto pSlangBaseType = pSlangType->getElementTypeLayout();

        auto pSlangVarLayout = pSlangType->getSpecializedTypePendingDataVarLayout();

        ReflectionPathLink deferredLink;
        deferredLink.pParent = pPath->pPrimary;
        deferredLink.pVar = pSlangVarLayout;

        ReflectionPath path;
        path.pPrimary = pPath->pPrimary;
        path.pDeferred = &deferredLink;

        return reflectType(
            pSlangBaseType,
            pBlock,
            &path,
            pProgramVersion);
    }

    ReflectionType::SharedPtr reflectType(
        TypeLayoutReflection*       pSlangType,
        ParameterBlockReflection*   pBlock,
        ReflectionPath*             pPath,
        ProgramVersion const*       pProgramVersion)
    {
        FALCOR_ASSERT(pSlangType);
        auto kind = pSlangType->getType()->getKind();
        switch (kind)
        {
        case TypeReflection::Kind::ParameterBlock:
        case TypeReflection::Kind::Resource:
        case TypeReflection::Kind::SamplerState:
        case TypeReflection::Kind::ConstantBuffer:
        case TypeReflection::Kind::ShaderStorageBuffer:
        case TypeReflection::Kind::TextureBuffer:
            return reflectResourceType(pSlangType, pBlock, pPath, pProgramVersion);
        case TypeReflection::Kind::Struct:
            return reflectStructType(pSlangType, pBlock, pPath, pProgramVersion);
        case TypeReflection::Kind::Array:
            return reflectArrayType(pSlangType, pBlock, pPath, pProgramVersion);
        case TypeReflection::Kind::Interface:
            return reflectInterfaceType(pSlangType, pBlock, pPath, pProgramVersion);
        case TypeReflection::Kind::Specialized:
            return reflectSpecializedType(pSlangType, pBlock, pPath, pProgramVersion);
        case TypeReflection::Kind::Scalar:
        case TypeReflection::Kind::Matrix:
        case TypeReflection::Kind::Vector:
            return reflectBasicType(pSlangType);
        case TypeReflection::Kind::None:
            return nullptr;
        case TypeReflection::Kind::GenericTypeParameter:
            // TODO: How to handle this type? Let it generate an error for now.
            throw ArgumentError("Unexpected Slang type");
        default:
            FALCOR_UNREACHABLE();
        }
        return nullptr;
    }

    static ParameterCategory getParameterCategory(TypeLayoutReflection* pTypeLayout)
    {
        ParameterCategory category = pTypeLayout->getParameterCategory();
        if (category == ParameterCategory::Mixed)
        {
            switch (pTypeLayout->getKind())
            {
            case TypeReflection::Kind::ConstantBuffer:
            case TypeReflection::Kind::ParameterBlock:
            case TypeReflection::Kind::None:
                category = ParameterCategory::ConstantBuffer;
                break;
            }
        }
        return category;
    }

    static ParameterCategory getParameterCategory(VariableLayoutReflection* pVarLayout)
    {
        return getParameterCategory(pVarLayout->getTypeLayout());
    }

    ReflectionVar::SharedPtr reflectVariable(
        VariableLayoutReflection*   pSlangLayout,
        ShaderVarOffset::RangeIndex rangeIndex,
        ParameterBlockReflection*   pBlock,
        ReflectionPath*             pPath,
        ProgramVersion const*       pProgramVersion)
    {
        FALCOR_ASSERT(pPath);
        std::string name(pSlangLayout->getName());

        ReflectionType::SharedPtr pType = reflectType(
            pSlangLayout->getTypeLayout(),
            pBlock,
            pPath,
            pProgramVersion);
        auto byteOffset = (ShaderVarOffset::ByteOffset) pSlangLayout->getOffset(SLANG_PARAMETER_CATEGORY_UNIFORM);

        ReflectionVar::SharedPtr pVar = ReflectionVar::create(
            name,
            pType,
            ShaderVarOffset(
                UniformShaderVarOffset(byteOffset),
                ResourceShaderVarOffset(rangeIndex, 0)));

        return pVar;
    }

    ReflectionVar::SharedPtr reflectTopLevelVariable(
        VariableLayoutReflection*   pSlangLayout,
        ShaderVarOffset::RangeIndex rangeIndex,
        ParameterBlockReflection*   pBlock,
        ProgramVersion const*       pProgramVersion)
    {
        ExtendedReflectionPath path(nullptr, pSlangLayout);

        return reflectVariable(pSlangLayout, rangeIndex, pBlock, &path, pProgramVersion);
    }

    static void storeShaderVariable(const ReflectionPath& path, SlangParameterCategory category, const std::string& name, ProgramReflection::VariableMap& varMap, ProgramReflection::VariableMap* pVarMapBySemantic, uint32_t count, uint32_t stride)
    {
        auto pVar = path.pPrimary->pVar;

        ProgramReflection::ShaderVariable var;
        const auto& pTypeLayout = pVar->getTypeLayout();
        var.type = getVariableType(pTypeLayout->getScalarType(), pTypeLayout->getRowCount(), pTypeLayout->getColumnCount());

        uint32_t baseIndex = (uint32_t)getRegisterIndexFromPath(path.pPrimary, category);
        for(uint32_t i = 0 ; i < std::max(count, 1u) ; i++)
        {
            var.bindLocation = baseIndex + (i*stride);
            var.semanticName = pVar->getSemanticName();
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
        auto pVar = path.pPrimary->pVar;
        TypeLayoutReflection* pTypeLayout = pVar->getTypeLayout();
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
            FALCOR_ASSERT((arrayKind == TypeReflection::Kind::Matrix) || (arrayKind == TypeReflection::Kind::Vector) || (arrayKind == TypeReflection::Kind::Scalar));
            uint32_t arraySize = (uint32_t)pTypeLayout->getTotalArrayElementCount();
            uint32_t arrayStride = (uint32_t)pTypeLayout->getElementStride(SLANG_PARAMETER_CATEGORY_UNIFORM);
            storeShaderVariable(path, category, name, varMap, pVarMapBySemantic, arraySize, arrayStride);
        }
        else if (kind == TypeReflection::Kind::Struct)
        {
            for (uint32_t f = 0; f < pTypeLayout->getFieldCount(); f++)
            {
                auto pField = pTypeLayout->getFieldByIndex(f);
                ExtendedReflectionPath newPath(&path, pField);
                std::string memberName = name + '.' + pField->getName();
                reflectVaryingParameter(newPath, memberName, category, varMap, pVarMapBySemantic);
            }
        }
        else
        {
            FALCOR_UNREACHABLE();
        }
    }

    static void reflectShaderIO(slang::EntryPointReflection* pEntryPoint, SlangParameterCategory category, ProgramReflection::VariableMap& varMap, ProgramReflection::VariableMap* pVarMapBySemantic = nullptr)
    {
        uint32_t entryPointParamCount = pEntryPoint->getParameterCount();
        for (uint32_t pp = 0; pp < entryPointParamCount; ++pp)
        {
            auto pVar = pEntryPoint->getParameterByIndex(pp);

            ExtendedReflectionPath path(nullptr, pVar);
            reflectVaryingParameter(path, pVar->getName(), category, varMap, pVarMapBySemantic);
        }
    }

    ProgramReflection::SharedPtr ProgramReflection::create(
        ProgramVersion const* pProgramVersion,
        slang::ShaderReflection* pSlangReflector,
        std::vector<slang::EntryPointLayout*> const& pSlangEntryPointReflectors,
        std::string& log)
    {
        return SharedPtr(new ProgramReflection(pProgramVersion, pSlangReflector, pSlangEntryPointReflectors, log));
    }

    ProgramReflection::SharedPtr ProgramReflection::createEmpty()
    {
        return SharedPtr(new ProgramReflection(nullptr));
    }

    std::shared_ptr<const ProgramVersion> ParameterBlockReflection::getProgramVersion() const
    {
        return mpProgramVersion->shared_from_this();
    }

    void ProgramReflection::finalize()
    {
        mpDefaultBlock->finalize();
    }

    ProgramReflection::ProgramReflection(ProgramVersion const* pProgramVersion)
        : mpProgramVersion(pProgramVersion)
    {
        ReflectionStructType::SharedPtr pGlobalStruct = ReflectionStructType::create(0, "", nullptr);

        ParameterBlockReflection::SharedPtr pDefaultBlock = ParameterBlockReflection::createEmpty(pProgramVersion);
        pDefaultBlock->setElementType(pGlobalStruct);
        setDefaultParameterBlock(pDefaultBlock);
    }

    EntryPointGroupReflection::EntryPointGroupReflection(
        ProgramVersion const* pProgramVersion)
        : ParameterBlockReflection(pProgramVersion)
    {}

    static bool isVaryingParameter(slang::VariableLayoutReflection* pSlangParam)
    {
        // TODO: It is unfortunate that Falcor has to maintain this logic,
        // since there is nearly identical logic already in Slang.
        //
        // The basic problem is that we want to know whether a parameter
        // is logically "uniform" or logically "varying."
        //
        // In the common cases, we can tell by looking at the kind(s)
        // of resources the parameter consumes; if it uses any
        // kinds of resources that only make sense for varying
        // parameters, then it is varying.
        //
        unsigned int categoryCount = pSlangParam->getCategoryCount();
        for( unsigned int ii = 0; ii < categoryCount; ++ii )
        {
            switch(pSlangParam->getCategoryByIndex(ii))
            {
            // Varying cross-stage input/output obviously marks
            // a varying parameter, as do the special categories
            // of input used for ray-tracing shaders.
            //
            case slang::ParameterCategory::VaryingInput:
            case slang::ParameterCategory::VaryingOutput:
            case slang::ParameterCategory::RayPayload:
            case slang::ParameterCategory::HitAttributes:
                return true;

            // Everything else indicates a uniform parameter.
            //
            default:
                return false;
            }
        }

        // If we get to the end of the loop above, then it
        // means that there must have been *zero* categories
        // of resources consumed by the parameter.
        //
        // There are two cases where that could have happened:
        //
        // 1. A parameter of a zero-size type (an empty `struct`
        //   or a `void` parameter). In this case uniform-vs-varying
        //   is a meaningless distinction.
        //
        // 2. A varying "system value" parameter, which doesn't
        //   consume any application-bindable resources.
        //
        // Because case (1) is unimportant, we choose the default
        // behavior based on (2). If a parameter doesn't appear
        // to consume any resources, we assume it is varying.
        //
        return true;
    }

    static uint32_t getUniformParameterCount(
        slang::EntryPointReflection* pSlangEntryPoint)
    {
        uint32_t entryPointParamCount = pSlangEntryPoint->getParameterCount();
        uint32_t uniformParamCount = 0;
        for (uint32_t pp = 0; pp < entryPointParamCount; ++pp)
        {
            auto pVar = pSlangEntryPoint->getParameterByIndex(pp);

            if(isVaryingParameter(pVar))
                continue;

            uniformParamCount++;
        }
        return uniformParamCount;
    }

    EntryPointGroupReflection::SharedPtr EntryPointGroupReflection::create(
        ProgramVersion const*   pProgramVersion,
        uint32_t                groupIndex,
        std::vector<slang::EntryPointLayout*> const& pSlangEntryPointReflectors)
    {
        // We are going to expect/require that all the entry points have the
        // same uniform parameters - or at least that for all the uniform
        // parameters they declare there is a match.
        //
        // We will start by finding out which of the entry points has the
        // most uniform parameters.
        //
        auto pProgram = pProgramVersion->getProgram();
        uint32_t entryPointCount = pProgram->getGroupEntryPointCount(groupIndex);
        FALCOR_ASSERT(entryPointCount != 0);

        slang::EntryPointLayout* pBestEntryPoint = pSlangEntryPointReflectors[pProgram->getGroupEntryPointIndex(groupIndex, 0)];
        for (uint32_t ee = 0; ee < entryPointCount; ++ee)
        {
            slang::EntryPointReflection* pSlangEntryPoint = pSlangEntryPointReflectors[pProgram->getGroupEntryPointIndex(groupIndex, ee)];

            if (getUniformParameterCount(pSlangEntryPoint) > getUniformParameterCount(pBestEntryPoint))
            {
                pBestEntryPoint = pSlangEntryPoint;
            }
        }

        auto pGroup = SharedPtr(new EntryPointGroupReflection(pProgramVersion));

        // The layout for an entry point either represents a Slang `struct` type
        // for the entry-point parameters, or it represents a Slang `ConstantBuffer<X>`
        // where `X` is the `struct` type for the entry-point parameters.
        //
        auto pSlangEntryPointVarLayout = pBestEntryPoint->getVarLayout();
        auto pSlangEntryPointTypeLayout = pBestEntryPoint->getTypeLayout();
        ExtendedReflectionPath entryPointPath(nullptr, pSlangEntryPointVarLayout);

        // We need to detect the latter case, because it means that a "default" constant
        // buffer has been allocated for the parameters.
        //
        // Note: in recent Slang releases, we could just check if the "kind" of the
        // `pSlangEntryPointTypeLayout` is `ConstantBuffer`, but in some existing
        // releases that won't work (due to Slang bugs).
        //
        // Instead, we check whether the type layout has a "container" layout
        // associated with it, which should only happen for `ConstantBuffer<...>`
        // or `ParameterBlock<...>` types.
        //
        bool hasDefaultConstantBuffer = false;
        if (pSlangEntryPointTypeLayout->getContainerVarLayout() != nullptr)
        {
            hasDefaultConstantBuffer = true;
        }

        // In the case where theere is no default constant buffer, the variable
        // and type layouts for the entry point itself are what we want to reflect
        // as the "element type."
        //
        auto pSlangElementVarLayout = pSlangEntryPointVarLayout;
        auto pSlangElementTypeLayout = pSlangEntryPointTypeLayout;
        ReflectionPath* pElementPath = &entryPointPath;

        // If there is a default constant buffer, though, we need to drill down
        // to its element type to get the information we want.
        //
        if (hasDefaultConstantBuffer)
        {
            pSlangElementVarLayout = pSlangEntryPointTypeLayout->getElementVarLayout();
            pSlangElementTypeLayout = pSlangElementVarLayout->getTypeLayout();
        }
        ExtendedReflectionPath elementPath(&entryPointPath, pSlangElementVarLayout);
        if (hasDefaultConstantBuffer)
        {
            pElementPath = &elementPath;
        }

        ReflectionStructType::BuildState elementTypeBuildState;

        std::string name;
        if(entryPointCount == 1)
            name = pBestEntryPoint->getName();

        auto pElementType = ReflectionStructType::create(pSlangElementTypeLayout->getSize(), name, pSlangElementTypeLayout);
        pGroup->setElementType(pElementType);

        uint32_t entryPointParamCount = pBestEntryPoint->getParameterCount();
        for (uint32_t pp = 0; pp < entryPointParamCount; ++pp)
        {
            auto pSlangParam = pBestEntryPoint->getParameterByIndex(pp);

            // Note: Due to some quirks on the Slang reflection information,
            // we do not currently need to append the parameter to the
            // reflection path(s) we computed outside the loop.
            //
            // TODO: We probably need to revisit this choice if/when we
            // want to reflect all the parameters using the existing
            // logic that handles `struct` types.
            //
            ExtendedReflectionPath path(nullptr, pSlangParam);

            if(isVaryingParameter(pSlangParam))
                continue;

            auto pParam = reflectVariable(
                pSlangParam,
                pElementType->getResourceRangeCount(),
                pGroup.get(),
                &path,
                pProgramVersion);

            pElementType->addMember(pParam, elementTypeBuildState);
        }

        // If the entry point had a default constant buffer allocated
        // for it, we need to extract its binding information. The
        // logic here is nearly identical to the logic for an explicit
        // constant buffer in user code. The main difference is that
        // entry-point `uniform` parameters should default to being
        // treated as a root constant buffer.
        //
        if (hasDefaultConstantBuffer)
        {
            extractDefaultConstantBufferBinding(pSlangEntryPointTypeLayout, &entryPointPath, pGroup.get(), /*shouldUseRootConstants:*/true);
        }

        pGroup->finalize();

        // TODO(tfoley): There is no guarantee that all the other entry
        // points in the group agree with the one we chose as the "best."
        // We should ideally iterate over the parameters of the other
        // entry points and perform a check to see if they match what
        // we extracted from the best entry point.
        //
        // TODO: alternatively, if Falcor could identify the entry point
        // groups more explicitly to Slang, we could skip the need for
        // this kind of matching/validation in the application layer.

        return pGroup;
    }

    static ShaderType getShaderTypeFromSlangStage(SlangStage stage)
    {
        switch( stage )
        {
#define CASE(SLANG_NAME, FALCOR_NAME) case SLANG_STAGE_##SLANG_NAME: return ShaderType::FALCOR_NAME

        CASE(VERTEX,    Vertex);
        CASE(HULL,      Hull);
        CASE(DOMAIN,    Domain);
        CASE(GEOMETRY,  Geometry);
        CASE(PIXEL,     Pixel);

        CASE(COMPUTE,   Compute);

#ifdef FALCOR_D3D12
        CASE(RAY_GENERATION,    RayGeneration);
        CASE(INTERSECTION,      Intersection);
        CASE(ANY_HIT,           AnyHit);
        CASE(CLOSEST_HIT,       ClosestHit);
        CASE(MISS,              Miss);
        CASE(CALLABLE,          Callable);
#endif
#undef CASE

        default:
            FALCOR_UNREACHABLE();
            return ShaderType::Count;
        }
    }

    ProgramReflection::ProgramReflection(
        ProgramVersion const* pProgramVersion,
        slang::ShaderReflection* pSlangReflector,
        std::vector<slang::EntryPointLayout*> const& pSlangEntryPointReflectors,
        std::string& log)
        : mpProgramVersion(pProgramVersion)
        , mpSlangReflector(pSlangReflector)
    {
        // For Falcor's purposes, the global scope of a program can be treated
        // much like a user-defined `struct` type, where the fields are the
        // global shader parameters.
        //
        // Slang provides two ways to iterate over the parameters of a program:
        //
        // 1. We can directly query `getParameterCount()` and then `getParameterByIndex()`,
        //    to enumerate all the global shader parameters.
        //
        // 2. We can query `getGlobalParamsTypeLayout()` which returns a type layout
        //    that represents all of the global-scope parameters bundled together.
        //
        // Our code will mostly use option (1), but we will do a little bit of
        // option (2) to be able to get the total size of the global parameters,
        // for cases where a default constant buffer is needed for the global
        // scope.
        //
        auto pSlangGlobalParamsTypeLayout = pSlangReflector->getGlobalParamsTypeLayout();

        // The Slang type layout for the global scope either directly represents the
        // parameters as a struct type `G`, or it represents those parameters wrapped
        // up into a constant buffer like `ConstantBuffer<G>`. If we are in the latter
        // case, then we want to get the element type (`G`) from the constant buffer
        // type layout.
        //
        if (auto pElementTypeLayout = pSlangGlobalParamsTypeLayout->getElementTypeLayout())
            pSlangGlobalParamsTypeLayout = pElementTypeLayout;

        // Once we have the Slang type layout for the `struct` of global parameters,
        // we can easily query its size in bytes.
        //
        size_t slangGlobalParamsSize = pSlangGlobalParamsTypeLayout->getSize(slang::ParameterCategory::Uniform);

        ReflectionStructType::SharedPtr pGlobalStruct = ReflectionStructType::create(slangGlobalParamsSize, "", nullptr);
        ParameterBlockReflection::SharedPtr pDefaultBlock = ParameterBlockReflection::createEmpty(pProgramVersion);
        pDefaultBlock->setElementType(pGlobalStruct);

        ReflectionStructType::BuildState buildState;
        for (uint32_t i = 0; i < pSlangReflector->getParameterCount(); i++)
        {
            VariableLayoutReflection* pSlangLayout = pSlangReflector->getParameterByIndex(i);

            ReflectionVar::SharedPtr pVar = reflectTopLevelVariable(
                pSlangLayout,
                pGlobalStruct->getResourceRangeCount(),
                pDefaultBlock.get(),
                pProgramVersion);
            if(pVar) pGlobalStruct->addMember(pVar, buildState);
        }

        pDefaultBlock->finalize();
        setDefaultParameterBlock(pDefaultBlock);

        auto pProgram = pProgramVersion->getProgram();

        auto entryPointGroupCount = pProgram->getEntryPointGroupCount();

        for(uint32_t gg = 0; gg < entryPointGroupCount; ++gg)
        {
            EntryPointGroupReflection::SharedPtr pEntryPointGroup = EntryPointGroupReflection::create(
                pProgramVersion,
                gg,
                pSlangEntryPointReflectors);
            mEntryPointGroups.push_back(pEntryPointGroup);
        }

        // Reflect per-stage parameters
        for(auto pSlangEntryPoint : pSlangEntryPointReflectors)
        {
            switch (pSlangEntryPoint->getStage())
            {
            case SLANG_STAGE_COMPUTE:
            {
                SlangUInt sizeAlongAxis[3];
                pSlangEntryPoint->getComputeThreadGroupSize(3, &sizeAlongAxis[0]);
                mThreadGroupSize.x = (uint32_t)sizeAlongAxis[0];
                mThreadGroupSize.y = (uint32_t)sizeAlongAxis[1];
                mThreadGroupSize.z = (uint32_t)sizeAlongAxis[2];
            }
            break;
            case SLANG_STAGE_FRAGMENT:
                reflectShaderIO(pSlangEntryPoint, SLANG_PARAMETER_CATEGORY_FRAGMENT_OUTPUT, mPsOut);
                break;
            case SLANG_STAGE_VERTEX:
                reflectShaderIO(pSlangEntryPoint, SLANG_PARAMETER_CATEGORY_VERTEX_INPUT, mVertAttr, &mVertAttrBySemantic);
                break;
            default:
                break;
            }
        }

        // Get hashed strings
        uint32_t hashedStringCount = (uint32_t)pSlangReflector->getHashedStringCount();
        mHashedStrings.reserve(hashedStringCount);
        for (uint32_t i = 0; i < hashedStringCount; ++i)
        {
            size_t stringSize;
            const char *stringData = pSlangReflector->getHashedString(i, &stringSize);
            uint32_t stringHash = spComputeStringHash(stringData, stringSize);
            mHashedStrings.push_back(HashedString{ stringHash, std::string(stringData, stringData + stringSize) });
        }
    }

    void ProgramReflection::setDefaultParameterBlock(const ParameterBlockReflection::SharedPtr& pBlock)
    {
        mpDefaultBlock = pBlock;
    }

    int32_t ReflectionStructType::addMemberIgnoringNameConflicts(
        const std::shared_ptr<const ReflectionVar>& pVar,
        ReflectionStructType::BuildState&           ioBuildState)
    {
        auto memberIndex = int32_t(mMembers.size());
        mMembers.push_back(pVar);

        auto pFieldType = pVar->getType();
        auto fieldRangeCount = pFieldType->getResourceRangeCount();
        for (uint32_t rr = 0; rr < fieldRangeCount; ++rr)
        {
            auto fieldRange = pFieldType->getResourceRange(rr);

            switch (fieldRange.descriptorType)
            {
            case ShaderResourceType::Cbv:
                fieldRange.baseIndex = ioBuildState.cbCount;
                ioBuildState.cbCount += fieldRange.count;
                break;

            case ShaderResourceType::TextureSrv:
            case ShaderResourceType::RawBufferSrv:
            case ShaderResourceType::TypedBufferSrv:
            case ShaderResourceType::StructuredBufferSrv:
            case ShaderResourceType::AccelerationStructureSrv:
                fieldRange.baseIndex = ioBuildState.srvCount;
                ioBuildState.srvCount += fieldRange.count;
                break;

            case ShaderResourceType::TextureUav:
            case ShaderResourceType::RawBufferUav:
            case ShaderResourceType::TypedBufferUav:
            case ShaderResourceType::StructuredBufferUav:
                fieldRange.baseIndex = ioBuildState.uavCount;
                ioBuildState.uavCount += fieldRange.count;
                break;

            case ShaderResourceType::Sampler:
                fieldRange.baseIndex = ioBuildState.samplerCount;
                ioBuildState.samplerCount += fieldRange.count;
                break;

            case ShaderResourceType::Dsv:
            case ShaderResourceType::Rtv:
                break;

            default:
                FALCOR_UNREACHABLE();
                break;
            }

            mResourceRanges.push_back(fieldRange);
        }

        return memberIndex;
    }

    int32_t ReflectionStructType::addMember(
        const std::shared_ptr<const ReflectionVar>& pVar,
        ReflectionStructType::BuildState&           ioBuildState)
    {
        if (mNameToIndex.find(pVar->getName()) != mNameToIndex.end())
        {
            int32_t index = mNameToIndex[pVar->getName()];
            if (*pVar != *mMembers[index])
            {
                throw RuntimeError("Mismatch in variable declarations between different shader stages. Variable name is '{}', struct name is '{}'.", pVar->getName(), mName);
            }
            return -1;
        }
        auto memberIndex = addMemberIgnoringNameConflicts(pVar, ioBuildState);
        mNameToIndex[pVar->getName()] = memberIndex;
        return memberIndex;
    }


    ReflectionVar::SharedPtr ReflectionVar::create(
        const std::string& name,
        const ReflectionType::SharedConstPtr& pType,
        ShaderVarOffset const& bindLocation)
    {
        return SharedPtr(new ReflectionVar(
            name,
            pType,
            bindLocation));
    }

    ReflectionVar::ReflectionVar(
        const std::string& name,
        const ReflectionType::SharedConstPtr& pType,
        ShaderVarOffset const& bindLocation)
        : mName(name)
        , mpType(pType)
        , mBindLocation(bindLocation)
    {}

    //

    ParameterBlockReflection::ParameterBlockReflection(
        ProgramVersion const* pProgramVersion)
        : mpProgramVersion(pProgramVersion)
    {
    }

    ParameterBlockReflection::SharedPtr ParameterBlockReflection::createEmpty(
        ProgramVersion const* pProgramVersion)
    {
        return SharedPtr(new ParameterBlockReflection(pProgramVersion));
    }

    void ParameterBlockReflection::setElementType(
        ReflectionType::SharedConstPtr const& pElementType)
    {
        FALCOR_ASSERT(!mpElementType);
        mpElementType = pElementType;
    }

    ParameterBlockReflection::SharedPtr ParameterBlockReflection::create(
        ProgramVersion const* pProgramVersion,
        ReflectionType::SharedConstPtr const& pElementType)
    {
        auto pResult = createEmpty(pProgramVersion);
        pResult->setElementType(pElementType);

#if FALCOR_HAS_D3D12
        ReflectionStructType::BuildState counters;
#endif

        auto rangeCount = pElementType->getResourceRangeCount();
        for (uint32_t rangeIndex = 0; rangeIndex < rangeCount; ++rangeIndex)
        {
            auto const& rangeInfo = pElementType->getResourceRange(rangeIndex);

            ResourceRangeBindingInfo bindingInfo;

            uint32_t regIndex = 0;
            uint32_t regSpace = 0;

#if FALCOR_HAS_D3D12
            switch (rangeInfo.descriptorType)
            {
            case ShaderResourceType::Cbv:
                regIndex += counters.cbCount;
                counters.cbCount += rangeInfo.count;
                break;

            case ShaderResourceType::TextureSrv:
            case ShaderResourceType::RawBufferSrv:
            case ShaderResourceType::TypedBufferSrv:
            case ShaderResourceType::StructuredBufferSrv:
            case ShaderResourceType::AccelerationStructureSrv:
                regIndex += counters.srvCount;
                counters.srvCount += rangeInfo.count;
                break;

            case ShaderResourceType::TextureUav:
            case ShaderResourceType::RawBufferUav:
            case ShaderResourceType::TypedBufferUav:
            case ShaderResourceType::StructuredBufferUav:
                regIndex += counters.uavCount;
                counters.uavCount += rangeInfo.count;
                break;

            case ShaderResourceType::Sampler:
                regIndex += counters.samplerCount;
                counters.samplerCount += rangeInfo.count;
                break;

            case ShaderResourceType::Dsv:
            case ShaderResourceType::Rtv:
                break;

            default:
                FALCOR_UNREACHABLE();
                break;
            }
#endif

            bindingInfo.regIndex = regIndex;
            bindingInfo.regSpace = regSpace;

            pResult->addResourceRange(bindingInfo);
        }

        pResult->finalize();
        return pResult;
    }

    ParameterBlockReflection::SharedPtr ParameterBlockReflection::create(
        ProgramVersion const* pProgramVersion,
        slang::TypeLayoutReflection* pSlangElementType)
    {
        auto pResult = ParameterBlockReflection::createEmpty(pProgramVersion);

        ReflectionPath path;

        auto pElementType = reflectType(
            pSlangElementType,
            pResult.get(),
            &path,
            pProgramVersion);
        pResult->setElementType(pElementType);

        pResult->finalize();

        return pResult;
    }


    static ShaderResourceType getShaderResourceType(
        const ReflectionResourceType* pType)
    {
        auto shaderAccess = pType->getShaderAccess();
        switch (pType->getType())
        {
        case ReflectionResourceType::Type::ConstantBuffer:
            return ShaderResourceType::Cbv;
            break;
        case ReflectionResourceType::Type::Texture:
            return shaderAccess == ReflectionResourceType::ShaderAccess::Read
                ? ShaderResourceType::TextureSrv
                : ShaderResourceType::TextureUav;
            break;
        case ReflectionResourceType::Type::RawBuffer:
            return shaderAccess == ReflectionResourceType::ShaderAccess::Read
                ? ShaderResourceType::RawBufferSrv
                : ShaderResourceType::RawBufferUav;
            break;
        case ReflectionResourceType::Type::StructuredBuffer:
            return shaderAccess == ReflectionResourceType::ShaderAccess::Read
                ? ShaderResourceType::StructuredBufferSrv
                : ShaderResourceType::StructuredBufferUav;
            break;
        case ReflectionResourceType::Type::TypedBuffer:
            return shaderAccess == ReflectionResourceType::ShaderAccess::Read
                ? ShaderResourceType::TypedBufferSrv
                : ShaderResourceType::TypedBufferUav;
            break;
        case ReflectionResourceType::Type::AccelerationStructure:
            FALCOR_ASSERT(shaderAccess == ReflectionResourceType::ShaderAccess::Read);
            return ShaderResourceType::AccelerationStructureSrv;
            break;
        case ReflectionResourceType::Type::Sampler:
            return ShaderResourceType::Sampler;
            break;
        default:
            FALCOR_UNREACHABLE();
            return ShaderResourceType::Count;
        }
    }

    const ReflectionVar::SharedConstPtr ParameterBlockReflection::getResource(const std::string& name) const
    {
        return getElementType()->findMember(name);
    }

    void ParameterBlockReflection::addResourceRange(
        ResourceRangeBindingInfo const& bindingInfo)
    {
        mResourceRanges.push_back(bindingInfo);
    }

#if FALCOR_HAS_D3D12
    struct ParameterBlockReflectionFinalizer
    {
        struct SetIndex
        {
            SetIndex(
                uint32_t                regSpace,
                ShaderResourceType      descriptorType)
                : regSpace(regSpace)
                , isSampler(descriptorType == ShaderResourceType::Sampler)
            {}
            bool isSampler = false;
            uint32_t regSpace;
            bool operator<(const SetIndex& other) const
            {
                return (regSpace == other.regSpace) ? isSampler < other.isSampler : regSpace < other.regSpace;
            }
        };

        std::map<SetIndex, uint32_t> newSetIndices;
        ParameterBlockReflection* pPrimaryReflector;

        uint32_t computeDescriptorSetIndex(
            uint32_t                regSpace,
            ShaderResourceType      descriptorType)
        {
            SetIndex origIndex(regSpace, descriptorType);
            uint32_t setIndex;
            if (newSetIndices.find(origIndex) == newSetIndices.end())
            {
                // New set
                setIndex = (uint32_t) pPrimaryReflector->mDescriptorSets.size();
                newSetIndices[origIndex] = setIndex;
                pPrimaryReflector->mDescriptorSets.push_back({});
            }
            else
            {
                setIndex = newSetIndices[origIndex];
            }
            return setIndex;
        }

        uint32_t computeDescriptorSetIndex(
            const ReflectionType::ResourceRange&                    range,
            const ParameterBlockReflection::ResourceRangeBindingInfo& bindingInfo)
        {
            return computeDescriptorSetIndex(bindingInfo.regSpace, range.descriptorType);
        };

        void addSubObjectResources(
            uint32_t subObjectResourceRangeIndex,
            ParameterBlockReflection const* pSubObjectReflector,
            bool shouldSkipDefaultConstantBufferRange)
        {
            // TODO: this function needs to accept a multiplier that gets
            // applied to all of the counts on the way down, to deal with
            // arrays of constant buffers.

            FALCOR_ASSERT(pSubObjectReflector);
            auto subSetCount = pSubObjectReflector->getD3D12DescriptorSetCount();
            for (uint32_t subSetIndex = 0; subSetIndex < subSetCount; ++subSetIndex)
            {
                auto& subSet = pSubObjectReflector->getD3D12DescriptorSetInfo(subSetIndex);

                FALCOR_ASSERT(subSet.layout.getRangeCount() != 0);
                auto subRange = subSet.layout.getRange(0);

                auto setIndex = computeDescriptorSetIndex(subRange.regSpace, subRange.type);
                auto& setInfo = pPrimaryReflector->mDescriptorSets[setIndex];

                ParameterBlockReflection::DescriptorSetInfo::SubObjectInfo subObjectInfo;
                subObjectInfo.resourceRangeIndexOfSubObject = subObjectResourceRangeIndex;
                subObjectInfo.setIndexInSubObject = subSetIndex;
                setInfo.subObjects.push_back(subObjectInfo);

                auto subLayoutRangeCount = subSet.layout.getRangeCount();
                for (size_t r = 0; r < subLayoutRangeCount; ++r)
                {
                    if( shouldSkipDefaultConstantBufferRange
                        && subSetIndex == 0
                        && r == 0 )
                    {
                        // Skip the range corresponding to the default constant buffer.
                        continue;
                    }

                    auto subRange = subSet.layout.getRange(r);
                    setInfo.layout.addRange(
                        subRange.type,
                        subRange.baseRegIndex,
                        subRange.descCount,
                        subRange.regSpace);
                }
            }
        }

        void finalize(ParameterBlockReflection* pReflector)
        {
            pPrimaryReflector = pReflector;

            if (pReflector->hasDefaultConstantBuffer())
            {
                auto descriptorType = ShaderResourceType::Cbv;
                auto& bindingInfo = pReflector->mDefaultConstantBufferBindingInfo;

                if(!bindingInfo.useRootConstants)
                {
                    auto setIndex = computeDescriptorSetIndex(bindingInfo.regSpace, descriptorType);

                    bindingInfo.descriptorSetIndex = setIndex;
                    auto& setInfo = pReflector->mDescriptorSets[setIndex];

                    setInfo.layout.addRange(
                        descriptorType,
                        bindingInfo.regIndex,
                        1,
                        bindingInfo.regSpace);
                }
            }

            // Iterate over descriptors
            auto resourceRangeCount = pReflector->mResourceRanges.size();
            for (uint32_t rangeIndex = 0; rangeIndex < resourceRangeCount; ++rangeIndex)
            {
                const auto& range = pReflector->getElementType()->getResourceRange(rangeIndex);
                auto& rangeBindingInfo = pReflector->mResourceRanges[rangeIndex];

                switch (rangeBindingInfo.flavor)
                {
                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Simple:
                {
                    auto setIndex = computeDescriptorSetIndex(range, rangeBindingInfo);

                    rangeBindingInfo.descriptorSetIndex = setIndex;
                    auto& setInfo = pReflector->mDescriptorSets[setIndex];

                    setInfo.layout.addRange(
                        range.descriptorType,
                        rangeBindingInfo.regIndex,
                        range.count,
                        rangeBindingInfo.regSpace);

                    setInfo.resourceRangeIndices.push_back(rangeIndex);
                }
                break;

                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::RootDescriptor:
                    if (range.count > 1)
                    {
                        throw RuntimeError("Root descriptor at register index {} in space {} is illegal. Root descriptors cannot be arrays.",  rangeBindingInfo.regIndex, rangeBindingInfo.regSpace);
                    }
                    pReflector->mRootDescriptorRangeIndices.push_back(rangeIndex);
                    break;

                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ConstantBuffer:
                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ParameterBlock:
                case ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface:
                    break;
                default:
                    FALCOR_UNREACHABLE();
                }
            }

            // Iterate over constant buffers
            for (uint32_t rangeIndex = 0; rangeIndex < resourceRangeCount; ++rangeIndex)
            {
                auto& rangeBindingInfo = pReflector->mResourceRanges[rangeIndex];

                if (rangeBindingInfo.flavor != ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ConstantBuffer)
                    continue;

                addSubObjectResources(rangeIndex, rangeBindingInfo.pSubObjectReflector.get(), false);
            }

            // Iterate over parameter blocks
            for (uint32_t rangeIndex = 0; rangeIndex < resourceRangeCount; ++rangeIndex)
            {
                auto& rangeBindingInfo = pReflector->mResourceRanges[rangeIndex];
                if (rangeBindingInfo.flavor != ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::ParameterBlock)
                    continue;

                pReflector->mParameterBlockSubObjectRangeIndices.push_back(rangeIndex);
            }

            // Iterate over interfaces
            for (uint32_t rangeIndex = 0; rangeIndex < resourceRangeCount; ++rangeIndex)
            {
                auto& rangeBindingInfo = pReflector->mResourceRanges[rangeIndex];

                if(rangeBindingInfo.flavor != ParameterBlockReflection::ResourceRangeBindingInfo::Flavor::Interface)
                    continue;

                // TODO(tfoley): need to figure out what exactly is appropriate here.
                if( auto pSubObjectReflector = rangeBindingInfo.pSubObjectReflector )
                {
                    addSubObjectResources(rangeIndex, pSubObjectReflector.get(), pSubObjectReflector->hasDefaultConstantBuffer());
                }
            }

            // TODO: Do we need to handle interface sub-object slots here?
        }
    };
#endif // FALCOR_HAS_D3D12
    bool ParameterBlockReflection::hasDefaultConstantBuffer() const
    {
        // A parameter block needs a "default" constant buffer whenever its element type requires it to store ordinary/uniform data
        return getElementType()->getByteSize() != 0;
    }

    void ParameterBlockReflection::setDefaultConstantBufferBindingInfo(DefaultConstantBufferBindingInfo const& info)
    {
        mDefaultConstantBufferBindingInfo = info;
    }

    ParameterBlockReflection::DefaultConstantBufferBindingInfo const& ParameterBlockReflection::getDefaultConstantBufferBindingInfo() const
    {
        return mDefaultConstantBufferBindingInfo;
    }

    void ParameterBlockReflection::finalize()
    {
        FALCOR_ASSERT(getElementType()->getResourceRangeCount() == mResourceRanges.size());
#if FALCOR_HAS_D3D12
        ParameterBlockReflectionFinalizer finalizer;
        finalizer.finalize(this);
#endif
    }

    std::shared_ptr<const ProgramVersion> ProgramReflection::getProgramVersion() const
    {
        return mpProgramVersion ? mpProgramVersion->shared_from_this() : ProgramVersion::SharedPtr();
    }

    ParameterBlockReflection::SharedConstPtr ProgramReflection::getParameterBlock(const std::string& name) const
    {
        if(name == "")
            return mpDefaultBlock;

        return mpDefaultBlock->getElementType()->findMember(name)->getType()->asResourceType()->getParameterBlockReflector()->shared_from_this();
    }

    TypedShaderVarOffset ReflectionType::findMemberByOffset(size_t offset) const
    {
        if (auto pStructType = asStructType())
        {
            return pStructType->findMemberByOffset(offset);
        }

        return TypedShaderVarOffset::kInvalid;
    }

    TypedShaderVarOffset ReflectionStructType::findMemberByOffset(size_t offset) const
    {
        for (auto pMember : mMembers)
        {
            auto memberOffset = pMember->getBindLocation();
            auto memberUniformOffset = memberOffset.getUniform().getByteOffset();
            auto pMemberType = pMember->getType();
            auto memberByteSize = pMember->getType()->getByteSize();

            if (offset >= memberUniformOffset)
            {
                if (offset < memberUniformOffset + memberByteSize)
                {
                    return TypedShaderVarOffset(
                        pMemberType.get(),
                        memberOffset);
                }
            }
        }

        return TypedShaderVarOffset::kInvalid;
    }

    ReflectionVar::SharedConstPtr ReflectionType::findMember(const std::string& name) const
    {
        if (auto pStructType = asStructType())
        {
            size_t fieldIndex = pStructType->getMemberIndex(name);
            if (fieldIndex == ReflectionStructType::kInvalidMemberIndex) return nullptr;

            return pStructType->getMember(fieldIndex);
        }

        return nullptr;
    }

    int32_t ReflectionStructType::getMemberIndex(const std::string& name) const
    {
        auto it = mNameToIndex.find(name);
        if (it == mNameToIndex.end()) return kInvalidMemberIndex;
        return it->second;
    }

    const ReflectionVar::SharedConstPtr& ReflectionStructType::getMember(const std::string& name) const
    {
        static ReflectionVar::SharedConstPtr pNull;
        auto index = getMemberIndex(name);
        return (index == kInvalidMemberIndex) ? pNull : getMember(index);
    }

    const ReflectionResourceType* ReflectionType::asResourceType() const
    {
        // In the past, Falcor relied on undefined behavior checking `this` for nullptr, returning nullptr if `this` was nullptr.
        FALCOR_ASSERT(this);
        return this->getKind() == ReflectionType::Kind::Resource ? static_cast<const ReflectionResourceType*>(this) : nullptr;
    }

    const ReflectionBasicType* ReflectionType::asBasicType() const
    {
        // In the past, Falcor relied on undefined behavior checking `this` for nullptr, returning nullptr if `this` was nullptr.
        FALCOR_ASSERT(this);
        return this->getKind() == ReflectionType::Kind::Basic ? static_cast<const ReflectionBasicType*>(this) : nullptr;
    }

    const ReflectionStructType* ReflectionType::asStructType() const
    {
        // In the past, Falcor relied on undefined behavior checking `this` for nullptr, returning nullptr if `this` was nullptr.
        FALCOR_ASSERT(this);
        return this->getKind() == ReflectionType::Kind::Struct ? static_cast<const ReflectionStructType*>(this) : nullptr;
    }

    const ReflectionArrayType* ReflectionType::asArrayType() const
    {
        // In the past, Falcor relied on undefined behavior checking `this` for nullptr, returning nullptr if `this` was nullptr.
        FALCOR_ASSERT(this);
        return this->getKind() == ReflectionType::Kind::Array ? static_cast<const ReflectionArrayType*>(this) : nullptr;
    }

    const ReflectionInterfaceType* ReflectionType::asInterfaceType() const
    {
        // In the past, Falcor relied on undefined behavior checking `this` for nullptr, returning nullptr if `this` was nullptr.
        FALCOR_ASSERT(this);
        return this->getKind() == ReflectionType::Kind::Interface ? static_cast<const ReflectionInterfaceType*>(this) : nullptr;
    }

    const ReflectionType* ReflectionType::unwrapArray() const
    {
        const ReflectionType* pType = this;
        while (auto pArrayType = pType->asArrayType())
        {
            pType = pArrayType->getElementType().get();
        }
        return pType;
    }

    uint32_t ReflectionType::getTotalArrayElementCount() const
    {
        uint32_t result = 1;

        const ReflectionType* pType = this;
        while (auto pArrayType = pType->asArrayType())
        {
            result *= pArrayType->getElementCount();
            pType = pArrayType->getElementType().get();
        }
        return result;
    }

    ReflectionArrayType::SharedPtr ReflectionArrayType::create(
        uint32_t arraySize,
        uint32_t arrayStride,
        const ReflectionType::SharedConstPtr& pType,
        ByteSize byteSize,
        slang::TypeLayoutReflection*    pSlangTypeLayout)
    {
        return SharedPtr(new ReflectionArrayType(arraySize, arrayStride, pType, byteSize, pSlangTypeLayout));
    }

    ReflectionArrayType::ReflectionArrayType(
        uint32_t elementCount,
        uint32_t elementByteStride,
        const ReflectionType::SharedConstPtr& pElementType,
        ByteSize byteSize,
        slang::TypeLayoutReflection*    pSlangTypeLayout)
        : ReflectionType(ReflectionType::Kind::Array, byteSize, pSlangTypeLayout)
        , mElementCount(elementCount)
        , mElementByteStride(elementByteStride)
        , mpElementType(pElementType)
    {
        auto rangeCount = pElementType->getResourceRangeCount();
        for (uint32_t rr = 0; rr < rangeCount; ++rr)
        {
            auto range = pElementType->getResourceRange(rr);
            range.count *= elementCount;
            range.baseIndex *= elementCount;
            mResourceRanges.push_back(range);
        }
    }

    ReflectionResourceType::SharedPtr ReflectionResourceType::create(
        Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess,
        slang::TypeLayoutReflection* pSlangTypeLayout)
    {
        return SharedPtr(new ReflectionResourceType(type, dims, structuredType, retType, shaderAccess, pSlangTypeLayout));
    }

    ReflectionResourceType::ReflectionResourceType(Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess,
        slang::TypeLayoutReflection* pSlangTypeLayout)
        : ReflectionType(ReflectionType::Kind::Resource, 0, pSlangTypeLayout)
        , mType(type)
        , mStructuredType(structuredType)
        , mReturnType(retType)
        , mShaderAccess(shaderAccess)
        , mDimensions(dims)
    {
        ResourceRange range;
        range.descriptorType = getShaderResourceType(this);
        range.count = 1;
        range.baseIndex = 0;

        mResourceRanges.push_back(range);
    }

    void ReflectionResourceType::setStructType(const ReflectionType::SharedConstPtr& pType)
    {
        mpStructType = pType;
    }

    ReflectionBasicType::SharedPtr ReflectionBasicType::create(Type type, bool isRowMajor, size_t size,
        slang::TypeLayoutReflection*    pSlangTypeLayout)
    {
        return SharedPtr(new ReflectionBasicType(type, isRowMajor, size, pSlangTypeLayout));
    }

    ReflectionBasicType::ReflectionBasicType(Type type, bool isRowMajor, size_t size,
        slang::TypeLayoutReflection*    pSlangTypeLayout)
        : ReflectionType(ReflectionType::Kind::Basic, size, pSlangTypeLayout)
        , mType(type)
        , mIsRowMajor(isRowMajor)
    {}

    ReflectionStructType::SharedPtr ReflectionStructType::create(size_t size, const std::string& name,
        slang::TypeLayoutReflection*    pSlangTypeLayout)
    {
        return SharedPtr(new ReflectionStructType(size, name, pSlangTypeLayout));
    }

    ReflectionStructType::ReflectionStructType(size_t size, const std::string& name,
        slang::TypeLayoutReflection*    pSlangTypeLayout)
        : ReflectionType(ReflectionType::Kind::Struct, size, pSlangTypeLayout)
        , mName(name) {}

    ParameterBlockReflection::BindLocation ParameterBlockReflection::getResourceBinding(const std::string& name) const
    {
        return getElementType()->getMemberOffset(name);
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
        if (mElementCount != other.mElementCount) return false;
        if (mElementByteStride != other.mElementByteStride) return false;
        if (*mpElementType != *other.mpElementType) return false;
        return true;
    }

    bool ReflectionStructType::operator==(const ReflectionStructType& other) const
    {
        // We only care about the struct layout. Checking the members should be enough
        if (mMembers.size() != other.mMembers.size()) return false;
        for (size_t i = 0; i < mMembers.size(); i++)
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
        if (mShaderAccess != other.mShaderAccess) return false;
        if (mType != other.mType) return false;
        bool hasStruct = (mpStructType != nullptr);
        bool otherHasStruct = (other.mpStructType != nullptr);
        if (hasStruct != otherHasStruct) return false;
        if (hasStruct && (*mpStructType != *other.mpStructType)) return false;

        return true;
    }

    bool ReflectionVar::operator==(const ReflectionVar& other) const
    {
        if (*mpType != *other.mpType) return false;
        if (mBindLocation != other.mBindLocation) return false;
        if (mName != other.mName) return false;

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

    ReflectionType::SharedPtr ProgramReflection::findType(const std::string& name) const
    {
        auto iter = mMapNameToType.find(name);
        if( iter != mMapNameToType.end() )
            return iter->second;

        auto pSlangType = mpSlangReflector->findTypeByName(name.c_str());
        if (!pSlangType) return nullptr;
        auto pSlangTypeLayout = mpSlangReflector->getTypeLayout(pSlangType);

        auto pFalcorTypeLayout = reflectType(pSlangTypeLayout, nullptr, nullptr, mpProgramVersion);
        if (!pFalcorTypeLayout) return nullptr;

        mMapNameToType.insert(std::make_pair(name, pFalcorTypeLayout));

        return pFalcorTypeLayout;
    }

    ReflectionVar::SharedConstPtr ProgramReflection::findMember(const std::string& name) const
    {
        return mpDefaultBlock->findMember(name);
    }

    ReflectionInterfaceType::SharedPtr ReflectionInterfaceType::create(
        slang::TypeLayoutReflection*    pSlangTypeLayout)
    {
        return SharedPtr(new ReflectionInterfaceType(pSlangTypeLayout));
    }

    ReflectionInterfaceType::ReflectionInterfaceType(
        slang::TypeLayoutReflection*    pSlangTypeLayout)
        : ReflectionType(Kind::Interface, 0, pSlangTypeLayout)
    {
        ResourceRange range;

        range.descriptorType = ShaderResourceType::Cbv;
        range.count = 1;
        range.baseIndex = 0;

        mResourceRanges.push_back(range);
    }

    bool ReflectionInterfaceType::operator==(const ReflectionInterfaceType& other) const
    {
        // TODO: properly double-check this
        return true;
    }

    bool ReflectionInterfaceType::operator==(const ReflectionType& other) const
    {
        auto pOtherInterface = other.asInterfaceType();
        if (!pOtherInterface) return false;
        return (*this == *pOtherInterface);
    }

}
