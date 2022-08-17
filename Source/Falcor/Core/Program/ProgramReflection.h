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
#pragma once
#include "Core/Assert.h"
#include "Core/Macros.h"
#include "Core/API/ShaderResourceType.h"
#include "Utils/Math/Vector.h"
#if FALCOR_HAS_D3D12
#include "Core/API/Shared/D3D12DescriptorSet.h"
#endif

#include <slang.h>

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace Falcor
{
    class ProgramVersion;
    class ReflectionVar;
    class ReflectionType;
    class ReflectionResourceType;
    class ReflectionBasicType;
    class ReflectionStructType;
    class ReflectionArrayType;
    class ReflectionInterfaceType;
    class ParameterBlockReflection;

    /** Represents the offset of a uniform shader variable relative to its enclosing type/buffer/block.

    A `UniformShaderVarOffset` is a simple wrapper around a byte offset for a uniform shader variable.
    It is used to make API signatures less ambiguous (e.g., about whether an integer represents an
    index, an offset, a count, etc.

    A `UniformShaderVarOffset` can also encode an invalid offset (represented as an all-ones bit pattern),
    to indicate that a particular uniform variable is not present.

    A `UniformShaderVarOffset` can be obtained from a reflection type or `ParameterBlock` using the
    `[]` subscript operator:

        UniformShaderVarOffset aOffset = pSomeType["a"]; // get offset of field `a` inside `pSomeType`
        UniformShaderVarOffset bOffset = pBlock["b"]; // get offset of parameter `b` inside parameter block
    */
    struct UniformShaderVarOffset
    {
        /** Type used to store the underlying byte offset.
        */
        typedef uint32_t ByteOffset;

        /** Construct from an explicit byte offset.
        */
        explicit UniformShaderVarOffset(size_t offset)
            : mByteOffset(ByteOffset(offset))
        {}

        /** Custom enumeration type used to represent a zero offset.

        Can be used to initialize a `UniformShaderVarOffset` when an explicit zero offset is desired:

            UniformShaderVarOffset myOffset = UniformShaderVarOffset::kZero;

        */
        enum Zero { kZero = 0 };

        /** Construct an explicit zero offset.
        */
        UniformShaderVarOffset(Zero)
            : mByteOffset(0)
        {}

        /** Custom enumeration type used to represent an invalid offset.

        Can be used to explicitly initialize a `UniformShaderVarOffset` to an invalid offset

            UniformShaderVarOffset myOffset = UniformShaderVarOffset::kInvalid;

        Note that the default constructor also creates an invalid offset, so this could instead
        be written more simply as:

            UniformShaderVarOffset myOffset;
        */
        enum Invalid { kInvalid = -1 };

        /** Default constructor: creates an invalid offset.
        */
        UniformShaderVarOffset(Invalid _ = kInvalid)
            : mByteOffset(ByteOffset(-1))
        {}

        /** Get the raw byte offset.
        */
        ByteOffset getByteOffset() const
        {
            return mByteOffset;
        }

        /** Check whether this offset is valid.

        An invalid offset has an all-ones bit pattern (`ByteOffset(-1)`).
        */
        bool isValid() const
        {
            return mByteOffset != ByteOffset(-1);
        }

        /** Compare this offset to another offset.
        */
        bool operator==(UniformShaderVarOffset const& other) const
        {
            return mByteOffset == other.mByteOffset;
        }

        /** Compare this offset to another offset.
        */
        bool operator!=(UniformShaderVarOffset const& other) const
        {
            return mByteOffset != other.mByteOffset;
        }

        /** Compare this offset to an invalid offset.

        This operator allows for checks like:

            if(myOffset == UniformShaderVarOffset::kInvalid) { ... }
        */
        bool operator==(Invalid _) const
        {
            return !isValid();
        }

        /** Compare this offset to an invalid offset.

        This operator allows for checks like:

            if(myOffset != UniformShaderVarOffset::kInvalid) { ... }
        */
        bool operator!=(Invalid _) const
        {
            return isValid();
        }

        /** Add an additional byte offset to this offset.

        If this offset is invalid, returns an invalid offset.
        */
        UniformShaderVarOffset operator+(size_t offset) const
        {
            if(!isValid()) return kInvalid;

            return UniformShaderVarOffset(mByteOffset + offset);
        }

        /** Add an additional byte offset to this offset.

        If either `this` or `other` is an invalid offset, returns an invalid offset.
        */
        UniformShaderVarOffset operator+(UniformShaderVarOffset other) const
        {
            if(!isValid()) return kInvalid;
            if(!other.isValid()) return kInvalid;

            return UniformShaderVarOffset(mByteOffset + other.mByteOffset);
        }

    private:
        // The underlying raw byte offset.
        ByteOffset mByteOffset = ByteOffset(-1);
    };

    /** Represents the offset of a resource-type shader variable relative to its enclosing type/buffer/block.

    A `ResourceShaderVarOffset` records the index of a descriptor range and an array index within that range.

    A `ResourceShaderVarOffset` can also encode an invalid offset (represented as an all-ones bit pattern
    for both the range and array indices), to indicate that a particular resource variable is not present.

    A `ResourceShaderVarOffset` can be obtained from a reflection type or `ParameterBlock` using the
    `[]` subscript operator:

        ResourceShaderVarOffset texOffset = pSomeType["tex"]; // get offset of texture `tex` inside `pSomeType`
        ResourceShaderVarOffset sampOffset = pBlock["samp"]; // get offset of sampler `samp` inside block

    Please note that the concepts of resource "ranges" are largely an implementation detail of
    the `ParameterBlock` type, and most user code should not attempt to explicitly work with
    or reason about resource ranges. In particular, there is *no* correspondence between resource
    range indices and the `register`s or `binding`s assigned to shader parameters.
    */
    struct ResourceShaderVarOffset
    {
    public:
        /** Custom enumeration type used to represent a zero offset.

        Can be used to initialize a `ResourceShaderVarOffset` when an explicit zero offset is desired:

            ResourceShaderVarOffset myOffset = ResourceShaderVarOffset::kZero;

        */
        enum Zero { kZero = 0 };

        /** Construct an explicit zero offset.
        */
        ResourceShaderVarOffset(Zero)
            : mRangeIndex(0)
            , mArrayIndex(0)
        {}

        /** Custom enumeration type used to represent an invalid offset.

        Can be used to initialize a `ResourceShaderVarOffset` when an explicit invalid is desired:

            ResourceShaderVarOffset myOffset = ResourceShaderVarOffset::kInvalid;

        Note that the default constructor also constructs an invalid offset, so this
        could be written more simply as:

            ResourceShaderVarOffset myOffset;

        */
        enum Invalid { kInvalid = -1 };

        /** Default constructor: constructs an invalid offset.
        */
        ResourceShaderVarOffset(Invalid _ = kInvalid)
            : mRangeIndex(RangeIndex(-1))
            , mArrayIndex(ArrayIndex(-1))
        {}

        /** Check if this is a valid offset.
        */
        bool isValid() const
        {
            return mRangeIndex != RangeIndex(-1);
        }

        /** Add a further offset to this offset.

        If either `this` or `other` is invalid, returns an invalid offset.
        */
        ResourceShaderVarOffset operator+(ResourceShaderVarOffset const& other) const
        {
            if(!isValid()) return kInvalid;
            if(!other.isValid()) return kInvalid;

            return ResourceShaderVarOffset(
                mRangeIndex + other.mRangeIndex,
                mArrayIndex + other.mArrayIndex);
        }

        /** Compare with another offset.
        */
        bool operator==(ResourceShaderVarOffset const& other) const
        {
            return mRangeIndex == other.mRangeIndex
                && mArrayIndex == other.mArrayIndex;
        }

        /** Compare with another offset.
        */
        bool operator!=(ResourceShaderVarOffset const& other) const
        {
            return !(*this == other);
        }

        /** Type used to store the resource/descriptor range.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        typedef uint32_t RangeIndex;

        /** Type used to store the array index within a range.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        typedef uint32_t ArrayIndex;

        /** Get the underlying resource/descriptor range index.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        RangeIndex getRangeIndex() const { return mRangeIndex; }

        /** Get the underlying array index into the resource/descriptor range.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        ArrayIndex getArrayIndex() const { return mArrayIndex; }

        /** Construct an offset representing an explicit resource range and array index.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        ResourceShaderVarOffset(
            RangeIndex rangeIndex,
            ArrayIndex arrayIndex)
            : mRangeIndex(rangeIndex)
            , mArrayIndex(arrayIndex)
        {}

        /** Construct an offset representing an explicit resource range.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        explicit ResourceShaderVarOffset(
            RangeIndex rangeIndex)
            : mRangeIndex(rangeIndex)
            , mArrayIndex(0)
        {}

    private:
        RangeIndex    mRangeIndex;
        ArrayIndex    mArrayIndex;
    };

    /** Represents the offset of a shader variable relative to its enclosing type/buffer/block.

    A `ShaderVarOffset` can be used to store the offset of a shader variable that might use
    ordinary/uniform data, resources like textures/buffers/samplers, or some combination.
    It effectively stores both a `UniformShaderVarOffset` and a `ResourceShaderVarOffset`

    A `ShaderVarOffset` can also encode an invalid offset, to indicate that a particular
    shader variable is not present.

    A `ShaderVarOffset` can be obtained from a reflection type or `ParameterBlock` using the
    `[]` subscript operator:

        ShaderVarOffset lightOffset = pSomeType["light"]; // get offset of variable `light` inside `pSomeType`
        ShaderVarOffset materialOffset = pBlock["material"]; // get offset of variable `material` inside block

    */
    struct ShaderVarOffset
    {
    public:
        /** Construct a shader variable offset from its underlying uniform and resource offsets.
        */
        ShaderVarOffset(
            UniformShaderVarOffset uniform,
            ResourceShaderVarOffset resource)
            : mUniform(uniform)
            , mResource(resource)
        {}

        /** Custom enumeration type used to represent an invalid offset.

        Can be used to initialize a `ShaderVarOffset` when an explicit invalid is desired:

            ShaderVarOffset myOffset = ShaderVarOffset::kInvalid;

        Note that the default constructor also constructs an invalid offset, so this
        could be written more simply as:

            ShaderVarOffset myOffset;

        */
        enum Invalid { kInvalid = -1 };

        /** Default constructor: constructs an invalid offset.
        */
        ShaderVarOffset(Invalid _ = kInvalid)
            : mUniform(UniformShaderVarOffset::kInvalid)
            , mResource(ResourceShaderVarOffset::kInvalid)
        {}

        /** Custom enumeration type used to represent a zero offset.

        Can be used to initialize a `ShaderVarOffset` when an explicit zero offset is desired:

            ShaderVarOffset myOffset = ShaderVarOffset::kZero;
        */
        enum Zero { kZero = 0 };

        /** Construct an explicit zero offset.
        */
        ShaderVarOffset(Zero)
            : mUniform(UniformShaderVarOffset::kZero)
            , mResource(ResourceShaderVarOffset::kZero)
        {}

        /** Check if this is a valid offset.
        */
        bool isValid() const
        {
            return mUniform.isValid();
        }

        /** Get the underlying uniform offset.
        */
        UniformShaderVarOffset getUniform() const
        {
            return mUniform;
        }

        /** Get the underlying uniform offset.

        This implicit conversion allows a `ShaderVarOffset` to be
        passed to functions that expect a `UniformShaderVarOffset`.
        */
        operator UniformShaderVarOffset() const
        {
            return mUniform;
        }

        /** Get the underlying resource offset.
        */
        ResourceShaderVarOffset getResource() const
        {
            return mResource;
        }

        /** Get the underlying resource offset.

        This implicit conversion allows a `ShaderVarOffset` to be
        passed to functions that expect a `ResourceShaderVarOffset`.
        */
        operator ResourceShaderVarOffset() const
        {
            return mResource;
        }

        /** Add an additional offset.

        If either `this` or `other` is invalid, returns an invalid offset.
        */
        ShaderVarOffset operator+(ShaderVarOffset const& other) const
        {
            if(!isValid()) return kInvalid;
            if(!other.isValid()) return kInvalid;

            return ShaderVarOffset(
                mUniform + other.mUniform,
                mResource + other.mResource);
        }

        /** Compare to another offset.
        */
        bool operator==(ShaderVarOffset const& other) const
        {
            return mUniform == other.mUniform
                && mResource == other.mResource;
        }

        /** Compare to another offset.
        */
        bool operator!=(ShaderVarOffset const& other) const
        {
            return !(*this == other);
        }

        /** Type used to store the underlying uniform byte offset.
        */
        using ByteOffset = UniformShaderVarOffset::ByteOffset;

        /** Get the uniform byte offset.
        */
        ByteOffset getByteOffset() const { return mUniform.getByteOffset(); }

        /** Type used to store the resource/descriptor range.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        using RangeIndex = ResourceShaderVarOffset::RangeIndex;

        /** Type used to store the array index within a range.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        using ArrayIndex = ResourceShaderVarOffset::ArrayIndex;

        /** Get the underlying resource range index.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        RangeIndex getResourceRangeIndex() const { return mResource.getRangeIndex(); }

        /** Get the underlying resource array index.

        Note: most user code should *not* need to work with explicit range/array indices.
        */
        ArrayIndex getResourceArrayIndex() const { return mResource.getArrayIndex(); }

    protected:
        UniformShaderVarOffset mUniform;
        ResourceShaderVarOffset mResource;
    };

    /** Represents the type of a shader variable and its offset relative to its enclosing type/buffer/block.

    A `TypedShaderVarOffset` is just a `ShaderVarOffset` plus a `ReflectionType` for
    the variable at the given offset.

    A `TypedShaderVarOffset` can also encode an invalid offset, to indicate that a particular
    shader variable is not present.

    A `TypedShaderVarOffset` can be obtained from a reflection type or `ParameterBlock` using the
    `[]` subscript operator:

        TypedShaderVarOffset lightOffset = pSomeType["light"]; // get type and offset of texture `light` inside `pSomeType`
        TypedShaderVarOffset materialOffset = pBlock["material"]; // get type and offset of sampler `material` inside block

    In addition, a `TypedShaderVarOffset` can be used to look up offsets for
    sub-fields/-elements of shader variables with structure or array types:

        UniformShaderVarOffset lightPosOffset = lightOffset["position"];
        ResourceShaderVarOffset diffuseMapOffset = materialOffset["diffuseMap"];

    Such offsets are always relative to the root type or block where lookup started.
    For example, in the above code `lightPosOffset` would be the offset of the
    field `light.position` relative to the enclosing type `pSomeType` and *not*
    the offset of the `position` field relative to the immediately enclosing `light` field.

    Because `TypedShaderVarOffset` inherits from `ShaderVarOffset` it can be used
    in all the same places, and also implicitly converts to both
    `UniformShaderVarOffset` and `ResourceShaderVarOffset`.
    */
    struct TypedShaderVarOffset : ShaderVarOffset
    {
    public:
        /** Default constructor: constructs an invalid offset.
        */
        TypedShaderVarOffset(Invalid _ = kInvalid)
        {}

        /** Get the type of the shader variable.
        */
        std::shared_ptr<const ReflectionType> getType() const
        {
            return mpType;
        }

        /** Check if `this` represents a valid offset.
        */
        bool isValid() const
        {
            return mpType != nullptr;
        }

        /** Look up type and offset of a sub-field with the given `name`.
        */
        TypedShaderVarOffset operator[](const std::string& name) const;

        /** Look up type and offset of a sub-field with the given `name`.
        */
        TypedShaderVarOffset operator[](const char*) const;

        /** Look up type and offset of a sub-element or sub-field with the given `index`.
        */
        TypedShaderVarOffset operator[](size_t index) const;

        /** Construct a typed shader variable offset from an explicit type and offset.

        The caller takes responsibility for ensuring that `pType` is a valid type
        for the data at `offset`.
        */
        TypedShaderVarOffset(
            const ReflectionType* pType,
            ShaderVarOffset       offset);

    private:
        std::shared_ptr<const ReflectionType> mpType;
    };

    /** Reflection and layout information for a type in shader code.
    */
    class FALCOR_API ReflectionType : public std::enable_shared_from_this<ReflectionType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionType>;

        virtual ~ReflectionType() = default;

        /** The kind of a type.

        Every type has a kind, which specifies which subclass of `ReflectionType` it uses.

        When adding new derived classes, this enumeration should be updated.
        */
        enum class Kind
        {
            Array,      ///< ReflectionArrayType
            Struct,     ///< ReflectionStructType
            Basic,      ///< ReflectionBasicType
            Resource,   ///< ReflectionResourceType
            Interface,  ///< ReflectionInterfaceType
        };

        /** Get the kind of this type.

        The kind tells us if we have an array, structure, etc.
        */
        Kind getKind() const { return mKind; }

        /** Dynamic-cast the current object to ReflectionResourceType
        */
        const ReflectionResourceType* asResourceType() const;

        /** Dynamic-cast the current object to ReflectionBasicType
        */
        const ReflectionBasicType* asBasicType() const;

        /** Dynamic-cast the current object to ReflectionStructType
        */
        const ReflectionStructType* asStructType() const;

        /** Dynamic-cast the current object to ReflectionArrayType
        */
        const ReflectionArrayType* asArrayType() const;

        /** Dynamic cast to ReflectionInterfaceType
        */
        const ReflectionInterfaceType* asInterfaceType() const;

        /** "Unwrap" any array types to get to the non-array type underneath.

        If `this` is not an array, then returns `this`.
        If `this` is an array, then applies `unwrapArray` to its element type.
        */
        const ReflectionType* unwrapArray() const;

        /** Get the total number of array elements represented by this type.

        If `this` is not an array, then returns 1.
        If `this` is an array, returns the number of elements times `getTotalArraySize()` for the element type.
        */
        uint32_t getTotalArrayElementCount() const;

        /** Type to represent the byte size of a shader type.
        */
        typedef size_t ByteSize;

        /** Get the size in bytes of instances of this type.

        This function only counts uniform/ordinary data, and not resources like textures/buffers/samplers.
        */
        ByteSize getByteSize() const { return mByteSize; }

        /** Find a field/member of this type with the given `name`.

        If this type doesn't have fields/members, or doesn't have a field/member matching `name`, then returns null.
        */
        std::shared_ptr<const ReflectionVar> findMember(const std::string& name) const;

        /** Get the (type and) offset of a field/member with the given `name`.

        If this type doesn't have fields/members, or doesn't have a field/member matching `name`,
        then logs an error and returns an invalid offset.
        */
        TypedShaderVarOffset getMemberOffset(const std::string& name) const;

        /** Find a typed member/element offset corresponding to the given byte offset.
        */
        TypedShaderVarOffset findMemberByOffset(size_t byteOffset) const;

        /** Get an offset that is zero bytes into this type.

        Useful for turning a `ReflectionType` into a `TypedShaderVarOffset` so
        that the `[]` operator can be used to look up members/elements.
        */
        TypedShaderVarOffset getZeroOffset() const;

        /** Compare types for equality.

        It is possible for two distinct `ReflectionType` instances to represent
        the same type with the same layout. The `==` operator must be used to
        tell if two types have the same structure.
        */
        virtual bool operator==(const ReflectionType& other) const = 0;

        /** Compare types for inequality.
        */
        bool operator!=(const ReflectionType& other) const { return !(*this == other); }

        /** A range of resources contained (directly or indirectly) in this type.

        Different types will contain different numbers of resources, and those
        resources will always be grouped into contiguous "ranges" that must be
        allocated together in descriptor sets to allow them to be indexed.

        Some examples:

        * A basic type like `float2` has zero resource ranges.

        * A resource type like `Texture2D` will have one resource range,
          with a corresponding descriptor type and an array count of one.

        * An array type like `float2[3]` or `Texture2D[4]` will have
          the same number of ranges as its element type, but the count
          of each range will be multiplied by the array element count.

        * A structure type like `struct { Texture2D a; Texture2D b[3]; }`
          will concatenate the resource ranges from its fields, in order.

        The `ResourceRange` type is mostly an implementation detail
        of `ReflectionType` that supports `ParameterBlock` and users
        should probably not rely on this information.
        */
        struct ResourceRange
        {
            // TODO(tfoley) consider renaming this to `DescriptorRange`.

            /** The type of descriptors that are stored in the range
            */
            ShaderResourceType descriptorType;

            /** The total number of descriptors in the range.
            */
            uint32_t count;

            /** If the enclosing type had its descriptors stored in
            flattened arrays, where would this range start?

            This is entirely an implementation detail of `ParameterBlock`.
            */
            uint32_t baseIndex;
        };

        /** Get the number of descriptor ranges contained in this type.
        */
        uint32_t getResourceRangeCount() const { return (uint32_t) mResourceRanges.size(); }

        /** Get information on a contained descriptor range.
        */
        ResourceRange const& getResourceRange(uint32_t index) const { return mResourceRanges[index]; }

        slang::TypeLayoutReflection* getSlangTypeLayout() const { return mpSlangTypeLayout; }

    protected:
        ReflectionType(Kind kind, ByteSize byteSize, slang::TypeLayoutReflection* pSlangTypeLayout)
            : mKind(kind)
            , mByteSize(byteSize)
            , mpSlangTypeLayout(pSlangTypeLayout)
        {}

        Kind mKind;
        ByteSize mByteSize = 0;
        std::vector<ResourceRange> mResourceRanges;
        slang::TypeLayoutReflection* mpSlangTypeLayout = nullptr;
    };

    /** Represents an array type in shader code.
    */
    class FALCOR_API ReflectionArrayType : public ReflectionType
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionArrayType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionArrayType>;

        /** Create a new object
        */
        static SharedPtr create(
            uint32_t elementCount,
            uint32_t elementByteStride,
            const ReflectionType::SharedConstPtr& pElementType,
            ByteSize byteSize,
            slang::TypeLayoutReflection*    pSlangTypeLayout);

        /** Get the number of elements in the array.
        */
        uint32_t getElementCount() const { return mElementCount; }

        /** Get the "stride" in bytes of the array.

        The stride is the number of bytes between consecutive
        array elements. It is *not* necessarily the same as
        the size of the array elements. For example an array
        of `float3`s in a constant buffer may have a stride
        of 16 bytes, but each element is only 12 bytes.
        */
        uint32_t getElementByteStride() const { return mElementByteStride; }

        /** Get the type of the array elements.
        */
        const ReflectionType::SharedConstPtr& getElementType() const { return mpElementType; }

        bool operator==(const ReflectionArrayType& other) const;
        bool operator==(const ReflectionType& other) const override;

    private:
        ReflectionArrayType(
            uint32_t                                elementCount,
            uint32_t                                elementByteStride,
            const ReflectionType::SharedConstPtr&   pElementType,
            ByteSize                                totalByteSize,
            slang::TypeLayoutReflection*    pSlangTypeLayout);

        uint32_t mElementCount = 0;
        uint32_t mElementByteStride = 0;
        ReflectionType::SharedConstPtr mpElementType;
    };

    /** Represents a `struct` type in shader code.
    */
    class FALCOR_API ReflectionStructType : public ReflectionType
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionStructType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionStructType>;

        /** Get the name of the struct type
        */
        const std::string& getName() const { return mName; }

        /** Get the total number members
        */
        uint32_t getMemberCount() const { return (uint32_t)mMembers.size(); }

        /** Get member by index
        */
        const std::shared_ptr<const ReflectionVar>& getMember(size_t index) const { return mMembers[index]; }

        /** Get member by name
        */
        const std::shared_ptr<const ReflectionVar>& getMember(const std::string& name) const;

        /** Constant used to indicate that member lookup failed.
        */
        static const int32_t kInvalidMemberIndex = -1;

        /** Get the index of a member

        Returns `kInvalidMemberIndex` if no such member exists.
        */
        int32_t getMemberIndex(const std::string& name) const;

        /** Find a member based on a byte offset.
        */
        TypedShaderVarOffset findMemberByOffset(size_t offset) const;

        bool operator==(const ReflectionStructType& other) const;
        bool operator==(const ReflectionType& other) const override;

        // TODO(tfoley): The following members are only needed to construct a type.

        /** Create a new structure type
            \param[in] size The size of the struct in bytes
            \param[in] name The name of the struct
        */
        static SharedPtr create(
            size_t              byteSize,
            const std::string& name,
            slang::TypeLayoutReflection*    pSlangTypeLayout);

        struct BuildState
        {
            uint32_t cbCount = 0;
            uint32_t srvCount = 0;
            uint32_t uavCount = 0;
            uint32_t samplerCount = 0;
        };

        /** Add a new member
        */
        int32_t addMember(const std::shared_ptr<const ReflectionVar>& pVar, BuildState& ioBuildState);

        int32_t addMemberIgnoringNameConflicts(const std::shared_ptr<const ReflectionVar>& pVar, BuildState& ioBuildState);

    private:
        ReflectionStructType(
            size_t size,
            const std::string& name,
            slang::TypeLayoutReflection*    pSlangTypeLayout);
        std::vector<std::shared_ptr<const ReflectionVar>> mMembers;   // Struct members
        std::unordered_map<std::string, int32_t> mNameToIndex; // Translates from a name to an index in mMembers
        std::string mName;
    };

    /** Reflection object for scalars, vectors and matrices
    */
    class FALCOR_API ReflectionBasicType : public ReflectionType
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionBasicType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionBasicType>;

        /** The type of the object
        */
        enum class Type
        {
            Bool,
            Bool2,
            Bool3,
            Bool4,

            Uint8,
            Uint8_2,
            Uint8_3,
            Uint8_4,

            Uint16,
            Uint16_2,
            Uint16_3,
            Uint16_4,

            Uint,
            Uint2,
            Uint3,
            Uint4,

            Uint64,
            Uint64_2,
            Uint64_3,
            Uint64_4,

            Int8,
            Int8_2,
            Int8_3,
            Int8_4,

            Int16,
            Int16_2,
            Int16_3,
            Int16_4,

            Int,
            Int2,
            Int3,
            Int4,

            Int64,
            Int64_2,
            Int64_3,
            Int64_4,

            Float16,
            Float16_2,
            Float16_3,
            Float16_4,

            Float16_2x2,
            Float16_2x3,
            Float16_2x4,
            Float16_3x2,
            Float16_3x3,
            Float16_3x4,
            Float16_4x2,
            Float16_4x3,
            Float16_4x4,

            Float,
            Float2,
            Float3,
            Float4,

            Float2x2,
            Float2x3,
            Float2x4,
            Float3x2,
            Float3x3,
            Float3x4,
            Float4x2,
            Float4x3,
            Float4x4,

            Float64,
            Float64_2,
            Float64_3,
            Float64_4,

            Unknown = -1
        };

        /** Create a new object
            \param[in] offset The offset of the variable relative to the parent variable
            \param[in] type The type of the object
            \param[in] isRowMajor For matrices, true means row-major, otherwise it's column-major
            \param[in] size The size of the object
        */
        static SharedPtr create(
            Type type, bool isRowMajor, size_t size,
            slang::TypeLayoutReflection*    pSlangTypeLayout);

        /** Get the object's type
        */
        Type getType() const { return mType; }

        /** Check if this is a row-major matrix or not. The result is only valid for matrices
        */
        bool isRowMajor() const { return mIsRowMajor; }

        bool operator==(const ReflectionBasicType& other) const;
        bool operator==(const ReflectionType& other) const override;
    private:
        ReflectionBasicType(
            Type type,
            bool isRowMajor,
            size_t size,
            slang::TypeLayoutReflection*    pSlangTypeLayout);
        Type mType;
        bool mIsRowMajor;
    };

    /** Reflection object for resources
    */
    class FALCOR_API ReflectionResourceType : public ReflectionType
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionResourceType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionResourceType>;

        /** Describes how the shader will access the resource
        */
        enum class ShaderAccess
        {
            Undefined,
            Read,
            ReadWrite
        };

        /** The expected return type
        */
        enum class ReturnType
        {
            Unknown,
            Float,
            Double,
            Int,
            Uint
        };

        /** The resource dimension
        */
        enum class Dimensions
        {
            Unknown,
            Texture1D,
            Texture2D,
            Texture3D,
            TextureCube,
            Texture1DArray,
            Texture2DArray,
            Texture2DMS,
            Texture2DMSArray,
            TextureCubeArray,
            AccelerationStructure,
            Buffer,

            Count
        };

        /** For structured-buffers, describes the type of the buffer
        */
        enum class StructuredType
        {
            Invalid,    ///< Not a structured buffer
            Default,    ///< Regular structured buffer
            Counter,    ///< RWStructuredBuffer with counter
            Append,     ///< AppendStructuredBuffer
            Consume     ///< ConsumeStructuredBuffer
        };

        /** The type of the resource
        */
        enum class Type
        {
            Texture,
            StructuredBuffer,
            RawBuffer,
            TypedBuffer,
            Sampler,
            ConstantBuffer,
            AccelerationStructure,
        };

        /** Create a new object
        */
        static SharedPtr create(
            Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess,
            slang::TypeLayoutReflection* pSlangTypeLayout);

        /** For structured- and constant-buffers, set a reflection-type describing the buffer's layout
        */
        void setStructType(const ReflectionType::SharedConstPtr& pType);

        /** Get the struct-type
        */
        const ReflectionType::SharedConstPtr& getStructType() const { return mpStructType; }

        const std::shared_ptr<const ParameterBlockReflection>& getParameterBlockReflector() const { return mpParameterBlockReflector; }
        void setParameterBlockReflector(const std::shared_ptr<const ParameterBlockReflection>& pReflector)
        {
            mpParameterBlockReflector = pReflector;
        }

        /** Get the dimensions
        */
        Dimensions getDimensions() const { return mDimensions; }

        /** Get the structured-buffer type
        */
        StructuredType getStructuredBufferType() const { return mStructuredType; }

        /** Get the resource return type
        */
        ReturnType getReturnType() const { return mReturnType; }

        /** Get the required shader access
        */
        ShaderAccess getShaderAccess() const { return mShaderAccess; }

        /** Get the resource type
        */
        Type getType() const { return mType; }

        /** For structured- and constant-buffers, return the underlying type size, otherwise returns 0
        */
        size_t getSize() const { return mpStructType ? mpStructType->getByteSize() : 0; }

        bool operator==(const ReflectionResourceType& other) const;
        bool operator==(const ReflectionType& other) const override;
    private:
        ReflectionResourceType(Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess,
            slang::TypeLayoutReflection* pSlangTypeLayout);

        Dimensions mDimensions;
        StructuredType mStructuredType;
        ReturnType mReturnType;
        ShaderAccess mShaderAccess;
        Type mType;
        ReflectionType::SharedConstPtr mpStructType;   // For constant- and structured-buffers
        std::shared_ptr<const ParameterBlockReflection> mpParameterBlockReflector; // For constant buffers and parameter blocks
    };

    /** Reflection object for resources
    */
    class FALCOR_API ReflectionInterfaceType : public ReflectionType
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionInterfaceType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionInterfaceType>;

        static SharedPtr create(
            slang::TypeLayoutReflection*    pSlangTypeLayout);

        bool operator==(const ReflectionInterfaceType& other) const;
        bool operator==(const ReflectionType& other) const override;

        const std::shared_ptr<const ParameterBlockReflection>& getParameterBlockReflector() const { return mpParameterBlockReflector; }
        void setParameterBlockReflector(const std::shared_ptr<const ParameterBlockReflection>& pReflector)
        {
            mpParameterBlockReflector = pReflector;
        }

    private:
        ReflectionInterfaceType(
            slang::TypeLayoutReflection*    pSlangTypeLayout);

        std::shared_ptr<const ParameterBlockReflection> mpParameterBlockReflector; // For interface types that have been specialized
    };

    /** An object describing a variable
    */
    class FALCOR_API ReflectionVar
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionVar>;
        using SharedConstPtr = std::shared_ptr<const ReflectionVar>;

        /** Create a new object
            \param[in] name The name of the variable
            \param[in] pType The type of the variable
            \param[in] bindLocation The offset of the variable relative to the parent object
        */
        static SharedPtr create(
            const std::string& name,
            const ReflectionType::SharedConstPtr& pType,
            ShaderVarOffset const& bindLocation);

        /** Get the variable name
        */
        const std::string& getName() const { return mName; }

        /** Get the variable type
        */
        const ReflectionType::SharedConstPtr& getType() const { return mpType; }

        /** Get the variable offset
        */
        ShaderVarOffset getBindLocation() const { return mBindLocation; }
        size_t getByteOffset() const { return mBindLocation.getByteOffset(); }
        size_t getOffset() const { return mBindLocation.getByteOffset(); }

        bool operator==(const ReflectionVar& other) const;
        bool operator!=(const ReflectionVar& other) const { return !(*this == other); }

    private:
        ReflectionVar(
            const std::string& name,
            const ReflectionType::SharedConstPtr& pType,
            ShaderVarOffset const& bindLocation);

        std::string mName;
        ReflectionType::SharedConstPtr mpType;
        ShaderVarOffset mBindLocation;
    };

    class ProgramReflection;

    /** A reflection object describing a parameter block
    */
    class FALCOR_API ParameterBlockReflection : public std::enable_shared_from_this<ParameterBlockReflection>
    {
    public:
        using SharedPtr = std::shared_ptr<ParameterBlockReflection>;
        using SharedConstPtr = std::shared_ptr<const ParameterBlockReflection>;

        static const uint32_t kInvalidIndex = 0xffffffff;

        /** Create a new parameter block reflector, for the given element type.
        */
        static SharedPtr create(
            ProgramVersion const* pProgramVersion,
            ReflectionType::SharedConstPtr const& pElementType);

        /** Create a new shader object reflector, for the given element type.
        */
        static SharedPtr create(
            ProgramVersion const* pProgramVersion,
            slang::TypeLayoutReflection* pElementType);

        /** Get the type of the contents of the parameter block.
        */
        ReflectionType::SharedConstPtr getElementType() const { return mpElementType; }

        using BindLocation = TypedShaderVarOffset;

        // TODO(tfoley): The following two functions really pertain to members, not just resources.

        /** Get the variable for a resource in the block
        */
        const ReflectionVar::SharedConstPtr getResource(const std::string& name) const;

        /** Get the bind-location for a resource in the block
        */
        BindLocation getResourceBinding(const std::string& name) const;

#if FALCOR_HAS_D3D12
        /// Information on how a particular descriptor set should be filled in.
        ///
        /// A single `ParameterBlock` may map to zero or more distinct descriptor
        /// sets, depending on what members it contains, and how those members
        /// are mapped to API registers/spaces.
        ///
        struct DescriptorSetInfo
        {
            /// The layout of the API descriptor set to allocate.
            D3D12DescriptorSet::Layout   layout;

            /// The indices of resource ranges within the parameter block
            /// to bind to the descriptor ranges of the above set.
            ///
            /// The order of entries in the `resourceRangeIndices`
            /// array will correspond to the order of the corresponding
            /// descriptor ranges in the `layout`.
            ///
            std::vector<uint32_t>   resourceRangeIndices;

            /// Information about a sub-object that should have some
            /// of its resource ranges bound into this descriptor set.
            ///
            struct SubObjectInfo
            {
                /// The index of the resource range that defines the
                /// sub-object.
                ///
                uint32_t resourceRangeIndexOfSubObject;

                /// The set index within the sub-object for which
                /// data should be written into this set.
                ///
                uint32_t setIndexInSubObject;
            };

            /// All of the sub-objects of this parameter block that should
            /// have data written into this descriptor sets.
            ///
            std::vector<SubObjectInfo> subObjects;
        };

        /** Get the number of descriptor sets that are needed for an object of this type.
        */
        uint32_t getD3D12DescriptorSetCount() const { return (uint32_t) mDescriptorSets.size(); }

        const DescriptorSetInfo& getD3D12DescriptorSetInfo(uint32_t index) const { return mDescriptorSets[index]; }

        /** Get the layout for the `index`th descriptor set that needs to be created for an object of this type.
        */
        const D3D12DescriptorSet::Layout& getD3D12DescriptorSetLayout(uint32_t index) const { return mDescriptorSets[index].layout; }
#endif // FALCOR_HAS_D3D12

        /** Describes binding information for a resource range.

        The resource ranges of a parameter block mirror those of its element type 1-to-1.
        Things like the descriptor type and count for a range can thus be queried on
        the element type, while the `ParameterBlockReflection` stores additional information
        pertinent to how resource ranges are bound to the pipeline state.
        */
        struct ResourceRangeBindingInfo
        {
            enum class Flavor
            {
                Simple,         ///< A simple resource range (texture/sampler/etc.)
                RootDescriptor, ///< A resource root descriptor (buffers only)
                ConstantBuffer, ///< A sub-object for a constant buffer
                ParameterBlock, ///< A sub-object for a parameter block
                Interface,      ///< A sub-object for an interface-type parameter
            };

            Flavor flavor = Flavor::Simple;
            ReflectionResourceType::Dimensions dimension = ReflectionResourceType::Dimensions::Unknown;

            uint32_t regIndex = 0;          ///< The register index
            uint32_t regSpace = 0;          ///< The register space

            uint32_t descriptorSetIndex = kInvalidIndex;    ///< The index of the descriptor set to be bound into, when flavor is Flavor::Simple.

            /// The reflection object for a sub-object range.
            ParameterBlockReflection::SharedConstPtr pSubObjectReflector;

            bool isDescriptorSet() const { return flavor == Flavor::Simple; }
            bool isRootDescriptor() const { return flavor == Flavor::RootDescriptor; }
        };

        struct DefaultConstantBufferBindingInfo
        {
            uint32_t regIndex = 0;          ///< The register index
            uint32_t regSpace = 0;          ///< The register space
            uint32_t descriptorSetIndex = kInvalidIndex;    ///< The index of the descriptor set to be bound into
            bool useRootConstants = false;
        };

        static SharedPtr createEmpty(
            ProgramVersion const* pProgramVersion);

        void setElementType(
            ReflectionType::SharedConstPtr const& pElementType);

        void addResourceRange(
            ResourceRangeBindingInfo const& bindingInfo);

        friend struct ParameterBlockReflectionFinalizer;
        void finalize();

        bool hasDefaultConstantBuffer() const;
        void setDefaultConstantBufferBindingInfo(DefaultConstantBufferBindingInfo const& info);
        DefaultConstantBufferBindingInfo const& getDefaultConstantBufferBindingInfo() const;

        /** Get the number of descriptor ranges contained in this type.
        */
        uint32_t getResourceRangeCount() const { return (uint32_t) mResourceRanges.size(); }

        ReflectionType::ResourceRange const& getResourceRange(uint32_t index) const { return getElementType()->getResourceRange(index); }

        /** Get binding information on a contained descriptor range.
        */
        ResourceRangeBindingInfo const& getResourceRangeBindingInfo(uint32_t index) const { return mResourceRanges[index]; }

        uint32_t getRootDescriptorRangeCount() const { return (uint32_t)mRootDescriptorRangeIndices.size(); }
        uint32_t getRootDescriptorRangeIndex(uint32_t index) const { return mRootDescriptorRangeIndices[index]; }

        uint32_t getParameterBlockSubObjectRangeCount() const { return (uint32_t) mParameterBlockSubObjectRangeIndices.size(); }
        uint32_t getParameterBlockSubObjectRangeIndex(uint32_t index) const { return mParameterBlockSubObjectRangeIndices[index]; }

        std::shared_ptr<const ProgramVersion> getProgramVersion() const;

        std::shared_ptr<const ReflectionVar> findMember(const std::string& name) const
        {
            return getElementType()->findMember(name);
        }

    protected:
        ParameterBlockReflection(
            ProgramVersion const* pProgramVersion);

    private:
        /// The element type of the parameter block
        ///
        /// For a `ConstantBuffer<T>` or `ParameterBlock<T>`,
        /// this will be the type `T`.
        ///
        ReflectionType::SharedConstPtr mpElementType;

        /// Binding information for the "default" constant buffer, if needed.
        ///
        DefaultConstantBufferBindingInfo mDefaultConstantBufferBindingInfo;

        /// Binding information for the resource ranges in the element type.
        ///
        /// For something like a `Texture2D` in the element type,
        /// this will record the corresponding `register` and `space`.
        ///
        std::vector<ResourceRangeBindingInfo> mResourceRanges;

#if FALCOR_HAS_D3D12
        /// Layout and binding information for all descriptor sets that
        /// must be created to represent the state of a parameter block
        /// using this reflector.
        ///
        /// Note: this array does *not* include information for descriptor
        /// sets that correspond to `ParameterBlock` sub-objects, since
        /// they are required to allocate and maintain their own
        /// descriptor sets that this object can simply re-use.
        ///
        std::vector<DescriptorSetInfo> mDescriptorSets;
#endif

        /// Indices of the resource ranges that represent root descriptors,
        /// and which therefore need their resources to be bound to the root signature.
        ///
        /// Note: this array does *not* include information for root descriptors
        /// that correspond to `ParameterBlock` and `ConstantBuffer` sub-objects, since they
        /// are required to allocate and maintain their own root descriptor range indices.
        ///
        std::vector<uint32_t> mRootDescriptorRangeIndices;

        /// Indices of the resource ranges that represent parameter blocks,
        /// and which therefore need their descriptor sets to be bound
        /// along with the descriptor sets directly stored on the parameter block
        ///
        std::vector<uint32_t> mParameterBlockSubObjectRangeIndices;

        ProgramVersion const* mpProgramVersion = nullptr;
    };

    typedef ParameterBlockReflection ParameterBlockReflection;

    class FALCOR_API EntryPointGroupReflection : public ParameterBlockReflection
    {
    public:
        using SharedPtr = std::shared_ptr<EntryPointGroupReflection>;
        using SharedConstPtr = std::shared_ptr<const EntryPointGroupReflection>;

        static SharedPtr create(
            ProgramVersion const*   pProgramVersion,
            uint32_t                groupIndex,
            std::vector<slang::EntryPointLayout*> const& pSlangEntryPointReflectors);

    private:
        EntryPointGroupReflection(
            ProgramVersion const* pProgramVersion);
    };
    typedef EntryPointGroupReflection EntryPointBaseReflection;

    /** Reflection object for an entire program. Essentially, it's a collection of ParameterBlocks
    */
    class FALCOR_API ProgramReflection
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramReflection>;
        using SharedConstPtr = std::shared_ptr<const ProgramReflection>;
        static const uint32_t kInvalidLocation = -1;

        /** Data structured describing a shader input/output variable. Used mostly to communicate VS inputs and PS outputs
        */
        struct ShaderVariable
        {
            uint32_t bindLocation = 0;      ///> The bind-location of the variable
            std::string semanticName;       ///> The semantic name of the variable
            ReflectionBasicType::Type type = ReflectionBasicType::Type::Unknown; ///> The type of the variable
        };
        using VariableMap = std::unordered_map<std::string, ShaderVariable>;

        using BindLocation = ParameterBlockReflection::BindLocation;

        /** Data structure describing a hashed string used in the program.
        */
        struct HashedString
        {
            uint32_t hash;
            std::string string;
        };

        /** Create a new object for a Slang reflector object
        */
        static SharedPtr create(
            ProgramVersion const* pProgramVersion,
            slang::ShaderReflection* pSlangReflector,
            std::vector<slang::EntryPointLayout*> const& pSlangEntryPointReflectors,
            std::string& log);

        static SharedPtr createEmpty();
        void finalize();

        std::shared_ptr<const ProgramVersion> getProgramVersion() const;

        /** Get parameter block by name
        */
        ParameterBlockReflection::SharedConstPtr getParameterBlock(const std::string& name) const;

        /** Get the default (unnamed) parameter block.
        */
        ParameterBlockReflection::SharedConstPtr getDefaultParameterBlock() const { return mpDefaultBlock; }

        /** For compute-shaders, return the required thread-group size
        */
        uint3 getThreadGroupSize() const { return mThreadGroupSize; }

        /** For pixel-shaders, check if we need to run the shader at sample frequency
        */
        bool isSampleFrequency() const { return mIsSampleFrequency; }

        /** Get a resource from the default parameter block
        */
        const ReflectionVar::SharedConstPtr getResource(const std::string& name) const;

        /** Search for a vertex attribute by its semantic name
        */
        const ShaderVariable* getVertexAttributeBySemantic(const std::string& semantic) const;

        /** Search for a vertex attribute by the variable name
        */
        const ShaderVariable* getVertexAttribute(const std::string& name) const;

        /** Get a pixel shader output variable
        */
        const ShaderVariable* getPixelShaderOutput(const std::string& name) const;

        /** Look up a type by name.
            \return nullptr if the type does not exist.
        */
        ReflectionType::SharedPtr findType(const std::string& name) const;

        ReflectionVar::SharedConstPtr findMember(const std::string& name) const;

        std::vector<EntryPointGroupReflection::SharedPtr> const& getEntryPointGroups() const { return mEntryPointGroups; }

        EntryPointGroupReflection::SharedPtr const& getEntryPointGroup(uint32_t index) const { return mEntryPointGroups[index]; }

        std::vector<HashedString> const& getHashedStrings() const { return mHashedStrings; }

    private:
        ProgramReflection(
            ProgramVersion const* pProgramVersion,
            slang::ShaderReflection* pSlangReflector,
            std::vector<slang::EntryPointLayout*> const& pSlangEntryPointReflectors,
            std::string& log);
        ProgramReflection(ProgramVersion const* pProgramVersion);
        ProgramReflection(const ProgramReflection&) = default;
        void setDefaultParameterBlock(const ParameterBlockReflection::SharedPtr& pBlock);

        ProgramVersion const* mpProgramVersion;

        ParameterBlockReflection::SharedPtr mpDefaultBlock;
        uint3 mThreadGroupSize;
        bool mIsSampleFrequency = false;

        VariableMap mPsOut;
        VariableMap mVertAttr;
        VariableMap mVertAttrBySemantic;

        slang::ShaderReflection* mpSlangReflector = nullptr;
        mutable std::map<std::string, ReflectionType::SharedPtr> mMapNameToType;

        std::vector<EntryPointGroupReflection::SharedPtr> mEntryPointGroups;

        std::vector<HashedString> mHashedStrings;
    };

    inline const std::string to_string(ReflectionBasicType::Type type)
    {
#define type_2_string(a) case ReflectionBasicType::Type::a: return #a;
        switch (type)
        {
            type_2_string(Bool);
            type_2_string(Bool2);
            type_2_string(Bool3);
            type_2_string(Bool4);
            type_2_string(Uint);
            type_2_string(Uint2);
            type_2_string(Uint3);
            type_2_string(Uint4);
            type_2_string(Int);
            type_2_string(Int2);
            type_2_string(Int3);
            type_2_string(Int4);
            type_2_string(Float);
            type_2_string(Float2);
            type_2_string(Float3);
            type_2_string(Float4);
            type_2_string(Float2x2);
            type_2_string(Float2x3);
            type_2_string(Float2x4);
            type_2_string(Float3x2);
            type_2_string(Float3x3);
            type_2_string(Float3x4);
            type_2_string(Float4x2);
            type_2_string(Float4x3);
            type_2_string(Float4x4);
        default:
            FALCOR_UNREACHABLE();
            return "";
        }
#undef type_2_string
    }

    inline const std::string to_string(ReflectionResourceType::ShaderAccess access)
    {
#define access_2_string(a) case ReflectionResourceType::ShaderAccess::a: return #a;
        switch (access)
        {
            access_2_string(Undefined);
            access_2_string(Read);
            access_2_string(ReadWrite);
        default:
            FALCOR_UNREACHABLE();
            return "";
        }
#undef access_2_string
    }

    inline const std::string to_string(ReflectionResourceType::ReturnType retType)
    {
#define type_2_string(a) case ReflectionResourceType::ReturnType::a: return #a;
        switch (retType)
        {
            type_2_string(Unknown);
            type_2_string(Float);
            type_2_string(Uint);
            type_2_string(Int);
        default:
            FALCOR_UNREACHABLE();
            return "";
        }
#undef type_2_string
    }

    inline const std::string to_string(ReflectionResourceType::Dimensions resource)
    {
#define type_2_string(a) case ReflectionResourceType::Dimensions::a: return #a;
        switch (resource)
        {
            type_2_string(Unknown);
            type_2_string(Texture1D);
            type_2_string(Texture2D);
            type_2_string(Texture3D);
            type_2_string(TextureCube);
            type_2_string(Texture1DArray);
            type_2_string(Texture2DArray);
            type_2_string(Texture2DMS);
            type_2_string(Texture2DMSArray);
            type_2_string(TextureCubeArray);
            type_2_string(Buffer);
        default:
            FALCOR_UNREACHABLE();
            return "";
        }
#undef type_2_string
    }

    inline const std::string to_string(ReflectionResourceType::Type type)
    {
#define type_2_string(a) case ReflectionResourceType::Type::a: return #a;
        switch (type)
        {
            type_2_string(Texture);
            type_2_string(ConstantBuffer);
            type_2_string(StructuredBuffer);
            type_2_string(RawBuffer);
            type_2_string(TypedBuffer);
            type_2_string(Sampler);
        default:
            FALCOR_UNREACHABLE();
            return "";
        }
#undef type_2_string
    }
}
