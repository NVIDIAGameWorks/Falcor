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
#include <unordered_map>
#include <unordered_set>
#include "Externals/Slang/slang.h"
#include "API/DescriptorSet.h"

namespace Falcor
{
    class ReflectionVar;
    class ReflectionResourceType;
    class ReflectionBasicType;
    class ReflectionStructType;
    class ReflectionArrayType;

    /** Base class for reflection types
    */
    class ReflectionType : public std::enable_shared_from_this<ReflectionType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionType>;
        static const uint32_t kInvalidOffset = -1;
        virtual ~ReflectionType() = default;

        /** Get a variable by name. The name can contain array indices and struct members
        */
        virtual std::shared_ptr<const ReflectionVar> findMember(const std::string& name) const;

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

        /** For ReflectionArrayType, recursively look for an underlying type which is not an array
        */
        const ReflectionType* unwrapArray() const;

        /** Get the total number of array elements. These accounts for all array-types until the ReflectionType returned by unwrapArray()
        */
        uint32_t getTotalArraySize() const;

        /** Get the size of the current object
        */
        virtual size_t getSize() const = 0;

        // Helper functions
        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace, uint32_t descOffset) const = 0;

        virtual bool operator==(const ReflectionType& other) const = 0;
        virtual bool operator!=(const ReflectionType& other) const { return !(*this == other); }
    protected:
        ReflectionType(size_t offset) : mOffset(offset) {}
        size_t mOffset;
    };

    /** Reflection object for array-types
    */
    class ReflectionArrayType : public ReflectionType, inherit_shared_from_this<ReflectionType, ReflectionArrayType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionArrayType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionArrayType>;

        /** Create a new object
        */
        static SharedPtr create(size_t offset, uint32_t arraySize, uint32_t arrayStride, const ReflectionType::SharedConstPtr& pType);

        /** Get the number of array-elements
        */
        uint32_t getArraySize() const { return mArraySize; }

        /** Get the size of each array-element
        */
        uint32_t getArrayStride() const { return mArrayStride; }

        /** Get the underlying reflection type
        */
        const ReflectionType::SharedConstPtr& getType() const { return mpType; }

        /** Get the size the array
        */
        virtual size_t getSize() const override { return mArrayStride * mArrayStride; }

        bool operator==(const ReflectionArrayType& other) const;
        bool operator==(const ReflectionType& other) const override;
    private:
        ReflectionArrayType(size_t offset, uint32_t arraySize, uint32_t arrayStride, const ReflectionType::SharedConstPtr& pType);
        uint32_t mArraySize = 0;
        uint32_t mArrayStride = 0;
        ReflectionType::SharedConstPtr mpType;

        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace, uint32_t descOffset) const override;
    };

    /** Reflection object for structs
    */
    class ReflectionStructType : public ReflectionType, inherit_shared_from_this<ReflectionType, ReflectionStructType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionStructType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionStructType>;

        /** Create a new object
            \param[in] offset The base offset of the object relative to the parent
            \param[in] size The size of the struct
            \param[in] name The name of the struct
        */
        static SharedPtr create(size_t offset, size_t size, const std::string& name = ""); // #PARAMBLOCK offset shouldn't be part of the type

        /** Add a new member
        */
        void addMember(const std::shared_ptr<const ReflectionVar>& pVar);

        /** Get the index of a member
        */
        size_t getMemberIndex(const std::string& name) const;

        /** Get member by name
        */
        const std::shared_ptr<const ReflectionVar>& getMember(const std::string& name) const { return getMember(getMemberIndex(name)); }

        /** Get member by index
        */
        const std::shared_ptr<const ReflectionVar>& getMember(size_t index) const { return mMembers[index]; }

        /** Get the total number members
        */
        uint32_t getMemberCount() const { return (uint32_t)mMembers.size(); }

        /** Get an iterator to beginning of the members vector
        */
        std::vector<std::shared_ptr<const ReflectionVar>>::const_iterator begin() const { return mMembers.begin(); }

        /** Get an iterator to the end of the members vector
        */
        std::vector<std::shared_ptr<const ReflectionVar>>::const_iterator end() const { return mMembers.end(); }

        /** Get the size of the struct
        */
        virtual size_t getSize() const override { return mSize; }

        /** Get the name of the struct
        */
        const std::string& getName() const { return mName; }

        bool operator==(const ReflectionStructType& other) const;
        bool operator==(const ReflectionType& other) const override;
    private:
        ReflectionStructType(size_t offset, size_t size, const std::string& name);
        std::vector<std::shared_ptr<const ReflectionVar>> mMembers;   // Struct members
        std::unordered_map<std::string, size_t> mNameToIndex; // Translates from a name to an index in mMembers
        size_t mSize;
        std::string mName;
        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace, uint32_t descOffset) const override;
    };

    /** Reflection object for scalars, vectors and matrices
    */
    class ReflectionBasicType : public ReflectionType, inherit_shared_from_this<ReflectionType, ReflectionBasicType>
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
            Uint,
            Uint2,
            Uint3,
            Uint4,
            Uint64,
            Uint64_2,
            Uint64_3,
            Uint64_4,
            Int,
            Int2,
            Int3,
            Int4,
            Int64,
            Int64_2,
            Int64_3,
            Int64_4,
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

            Unknown = -1
        };

        /** Create a new object
            \param[in] offset The offset of the variable relative to the parent variable
            \param[in] type The type of the object
            \param[in] isRowMajor For matrices, true means row-major, otherwise it's column-major
            \param[in] size The size of the object
        */
        static SharedPtr create(size_t offset, Type type, bool isRowMajor, size_t size); // #PARAMBLOCK offset shouldn't be part of the type

        /** Get the object's type
        */
        Type getType() const { return mType; }

        /** Check if this is a row-major matrix or not. The result is only valid for matrices
        */
        bool isRowMajor() const { return mIsRowMajor; }

        /** Get the size of the object
        */
        virtual size_t getSize() const override { return mSize; }

        bool operator==(const ReflectionBasicType& other) const;
        bool operator==(const ReflectionType& other) const override;
    private:
        ReflectionBasicType(size_t offset, Type type, bool isRowMajor, size_t size);
        Type mType;
        size_t mSize;
        bool mIsRowMajor;
        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace, uint32_t descOffset) const override;
    };

    /** Reflection object for resources
    */
    class ReflectionResourceType : public ReflectionType, public inherit_shared_from_this<ReflectionType, ReflectionResourceType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionResourceType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionResourceType>;

        /** Offset descriptor. Helper struct for constant- and structured-buffers.
            For valid offsets (which have matching fields in the buffer), the struct will contain the basic type and the number of elements (or 0 in case it's not an array)
            For invalid offsets, type will be Unknown
        */
        struct OffsetDesc
        {
            ReflectionBasicType::Type type = ReflectionBasicType::Type::Unknown;
            uint32_t count = 0;
        };
        using OffsetDescMap = std::unordered_map<size_t, OffsetDesc>; // For constant- and structured-buffers

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
            Buffer,
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
            ConstantBuffer
        };
        
        /** Create a new object
        */
        static SharedPtr create(Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess);

        /** For structured- and constant-buffers, set a reflection-type describing the buffer's layout
        */
        void setStructType(const ReflectionType::SharedConstPtr& pType);

        /** Get the struct-type
        */
        const ReflectionType::SharedConstPtr& getStructType() const { return mpStructType; }

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
        size_t getSize() const { return mpStructType ? mpStructType->getSize() : 0; }

        /** For structured- and constant-buffers, get an offset descriptor.
            This function is useful in cases we want to make sure that (a) a specific offset has an associated field in the buffer; and (b) the type of the field matches the user's expectations
        */
        const OffsetDesc& getOffsetDesc(size_t offset) const;

        bool operator==(const ReflectionResourceType& other) const;
        bool operator==(const ReflectionType& other) const override;
    private:
        ReflectionResourceType(Type type, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess);
        Dimensions mDimensions;
        StructuredType mStructuredType;
        ReturnType mReturnType;
        ShaderAccess mShaderAccess;
        Type mType;
        ReflectionType::SharedConstPtr mpStructType;   // For constant- and structured-buffers
        OffsetDescMap mOffsetDescMap;

        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace, uint32_t descOffset) const override;
    };

    /** An object describing a variable
    */
    class ReflectionVar
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionVar>;
        using SharedConstPtr = std::shared_ptr<const ReflectionVar>;
        static const uint32_t kInvalidOffset = ReflectionType::kInvalidOffset;

        /** Create a new object
            \param[in] name The name of the variable
            \param[in] pType The type of the variable
            \param[in] offset The offset of the variable relative to the parent object
            \param[in] descOffset In case of a resource, the offset in descriptors relative to the base descriptor-set
            \param[in] regSpace In case of a resource, the register space
        */
        static SharedPtr create(const std::string& name, const ReflectionType::SharedConstPtr& pType, size_t offset, uint32_t descOffset = 0, uint32_t regSpace = kInvalidOffset);

        /** Get the variable name
        */
        const std::string& getName() const { return mName; }

        /** Get the variable type
        */
        const ReflectionType::SharedConstPtr getType() const { return mpType; }

        /** Get the variable offset
        */
        size_t getOffset() const { return mOffset; }

        /** Get the register space
        */
        uint32_t getRegisterSpace() const { return mRegSpace; }

        /** Get the register index
        */
        uint32_t getRegisterIndex() const { return (uint32_t)getOffset(); }

        /** Get the descriptor-offset
        */
        uint32_t getDescOffset() const { return mDescOffset; }

        bool operator==(const ReflectionVar& other) const;
        bool operator!=(const ReflectionVar& other) const { return !(*this == other); }
    private:
        ReflectionVar(const std::string& name, const ReflectionType::SharedConstPtr& pType, size_t offset, uint32_t descOffset, uint32_t regSpace);
        ReflectionType::SharedConstPtr mpType;
        size_t mOffset = kInvalidOffset;
        uint32_t mRegSpace = kInvalidOffset;
        std::string mName;
        size_t mSize = 0;
        uint32_t mDescOffset = 0;
    };

    class ProgramReflection;

    /** A reflection object describing a parameter-bloc
    */
    class ParameterBlockReflection
    {
    public:
        using SharedPtr = std::shared_ptr<ParameterBlockReflection>;
        using SharedConstPtr = std::shared_ptr<const ParameterBlockReflection>;

        /** Data structure describing a resource
        */
        struct ResourceDesc
        {
            using Type = DescriptorSet::Type;
            uint32_t descOffset = 0;        ///> The offset in descriptros relative to the base descriptor-set
            uint32_t descCount = 0;         ///> The number of descriptors (can be more than 1 in case of an array)
            uint32_t regIndex = 0;          ///> The register index
            uint32_t regSpace = 0;          ///> The register space
            Type setType;                   ///> The required descriptor-set type
            std::string name;               ///> The name of the variable
            ReflectionResourceType::SharedConstPtr pType;   ///> The resource-type
        };

        /** Data structure describing a resource's bind-location. Not to be confused with register-space and register-index which are global indices, bind-location is relative to the ParameterBlock.
        */
        struct BindLocation
        {
            BindLocation() = default;
            BindLocation(uint32_t set, uint32_t range) : setIndex(set), rangeIndex(range){}
            static const uint32_t kInvalidLocation = -1;        ///> Invalid index
            uint32_t setIndex = kInvalidLocation;               ///> The set-index in the parameter-block
            uint32_t rangeIndex = kInvalidLocation;             ///> The range-index in the selected set
        };

        using ResourceVec = std::vector<ResourceDesc>;
        using SetLayoutVec = std::vector<DescriptorSet::Layout>;

        /** Create a new object
        */
        static SharedPtr create(const std::string& name);

        /** Get the name of the parameter block
        */
        const std::string& getName() const { return mName; }

        /** Check if the block contains any resources
        */
        bool isEmpty() const;

        /** Get the variable for a resource in the block
        */
        const ReflectionVar::SharedConstPtr getResource(const std::string& name) const;

        /** Get the bind-location for a resource in the block
        */
        BindLocation getResourceBinding(const std::string& name) const;

        /** Get a vector with all the resources in the block
        */
        const ResourceVec& getResourceVec() const { return mResources; }

        /** Get a vector with the required descriptor-set layouts for the block. Useful when creating root-signatures
        */
        const SetLayoutVec& getDescriptorSetLayouts() const { return mSetLayouts; }
    private:
        friend class ProgramReflection;
        void addResource(const ReflectionVar::SharedConstPtr& pVar);
        void finalize();
        ParameterBlockReflection(const std::string& name);
        ResourceVec mResources;
        ReflectionStructType::SharedPtr mpResourceVars;
        std::string mName;
        std::unordered_map<std::string, BindLocation> mResourceBindings;

        SetLayoutVec mSetLayouts;
    };

    /** Reflection object for an entire program. Essentialy, it's a collection of ParameterBlocks
    */
    class ProgramReflection
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

        /** Create a new object for a Slang reflector object
        */
        static SharedPtr create(slang::ShaderReflection* pSlangReflector ,std::string& log);

        /** Get the index of a parameter block
        */
        uint32_t getParameterBlockIndex(const std::string& name) const;

        /** Get parameter block by name
        */
        const ParameterBlockReflection::SharedConstPtr& getParameterBlock(const std::string& name) const;

        /** Get parameter block by index
        */
        const ParameterBlockReflection::SharedConstPtr& getParameterBlock(uint32_t index) const;

        /** Get the number of parameter blocks
        */
        size_t getParameterBlockCount() const { return mpParameterBlocks.size(); }
        
        /** Get the default (unnamed) parameter block. This function is an alias to getParameterBlock("");
        */
        const ParameterBlockReflection::SharedConstPtr& getDefaultParameterBlock() const { return mpDefaultBlock; }

        /** For compute-shaders, return the required thread-group size
        */
        uvec3 getThreadGroupSize() const { return mThreadGroupSize; }

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

        /** Resource bind type
        */
        enum class BindType
        {
            Cbv,        ///> Constant-buffer view
            Srv,        ///> Shader-resource view
            Uav,        ///> Unordered-access view
            Sampler,    ///> Sampler
        };

        /** Translate a global register location (space, index, type) to a relative bind-location in the default parameter block
        */
        const ParameterBlockReflection::BindLocation translateRegisterIndicesToBindLocation(uint32_t regSpace, uint32_t baseRegIndex, BindType type) const { return mResourceBindMap.at({regSpace, baseRegIndex, type}); }

    private:
        ProgramReflection(slang::ShaderReflection* pSlangReflector, std::string& log);
        void addParameterBlock(const ParameterBlockReflection::SharedConstPtr& pBlock);

        std::vector<ParameterBlockReflection::SharedConstPtr> mpParameterBlocks;
        std::unordered_map<std::string, size_t> mParameterBlocksIndices;

        ParameterBlockReflection::SharedConstPtr mpDefaultBlock;
        uvec3 mThreadGroupSize;
        bool mIsSampleFrequency = false;

        VariableMap mPsOut;
        VariableMap mVertAttr;
        VariableMap mVertAttrBySemantic;

        struct ResourceBinding
        {
            static const uint32_t kInvalidLocation = -1;
            uint32_t regSpace;
            uint32_t regIndex;
            BindType type;
            bool operator==(const ResourceBinding& other) const { return (regIndex == other.regIndex) && (regSpace == other.regSpace) && (type == other.type); }
        };

        struct ResourceBindingHash
        {
            std::size_t operator()(const ResourceBinding& d) const
            {
                std::hash<uint32_t> u32hash;
                size_t h = u32hash(d.regSpace) | ((u32hash(d.regIndex)) << 1);
                h |= u32hash((uint32_t)d.type) > 1;
                return h;
            }
        };

        std::unordered_map<ResourceBinding, ParameterBlockReflection::BindLocation, ResourceBindingHash> mResourceBindMap;
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
            should_not_get_here();
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
            should_not_get_here();
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
            should_not_get_here();
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
            should_not_get_here();
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
            should_not_get_here();
            return "";
        }
#undef type_2_string
    }
}