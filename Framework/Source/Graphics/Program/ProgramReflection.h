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
#include "Externals/slang/slang.h"

namespace Falcor
{
    class ReflectionVar;
    class ReflectionResourceType;
    class ReflectionBasicType;
    class ReflectionStructType;
    class ReflectionArrayType;

    class ReflectionType : public std::enable_shared_from_this<ReflectionType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionType>;
        static const uint32_t kInvalidOffset = -1;
        virtual ~ReflectionType() = default;
        virtual std::shared_ptr<const ReflectionVar> findMember(const std::string& name) const;

        const ReflectionResourceType* asResourceType() const;
        const ReflectionBasicType* asBasicType() const;
        const ReflectionStructType* asStructType() const;
        const ReflectionArrayType* asArrayType() const;
        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace) const = 0;
    protected:
        ReflectionType(size_t offset) : mOffset(offset) {}
        size_t mOffset;
    };

    class ReflectionArrayType : public ReflectionType, inherit_shared_from_this<ReflectionType, ReflectionArrayType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionArrayType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionArrayType>;
        static SharedPtr create(size_t offset, uint32_t arraySize, uint32_t arrayStride, const ReflectionType::SharedConstPtr& pType);

        uint32_t getArraySize() const { return mArraySize; }
        uint32_t getArrayStride() const { return mArrayStride; }
        const ReflectionType::SharedConstPtr& getType() const { return mpType; }
    private:
        ReflectionArrayType(size_t offset, uint32_t arraySize, uint32_t arrayStride, const ReflectionType::SharedConstPtr& pType);
        uint32_t mArraySize = 0;
        uint32_t mArrayStride = 0;
        ReflectionType::SharedConstPtr mpType;

        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace) const override;
    };

    class ReflectionStructType : public ReflectionType, inherit_shared_from_this<ReflectionType, ReflectionStructType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionStructType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionStructType>;

        static SharedPtr create(size_t offset, size_t size, const std::string& name = "");

        void addMember(const std::shared_ptr<const ReflectionVar>& pVar);
        const std::shared_ptr<const ReflectionVar>& getMember(const std::string& name);
        const std::shared_ptr<const ReflectionVar>& getMember(uint32_t index);
        uint32_t getMemberCount() const { return (uint32_t)mMembers.size(); }

        size_t getMemberIndex(const std::string& name) const;
        const std::shared_ptr<const ReflectionVar>& getMember(size_t index) const { return mMembers[index]; }

        std::vector<std::shared_ptr<const ReflectionVar>>::const_iterator begin() const { return mMembers.begin(); }
        std::vector<std::shared_ptr<const ReflectionVar>>::const_iterator end() const { return mMembers.end(); }

        size_t getSize() const { return mSize; }
        const std::string& getName() const { return mName; }
        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace) const override;
    private:
        ReflectionStructType(size_t offset, size_t size, const std::string& name);
        std::vector<std::shared_ptr<const ReflectionVar>> mMembers;   // Struct members
        std::unordered_map<std::string, size_t> mNameToIndex; // Translates from a name to an index in mMembers
        size_t mSize;
        std::string mName;
    };

    class ReflectionBasicType : public ReflectionType, inherit_shared_from_this<ReflectionType, ReflectionBasicType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionBasicType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionBasicType>;

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

        static SharedPtr create(size_t offset, Type type, bool isRowMajor);
        Type getType() const { return mType; }
    private:
        ReflectionBasicType(size_t offset, Type type, bool isRowMajor);
        Type mType;
        bool mIsRowMajor;
        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace) const override;
    };

    class ReflectionResourceType : public ReflectionType, inherit_shared_from_this<ReflectionType, ReflectionResourceType>
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionResourceType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionResourceType>;
        enum class ShaderAccess
        {
            Undefined,
            Read,
            ReadWrite
        };

        enum class ReturnType
        {
            Unknown,
            Float,
            Double,
            Int,
            Uint
        };

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

        enum class StructuredType
        {
            Invalid,    ///< Not a structured buffer
            Default,    ///< Regular structured buffer
            Counter,    ///< RWStructuredBuffer with counter
            Append,     ///< AppendStructuredBuffer
            Consume     ///< ConsumeStructuredBuffer
        };

        enum class Type
        {
            Texture,
            StructuredBuffer,
            RawBuffer,
            TypedBuffer,
            Sampler,
            ConstantBuffer,
        };
        
        static SharedPtr create(Type type, uint32_t regIndex, uint32_t regSpace, Dimensions dims, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess);
        void setStructType(const ReflectionStructType::SharedPtr& pType) { mpStructType = pType; }

        const ReflectionStructType::SharedPtr& getStructType() const { return mpStructType; }
        Dimensions getDimensions() const { return mDimensions; }
        StructuredType getStructuredBufferType() const { return mStructuredType; }
        ReturnType getReturnType() const { return mReturnType; }
        ShaderAccess getShaderAccess() const { return mShaderAccess; }
        uint32_t getRegisterIndex() const { return mRegIndex; }
        uint32_t getRegisterSpace() const { return mRegSpace; }
        Type getType() const { return mType; }
        size_t getSize() const { return mpStructType ? mpStructType->getSize() : 0; }
    private:
        ReflectionResourceType(Type type, uint32_t regIndex, uint32_t regSpace, StructuredType structuredType, ReturnType retType, ShaderAccess shaderAccess);
        Dimensions mDimensions;
        StructuredType mStructuredType;
        ReturnType mReturnType;
        ShaderAccess mShaderAccess;
        uint32_t mRegIndex;
        uint32_t mRegSpace;
        Type mType;
        ReflectionStructType::SharedPtr mpStructType;   // For constant- and structured-buffers

        virtual std::shared_ptr<const ReflectionVar> findMemberInternal(const std::string& name, size_t strPos, size_t offset, uint32_t regIndex, uint32_t regSpace) const override;
    };

    class ReflectionVar
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionVar>;
        using SharedConstPtr = std::shared_ptr<const ReflectionVar>;
        static const uint32_t kInvalidOffset = ReflectionType::kInvalidOffset;

        static SharedPtr create(const std::string& name, const ReflectionType::SharedConstPtr& pType, size_t offset, uint32_t regSpace = kInvalidOffset);

        const std::string& getName() const { return mName; }
        const ReflectionType::SharedConstPtr getType() const { return mpType; }
        size_t getOffset() const { return mOffset; }
        uint32_t getRegisterSpace() const { return mRegSpace; }
        uint32_t getRegisterIndex() const { return (uint32_t)getOffset(); }
    private:
        ReflectionVar(const std::string& name, const ReflectionType::SharedConstPtr& pType, size_t offset, uint32_t regSpace);
        ReflectionType::SharedConstPtr mpType;
        size_t mOffset = kInvalidOffset;
        uint32_t mRegSpace = kInvalidOffset;
        std::string mName;
        size_t mSize = 0;
    };

    class ProgramReflection;

    class ParameterBlockReflection
    {
    public:
        using SharedPtr = std::shared_ptr<ParameterBlockReflection>;
        using SharedConstPtr = std::shared_ptr<const ParameterBlockReflection>;
        using ResourceMap = std::unordered_map<std::string, ReflectionVar::SharedConstPtr>;
            
        static SharedPtr create(const std::string& name);
        const std::string& getName() const { return mName; }
        bool isEmpty() const;

        const ResourceMap& getResources() const { return mResources; }
        const ResourceMap& getConstantBuffers() const { return mConstantBuffers; }
        const ResourceMap& getStructuredBuffers() const { return mStructuredBuffers; }

        const ReflectionVar* getConstantBuffer(const std::string& name) const;
        const ReflectionVar* getResource(const std::string& name) const;
        const ReflectionVar* getStructuredBuffer(const std::string& name) const;
    private:
        friend class ProgramReflection;
        void addResource(const std::string& fullName, const ReflectionVar::SharedConstPtr& pVar);
        ParameterBlockReflection(const std::string& name);
        ResourceMap mResources;
        ResourceMap mConstantBuffers;
        ResourceMap mStructuredBuffers;

        std::string mName;
    };

    class ProgramReflection
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramReflection>;
        using SharedConstPtr = std::shared_ptr<const ProgramReflection>;

        static SharedPtr create(slang::ShaderReflection* pSlangReflector ,std::string& log);

        static void registerParameterBlock(const std::string& name);
        static void unregisterParameterBlock(const std::string& name);
        const ParameterBlockReflection::SharedConstPtr& getParameterBlock(const std::string& name) const;

        static const uint32_t kInvalidLocation = -1;
        struct ResourceBinding
        {
            uint32_t regIndex = kInvalidLocation;
            uint32_t regSpace = kInvalidLocation;
        };

        ResourceBinding getBufferBinding(const std::string& name) const;
    private:
        ProgramReflection(slang::ShaderReflection* pSlangReflector, std::string& log);
        void addParameterBlock(const ParameterBlockReflection::SharedConstPtr& pBlock);
        std::unordered_map<std::string, ParameterBlockReflection::SharedConstPtr> mParameterBlocks;
        static std::unordered_set<std::string> sParameterBlockRegistry;
        ParameterBlockReflection::SharedConstPtr mpGlobalBlock;
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