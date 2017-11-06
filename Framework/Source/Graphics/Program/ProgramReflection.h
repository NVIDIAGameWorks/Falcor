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
#include "Externals/slang/slang.h"

namespace Falcor
{
    class ReflectionType
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionType>;
        using SharedConstPtr = std::shared_ptr<const ReflectionType>;
        enum class Type
        {
            Unknown,
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

            ShaderResourceView,
            UnorderedAccessView,
            StructuredBuffer,
            RawBuffer,
            TypedBuffer,
            ConstantBuffer,
            Sampler,

            ParameterBlock,
            Struct,
        };
        static const uint32_t kInvalidOffset = -1;

        static SharedPtr create(Type type, uint32_t offset, uint32_t arraySize, uint32_t arrayStride);

        void addMember(const std::string& name, ReflectionType::SharedPtr pType);

        const ReflectionType::SharedConstPtr& getMember(const std::string& name);
        const ReflectionType::SharedConstPtr& getMember(uint32_t index);
        uint32_t getOffset() const { return mOffset; }
        uint32_t getArraySize() const { return mArraySize; }
        uint32_t getArrayStride() const { return mArrayStride; }
        Type getType() const { return mType; }

        std::vector<ReflectionType::SharedConstPtr>::const_iterator begin() const { return mMembers.begin(); }
        std::vector<ReflectionType::SharedConstPtr>::const_iterator end() const { return mMembers.end(); }
    private:
        ReflectionType();
        std::vector<ReflectionType::SharedConstPtr> mMembers;
        std::unordered_map<std::string, uint32_t> mMembersDictionary; // Translates from a name to an index in mMembers
        Type mType = Type::Unknown;
        uint32_t mOffset = kInvalidOffset;
        uint32_t mArraySize = 0;
        uint32_t mArrayStride = 0;
        bool mIsRowMajor = false;
    };

    class ReflectionVar
    {
    public:
        using SharedPtr = std::shared_ptr<ReflectionVar>;
        using SharedConstPtr = std::shared_ptr<const ReflectionVar>;
        static const uint32_t kInvalidOffset = ReflectionType::kInvalidOffset;

        static SharedPtr create(const std::string& name, ReflectionType::SharedConstPtr& pType, uint32_t offset, uint32_t regSpace = kInvalidOffset);

        ReflectionType::SharedConstPtr getType() const { return mpType; }
        uint32_t getOffset() const { return mOffset; }
        uint32_t getRegisterSpace() const { return mRegSpace; }
    private:
        ReflectionVar(const std::string& name, ReflectionType::SharedConstPtr& pType, uint32_t offset, uint32_t regSpace);
        ReflectionType::SharedConstPtr mpType;
        uint32_t mOffset = kInvalidOffset;
        uint32_t mRegSpace = kInvalidOffset;
        std::string& name;
    };

    class ProgramReflection
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramReflection>;
        using SharedConstPtr = std::shared_ptr<const ProgramReflection>;

        SharedPtr create(slang::ShaderReflection* pSlangReflector ,std::string& log);

        std::vector<ReflectionVar::SharedPtr>::const_iterator begin() const { return mParameterBlocks.begin(); }
        std::vector<ReflectionVar::SharedPtr>::const_iterator end() const { return mParameterBlocks.end(); }
    private:
        ProgramReflection(slang::ShaderReflection* pSlangReflector, std::string& log);
        void addBlock(const ReflectionVar::SharedConstPtr& pVar);
        std::vector<ReflectionVar::SharedPtr> mParameterBlocks;
        std::unordered_map<std::string, uint32_t> mNameToIndex;
    };
}