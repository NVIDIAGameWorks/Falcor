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
    /** This class holds all of the data required to reflect a program, including inputs, outputs, constants, textures and samplers declarations
    */
    class ProgramReflection
    {
    public:
        using SharedPtr = std::shared_ptr<ProgramReflection>;
        using SharedConstPtr = std::shared_ptr<const ProgramReflection>;
        using ReflectionHandleVector = std::vector<slang::ShaderReflection*>;

        enum class ShaderAccess
        {
            Undefined,
            Read,
            ReadWrite
        };

        /** Variable definition
        */
        struct Variable
        {
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
                GpuPtr,
                Resource
            };

            size_t location = 0;        ///< The offset of the variable from the start of the buffer or the location in case this is a global variable (FragOut, VertexAttribute)
            uint32_t arraySize = 0;     ///< Array size or 0 if not an array
            uint32_t arrayStride = 0;   ///< Stride between elements in the array. 0 If not an array
            bool isRowMajor = false;    ///< For matrices, tells if this is a row-major or column-major matrix
            Type type = Type::Unknown;  ///< The data type
        };

        /** Shader resource definition
        */
        struct Resource
        {
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

            enum class ResourceType
            {
                Unknown,
                Texture,
                StructuredBuffer,
                RawBuffer,
                TypedBuffer,
                Sampler
            };

            ShaderAccess shaderAccess = ShaderAccess::Undefined;
            ResourceType type = ResourceType::Unknown;      ///< Resource type
            Dimensions dims = Dimensions::Unknown;          ///< Resource dimensions
            ReturnType retType = ReturnType::Unknown;       ///< Resource return type
            uint32_t regIndex = -1;                         ///< Base shader register
            uint32_t arraySize = 0;                         ///< Array size , or 0 if not an array
            uint32_t shaderMask = 0;                        ///< A mask indicating in which shader stages the buffer is used
            uint32_t regSpace = 0;                          ///< The register space
            uint32_t descOffset = 0;						///< The offset to the element in the descriptor set
            Resource(Dimensions d, ReturnType r, ResourceType t, ShaderAccess s) : dims(d), retType(r), type(t), shaderAccess(s) {}
            Resource() {}
        };

        struct BindLocation
        {
            BindLocation() = default;
            BindLocation(uint32_t space, uint32_t index, ShaderAccess access) : baseRegIndex(index), regSpace(space), shaderAccess(access) {}
            uint32_t baseRegIndex = -1;
            uint32_t regSpace = -1;
            ShaderAccess shaderAccess = ShaderAccess(-1);

            std::size_t operator()(BindLocation b) const {
                return ((std::hash<uint32_t>()(baseRegIndex)
                    ^ (std::hash<uint32_t>()(regSpace) << 1)) >> 1)
                    ^ (std::hash<uint32_t>()((uint32_t)shaderAccess) << 1);
            }
            bool operator==(const BindLocation& other) const { return (baseRegIndex == other.baseRegIndex) && (regSpace == other.regSpace) && (shaderAccess == other.shaderAccess); }
            bool operator!=(const BindLocation& other) const { return !(*this == other); }
        };

        using VariableMap = std::unordered_map<std::string, Variable>;
        using ResourceMap = std::map < std::string, Resource >;
        using string_2_bindloc_map = std::unordered_map<std::string, BindLocation>;

        /** Invalid location of buffers and attributes
        */
        static const uint32_t kInvalidLocation = -1;

        /** This class holds all of the data required to reflect a buffer, either constant buffer or SSBO
        */
        class BufferReflection
        {
        public:
            using SharedPtr = std::shared_ptr<BufferReflection>;
            using SharedConstPtr = std::shared_ptr<const BufferReflection>;

            /** Buffer type
            */
            enum class Type
            {
                Constant,
                Structured,

                Count
            };

            enum class StructuredType
            {
                Invalid,    ///< Not a structured buffer
                Default,    ///< Regular structured buffer
                Counter,    ///< RWStructuredBuffer with counter
                Append,     ///< AppendStructuredBuffer
                Consume     ///< ConsumeStructuredBuffer
            };

            static const uint32_t kTypeCount = (uint32_t)Type::Count;

            /** Create a new object
                \param[in] name The name of the buffer as was declared in the program
                \param[in] regIndex The register index allocated for the buffer inside the program
                \param[in] regSpace The register space allocated for the buffer inside the program
                \param[in] size The size of the buffer
                \param[in] varMap Map describing each variable in the buffer, excluding resources
                \param[in] resourceMap Map describing the resources defined as part of the buffer. This map is only valid for APIs that support resource declarations nested inside buffers
                \param[in] shaderAccess How the buffer will be access by the shader
                \return A shared pointer for a new buffer object
            */
            static SharedPtr create(const std::string& name, uint32_t regSpace, uint32_t baseRegIndex, uint32_t arraySize, Type type, StructuredType structuredType, size_t size, const VariableMap& varMap, const ResourceMap& resourceMap, ShaderAccess shaderAccess);

            /** Get variable data
                \param[in] name The name of the requested variable
                \param[out] offset The offset of the variable or kInvalidLocation if the variable wasn't found. This is useful in cases where the requested variable is an array element, since the returned result will be different than Variable::offset
                \return Pointer to the variable data, or nullptr if the name wasn't found
            */
            const Variable* getVariableData(const std::string& name, size_t& offset) const;

            /** Get variable data
                \param[in] name The name of the requested variable
                \param[out] offset The offset of the variable or kInvalidLocation if the variable wasn't found. This is useful in cases where the requested variable is an array element, since the returned result will be different than Variable::offset
                \return Pointer to the variable data, or nullptr if the name wasn't found
            */
            FALCOR_DEPRECATED("The `allowNonIndexedArray` parameter has been deprecated. Please use the version of this function without this parameter.")
            const Variable* getVariableData(const std::string& name, size_t& offset, bool /*allowNonIndexedArray*/) const
            { return getVariableData(name, offset); }

            /** Get variable data
            \param[in] name The name of the requested variable
            \return Pointer to the variable data, or nullptr if the name wasn't found
            */
            const Variable* getVariableData(const std::string& name) const;

            /** Get variable data
            \param[in] name The name of the requested variable
            \return Pointer to the variable data, or nullptr if the name wasn't found
            */
            FALCOR_DEPRECATED("The `allowNonIndexedArray` parameter has been deprecated. Please use the version of this function without this parameter.")
            const Variable* getVariableData(const std::string& name, bool /*allowNonIndexedArray*/) const
            { return getVariableData(name); }

            /** Get resource data
            \param[in] name The name of the requested resource
            \return Pointer to the resource data, or nullptr if the name wasn't found or is not a resource
            */
            const Resource* getResourceData(const std::string& name) const;

            /** Get an iterator to the first variable
            */
            VariableMap::const_iterator varBegin() const { return mVariables.begin(); }

            /** Get an iterator to the end of the variable list
            */
            VariableMap::const_iterator varEnd() const {return mVariables.end(); }

            /** Get an iterator to the first resource
            */
            ResourceMap::const_iterator resourceBegin() const { return mResources.begin(); }

            /** Get an iterator to the end of the variable list
            */
            ResourceMap::const_iterator resourceEnd() const { return mResources.end(); }

            /** Get the buffer's name
            */
            const std::string& getName() const { return mName; }

            /** Get the required buffer size
            */
            size_t getRequiredSize() const { return mSizeInBytes; }
            
            /** Get the type of the buffer
            */
            Type getType() const { return mType; }

            /** Get the type of the structured buffer. Returns StructuredType::Invalid for constant buffers
            */
            StructuredType getStructuredType() const { return mStructuredType; }

            /** Get the variable count
            */
            size_t getVariableCount() const { return mVariables.size(); }

            /** Set a mask indicating in which shader stages the buffer is used
            */
            void setShaderMask(uint32_t mask) { mShaderMask = mask; }

            /** get a mask indicating in which shader stages the buffer is used
            */
            uint32_t getShaderMask() const { return mShaderMask; }

            /** Get the register index
            */
            uint32_t getRegisterIndex() const { return mRegIndex; }

            /** Get the register space
            */
            uint32_t getRegisterSpace() const { return mRegSpace; }

            /** Get the array size
            */
            uint32_t getArraySize() const { return mArraySize; }

            /** Get the shader access
            */
            ShaderAccess getShaderAccess() const { return mShaderAccess; }

        private:
            BufferReflection(const std::string& name, uint32_t regSpace, uint32_t baseRegIndex, uint32_t arraySize,Type type, StructuredType structuredType, size_t size, const VariableMap& varMap, const ResourceMap& resourceMap, ShaderAccess shaderAccess);
            std::string mName;
            size_t mSizeInBytes = 0;
            Type mType;
            StructuredType mStructuredType;
            ResourceMap mResources;
            VariableMap mVariables;
            uint32_t mShaderMask = 0;
            uint32_t mRegIndex;
            uint32_t mRegSpace = 0;
            uint32_t mArraySize = 0;
            ShaderAccess mShaderAccess;
        };

        /** Create a new object
        */
        static SharedPtr create(
            slang::ShaderReflection*    pSlangReflector,
            std::string&                log);

        /** Get a buffer binding index
        \param[in] name The buffer name in the program
        \return The bind location of the buffer if it is found, otherwise ProgramVersion#kInvalidLocation
        */
        BindLocation getBufferBinding(const std::string& name) const;

        using BufferMap = std::unordered_map<BindLocation, BufferReflection::SharedPtr, BindLocation>;

        /** Get a buffer list
        */
        const BufferMap& getBufferMap(BufferReflection::Type bufferType) const { return mBuffers[(uint32_t)bufferType].descMap; }

        /** Get a buffer descriptor
            \param[in] bindLocation The bindLocation of the requested buffer
            \return The buffer descriptor or nullptr, if the bind location isn't used
        */
        BufferReflection::SharedConstPtr getBufferDesc(uint32_t regSpace, uint32_t regIndex, ShaderAccess shaderAccess, BufferReflection::Type bufferType) const;

        /** Get a buffer descriptor
        \param[in] name The name of the requested buffer
        \return The buffer descriptor or nullptr, if the name doesn't exist
        */
        BufferReflection::SharedConstPtr getBufferDesc(const std::string& name, BufferReflection::Type bufferType) const;

        /** Get the descriptor for a vertex-attribute
            \param[in] name The attribute name in the program
            \return The variable desc of the attribute if it is found, otherwise nullptr
        */
        const Variable* getVertexAttribute(const std::string& name) const;

        /** Get the descriptor for a vertex-attribute with the given semantic
            \param[in] semantic The semantic name used in the program
            \return The variable desc of the attribute if it is found, otherwise nullptr
        */
        const Variable* getVertexAttributeBySemantic(const std::string& semantic) const;

        /** Get the descriptor for a fragment shader output
            \param[in] name The output variable name in the program
            \return The variable desc of the output if it is found, otherwise nullptr
        */
        const Variable* getFragmentOutput(const std::string& name) const;

        /** Get the descriptor for a shader resource
        \param[in] name The resource name in the program
        \return The resource desc of the output if it is found, otherwise nullptr
        */
        const Resource* getResourceDesc(const std::string& name) const;

        /** Get the resources map
        */

        const ResourceMap& getResourceMap() const { return mResources; }
        /** Helper struct that holds buffer-data
        */
        struct BufferData
        {
            BufferMap descMap;
            string_2_bindloc_map nameMap;
        };

        /** Get the thread group size. If the program is not a compute program, the return value is undefined
        */
        const uvec3& getThreadGroupSize() const { return mThreadGroupSize; }

        /** Check if the program has a pixel shader that is required to run per sample
        */
        bool isSampleFrequency() const { return mIsSampleFrequency; }

    // TODO(tfoley): switch this back
    public://private:
        bool init(
            slang::ShaderReflection*    pSlangReflector,
            std::string&                log);
        bool reflectVertexAttributes(
            slang::ShaderReflection*    pSlangReflector,
            std::string&                log);       // Input attributes
        bool reflectPixelShaderOutputs(
            slang::ShaderReflection*    pSlangReflector,
            std::string&                log);        // PS output (if PS exists)
        bool reflectResources(
            slang::ShaderReflection*    pSlangReflector,
            std::string&                log);              // SRV/UAV/ROV/Buffers and samplers

        BufferData mBuffers[BufferReflection::kTypeCount];
        VariableMap mFragOut;
        VariableMap mVertAttr;
        VariableMap mVertAttrBySemantic;
        ResourceMap mResources;
        uvec3 mThreadGroupSize;
        bool mIsSampleFrequency = false;
    };

    inline const std::string to_string(ProgramReflection::Variable::Type type)
    {
#define type_2_string(a) case ProgramReflection::Variable::Type::a: return #a;
        switch(type)
        {
            type_2_string(Unknown);
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
            type_2_string(GpuPtr);
        default:
            should_not_get_here();
            return "";
        }
#undef type_2_string
    }

    inline const std::string to_string(ProgramReflection::Resource::ResourceType type)
    {
#define type_2_string(a) case ProgramReflection::Resource::ResourceType::a: return #a;
        switch(type)
        {
            type_2_string(Unknown);
            type_2_string(Texture);
            type_2_string(StructuredBuffer);
            type_2_string(RawBuffer);
            type_2_string(Sampler);
            type_2_string(TypedBuffer);
        default:
            should_not_get_here();
            return "";
        }
#undef type_2_string
    }

    inline const std::string to_string(ProgramReflection::ShaderAccess access)
    {
#define access_2_string(a) case ProgramReflection::ShaderAccess::a: return #a;
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

    inline const std::string to_string(ProgramReflection::Resource::ReturnType retType)
    {
#define type_2_string(a) case ProgramReflection::Resource::ReturnType::a: return #a;
        switch(retType)
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

    inline const std::string to_string(ProgramReflection::Resource::Dimensions resource)
    {
#define type_2_string(a) case ProgramReflection::Resource::Dimensions::a: return #a;
        switch(resource)
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

    inline const std::string to_string(ProgramReflection::BufferReflection::Type type)
    {
#define type_2_string(a) case ProgramReflection::BufferReflection::Type::a: return #a;
        switch(type)
        {
            type_2_string(Constant);
            type_2_string(Structured);
        default:
            should_not_get_here();
            return "";
        }
#undef type_2_string
    }
}