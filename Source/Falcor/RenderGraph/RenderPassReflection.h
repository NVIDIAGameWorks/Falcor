/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

namespace Falcor
{
    class dlldecl RenderPassReflection
    {
    public:
        class dlldecl Field
        {
        public:
            /** The type of visibility the field has
            */
            enum class Visibility
            {
                Undefined = 0x0,
                Input = 0x1,  ///< Input field
                Output = 0x2,  ///< Output field
                Internal = 0x4,  ///< Internal field. You can use this value to ask the resource-cache for any required internal resource
            };

            /** Miscellaneous flags to control allocation and lifetime policy
            */
            enum class Flags
            {
                None = 0x0,         ///< None
                Optional = 0x1,     ///< Mark that field as optional. For output resources, it means that they don't have to be bound unless their result is required by the caller. For input resources, it means that the pass can function correctly without them being bound (but the behavior might be different)
                Persistent = 0x2,   ///< The resource bound to this field must not change between execute() calls (not the pointer nor the data). It can change only during the RenderGraph recompilation.
            };

            /** Field type
            */
            enum class Type
            {
                Texture1D,
                Texture2D,
                Texture3D,
                TextureCube,
                RawBuffer,
            };
            
            Field(const std::string& name, const std::string& desc, Visibility v);
            Field() = default;

            bool isValid() const;

            /** If the `mipLevel` argument to any of the `texture*` functions is set to `kMaxMipLevels`, it will generate the entire mip-chain based on the texture dimensions
            */
            static const uint32_t kMaxMipLevels = Texture::kMaxPossible;

            Field& rawBuffer(uint32_t size);
            Field& texture1D(uint32_t width = 0, uint32_t mipCount = 1, uint32_t arraySize = 1);
            Field& texture2D(uint32_t width = 0, uint32_t height = 0, uint32_t sampleCount = 1, uint32_t mipCount = 1, uint32_t arraySize = 1);
            Field& texture3D(uint32_t width = 0, uint32_t height = 0, uint32_t depth = 0, uint32_t arraySize = 1);
            Field& textureCube(uint32_t width = 0, uint32_t height = 0, uint32_t mipCount = 1, uint32_t arraySize = 1);
            Field& resourceType(Type type, uint32_t width, uint32_t height, uint32_t depth, uint32_t sampleCount, uint32_t mipCount, uint32_t arraySize);

            Field& format(ResourceFormat f);
            Field& bindFlags(ResourceBindFlags flags);
            Field& flags(Flags flags);
            Field& visibility(Visibility vis);
            Field& name(const std::string& name);
            Field& desc(const std::string& desc);

            const std::string& getName() const { return mName; }
            const std::string& getDesc() const { return mDesc; }
            uint32_t getWidth() const { return mWidth; }
            uint32_t getHeight() const { return mHeight; }
            uint32_t getDepth() const { return mDepth; }
            uint32_t getSampleCount() const { return mSampleCount; }
            uint32_t getArraySize() const { return mArraySize; }
            uint32_t getMipCount() const { return mMipCount; }
            ResourceFormat getFormat() const { return mFormat; }
            ResourceBindFlags getBindFlags() const { return mBindFlags; }
            Flags getFlags() const { return mFlags; }
            Type getType() const { return mType; }
            Visibility getVisibility() const { return mVisibility; }

            /** Overwrite previously unknown/unspecified fields with specified ones.
                If a property is specified both in the current object, as well as the other field, an error will be logged and the current field will be undefined
            */
            Field& merge(const Field& other);

            bool operator==(const Field& other) const;
            bool operator!=(const Field& other) const { return !(other == *this); }
        private:
            friend class RenderPassReflection;

            Type mType = Type::Texture2D;
            std::string mName;                             ///< The field's name
            std::string mDesc;                             ///< A description of the field
            uint32_t mWidth = 0;                           ///< For texture, the width. For buffers, the size in bytes. 0 means don't care - the pass will use whatever is bound (the RenderGraph will use the window size if this field is 0)
            uint32_t mHeight = 0;                          ///< 0 means don't care - the pass will use whatever is bound (the RenderGraph will use the window size if this field is 0)
            uint32_t mDepth = 0;                           ///< 0 means don't care - the pass will use whatever is bound (the RenderGraph will use the window size if this field is 0)
            uint32_t mSampleCount = 1;                     ///< 0 means don't care - the pass will use whatever is bound
            uint32_t mMipCount = 1;                        ///< The required mip-level count. Only valid for textures
            uint32_t mArraySize = 1;                       ///< The required array-size. Only valid for textures
            ResourceFormat mFormat = ResourceFormat::Unknown; ///< Unknown means use the back-buffer format for output resources, don't care for input resources
            ResourceBindFlags mBindFlags = ResourceBindFlags::None;  ///< The required bind flags. The default for outputs is RenderTarget, for inputs is ShaderResource and for InOut (RenderTarget | ShaderResource)
            Flags mFlags = Flags::None;                    ///< The field flags
            Visibility mVisibility = Visibility::Undefined;
        };

        Field& addInput(const std::string& name, const std::string& desc);
        Field& addOutput(const std::string& name, const std::string& desc);
        Field& addInputOutput(const std::string& name, const std::string& desc);
        Field& addInternal(const std::string& name, const std::string& desc);

        size_t getFieldCount() const { return mFields.size(); }
        const Field* getField(size_t f) const { return f <= mFields.size() ? &mFields[f] : nullptr; }
        const Field* getField(const std::string& name) const;
        Field& addField(const Field& field);

        bool operator==(const RenderPassReflection& other) const;
        bool operator!=(const RenderPassReflection& other) const { return !(*this == other); }

    private:
        Field& addField(const std::string& name, const std::string& desc, Field::Visibility visibility);
        std::vector<Field> mFields;
    };

    enum_class_operators(RenderPassReflection::Field::Visibility);
    enum_class_operators(RenderPassReflection::Field::Flags);

    inline std::string to_string(RenderPassReflection::Field::Type t)
    {
#define t2s(ft) case RenderPassReflection::Field::Type::ft: return #ft;
        switch (t)
        {
            t2s(RawBuffer);
            t2s(Texture1D);
            t2s(Texture2D);
            t2s(Texture3D);
            t2s(TextureCube);
        default:
            should_not_get_here();
            return "";
        }
#undef t2s
    }

    inline RenderPassReflection::Field::Type resourceTypeToFieldType(Resource::Type t)
    {
        switch (t)
        {
        case Resource::Type::Texture1D:
            return RenderPassReflection::Field::Type::Texture1D;
        case Resource::Type::Texture2D:
        case Resource::Type::Texture2DMultisample:
            return RenderPassReflection::Field::Type::Texture2D;
        case Resource::Type::Texture3D:
            return RenderPassReflection::Field::Type::Texture3D;
        case Resource::Type::TextureCube:
            return RenderPassReflection::Field::Type::TextureCube;
        default:
            throw std::runtime_error("resourceTypeToFieldType - No RenderPassReflection::Field::Type exists for Resource::Type::" + to_string(t));
        }
    }
}
