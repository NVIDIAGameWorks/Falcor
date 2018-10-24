/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "Graphics/Program/ProgramReflection.h"

namespace Falcor
{
    class RenderPassReflection
    {
    public:
        /** Global render-pass flags
        */
        enum class Flags
        {
            None = 0x0,
            ForceExecution = 0x1,  ///< Will force the execution of the pass even if no edges are connected to it
        };

        class Field
        {
        public:
            enum class Type
            {
                None        = 0x0,
                Input       = 0x1,  // Input field
                Output      = 0x2,  // Output field
                Internal    = 0x4,  // Internal field. You can use this value to ask the resource-cache for any required internal resource
            };

            enum class Flags
            {
                None = 0x0,         ///< None
                Optional = 0x1,     ///< Mark that field as optional. For output resources, it means that they don't have to be bound unless their result is required by the caller. For input resources, it means that the pass can function correctly without them being bound (but the behavior might be different)
                Persistent = 0x2,   ///< The resource bound to this field must not change between execute() calls (not the pointer nor the data). It can change only during the RenderGraph recompilation.
            };

            Field();
            Field(const std::string& name, Type type);

            bool isValid() const;

            Field& setResourceType(const ReflectionResourceType::SharedConstPtr& pType) { mpType = pType; return *this; }
            Field& setDimensions(uint32_t w, uint32_t h, uint32_t d) { mWidth = w; mHeight = h; mDepth = d; return *this; }
            Field& setSampleCount(uint32_t count) { mSampleCount = count; return *this; }
            Field& setFormat(ResourceFormat format) { mFormat = format; return *this; }
            Field& setBindFlags(Resource::BindFlags flags) { mBindFlags = flags; return *this; }
            Field& setFlags(Flags flags) { mFlags = flags; return *this; }
            Field& setArraySize(uint32_t arraySize) { mArraySize = arraySize; return *this; }

            const std::string& getName() const { return mName; }
            const ReflectionResourceType::SharedConstPtr& getResourceType() const { return mpType; }
            uint32_t getWidth() const { return mWidth; }
            uint32_t getHeight() const { return mHeight; }
            uint32_t getDepth() const { return mDepth; }
            uint32_t getSampleCount() const { return mSampleCount; }
            uint32_t getArraySize() const { return mArraySize; }
            ResourceFormat getFormat() const { return mFormat; }
            Resource::BindFlags getBindFlags() const { return mBindFlags; }
            Flags getFlags() const { return mFlags; }
            Type getType() const { return mType; }

        private:
            static const ReflectionResourceType::SharedPtr kpTex2DType;

            std::string mName;                             ///< The field's name
            ReflectionResourceType::SharedConstPtr mpType = kpTex2DType; ///< The resource type. The default is a 2D texture
            uint32_t mWidth = 0;                           ///< For texture, the width. For buffers, the size in bytes. 0 means don't care - the pass will use whatever is bound (the RenderGraph will use the window size if this field is 0)
            uint32_t mHeight = 0;                          ///< 0 means don't care - the pass will use whatever is bound (the RenderGraph will use the window size if this field is 0)
            uint32_t mDepth = 0;                           ///< 0 means don't care - the pass will use whatever is bound (the RenderGraph will use the window size if this field is 0)
            uint32_t mSampleCount = 1;                     ///< 0 means don't care - the pass will use whatever is bound
            uint32_t mMipLevels = 1;                       ///< The required mip-level count. Only valid for textures
            uint32_t mArraySize = 1;                       ///< The required array-size. Only valid for textures
            ResourceFormat mFormat = ResourceFormat::Unknown; ///< Unknown means use the back-buffer format for output resources, don't care for input resources
            Resource::BindFlags mBindFlags = Resource::BindFlags::None;  ///< The required bind flags. The default for outputs is RenderTarget, for inputs is ShaderResource and for InOut (RenderTarget | ShaderResource)
            Flags mFlags = Flags::None;                    ///< The field flags
            Type mType;
        };

        Field& addInput(const std::string& name);
        Field& addOutput(const std::string& name);
        Field& addInputOutput(const std::string& name);
        Field& addInternal(const std::string& name);
        RenderPassReflection& setFlags(RenderPassReflection::Flags flags) { mFlags = flags; return *this; }

        size_t getFieldCount() const { return mFields.size(); }
        const Field& getField(size_t f) const { return mFields[f]; }
        const Field& getField(const std::string& name, Field::Type type = Field::Type::None) const;
        Flags getFlags() const { return mFlags; }
    private:
        Field& addField(const std::string& name, Field::Type type);
        RenderPassReflection::Flags mFlags = Flags::None;
        std::vector<Field> mFields;
    };

    enum_class_operators(RenderPassReflection::Field::Type);
    enum_class_operators(RenderPassReflection::Field::Flags);
    enum_class_operators(RenderPassReflection::Flags);
}