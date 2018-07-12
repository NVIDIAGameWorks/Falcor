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
        class Field
        {
        public:
            enum class Type
            {
                Input,
                Output,
                Inout
            };

            enum class Flags
            {
                None = 0x0,
                Optional = 0x1,
                Persistent = 0x2,
            };

            Field(const std::string& name, Type type);
            Field& setResourceType(const ReflectionResourceType::SharedConstPtr& pType) { mpType = pType; return *this; }
            Field& setDimensions(uint32_t w, uint32_t h, uint32_t d) { mWidth = w; mHeight = h; mDepth = d; return *this; }
            Field& setSampleCount(uint32_t count) { mSampleCount = count; return *this; }
            Field& setFormat(ResourceFormat format) { mFormat = format; return *this; }
            Field& setBindFlags(Resource::BindFlags flags) { mBindFlags = flags; return *this; }
            Field& setFlags(Flags flags) { mFlags = flags; return *this; }

            const std::string& getName() const { return mName; }
            const ReflectionResourceType::SharedConstPtr& getReflectionType() const { return mpType; }
            uint32_t getWidth() const { return mWidth; }
            uint32_t getHeight() const { return mHeight; }
            uint32_t getDepth() const { return mDepth; }
            uint32_t getSampleCount() const { return mSampleCount; }
            ResourceFormat getFormat() const { return mFormat; }
            Resource::BindFlags getBindFlags() const { return mBindFlags; }
            Flags getFlags() const { return mFlags; }
            Type getType() const { return mType; }
        private:
            std::string mName;                             ///< The field's name
            ReflectionResourceType::SharedConstPtr mpType; ///< The resource type
            uint32_t mWidth = 0;                           ///< For output resources, 0 means use the window size(textures) or the size in bytes (buffers). For input resources 0 means don't care
            uint32_t mHeight = 0;                          ///< For output resources, 0 means use the window size. For input resources 0 means don't care
            uint32_t mDepth = 0;                           ///< For output resources, 0 means use the window size. For input resources 0 means don't care
            uint32_t mSampleCount = 0;                     ///< 0 means don't care (which means 1 for output resources)
            ResourceFormat mFormat = ResourceFormat::Unknown; ///< Unknown means use the back-buffer format for output resources, don't care for input resources
            Resource::BindFlags mBindFlags = Resource::BindFlags::None;  ///< The required bind flags
            Flags mFlags = Flags::None;
            Type mType;
        };

        Field& addField(const std::string& name, Field::Type type);
        size_t getFieldCount() const { return mFields.size(); }
        const Field& getField(size_t f) const { return mFields[f]; }
    private:
        std::vector<Field> mFields;
    };

    enum_class_operators(RenderPassReflection::Field::Flags);
}