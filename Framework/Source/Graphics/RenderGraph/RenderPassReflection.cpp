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
#include "Framework.h"
#include "RenderPassReflection.h"

namespace Falcor
{
    RenderPassReflection::Field::Field(const std::string& name, const std::string& desc, Visibility v) : mName(name), mVisibility(v), mDesc(desc)
    {
    }

    RenderPassReflection::Field& RenderPassReflection::Field::texture1D(uint32_t width)
    {
        mType = Type::Texture1D;
        mWidth = width;
        mHeight = 1;
        mDepth = 1;
        return *this;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::texture2D(uint32_t width, uint32_t height, uint32_t sampleCount)
    {
        mType = Type::Texture2D;
        mWidth = width;
        mHeight = height;
        mSampleCount = sampleCount;
        mDepth = 1;
        return *this;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::texture3D(uint32_t width, uint32_t height, uint32_t depth)
    {
        mType = Type::Texture3D;
        mWidth = width;
        mHeight = height;
        mDepth = depth;
        return *this;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::textureCube(uint32_t width, uint32_t height)
    {
        mType = Type::TextureCube;
        mWidth = width;
        mHeight = height;
        mDepth = 1;
        return *this;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::arraySize(uint32_t a) { mArraySize = a; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::mipLevels(uint32_t m) { mMipLevels = m; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::format(ResourceFormat f) { mFormat = f; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::bindFlags(Resource::BindFlags flags) { mBindFlags = flags; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::flags(Flags flags) { mFlags = flags; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::visibility(Visibility vis) { mVisibility = vis; return *this; }

    bool RenderPassReflection::Field::isValid() const
    {
        if (mType == Type::Texture3D && mArraySize > 1)
        {
            logWarning("Trying to create a Texture3D RenderPassReflection::Field `" + mName + "` with array-size larger than 1. This is illegal.");
            return false;
        }

        if (mSampleCount > 1 && mMipLevels > 1)
        {
            logWarning("Trying to create a multisampled RenderPassReflection::Field `" + mName + "` with mip-count larger than 1. This is illegal.");
            return false;
        }

        return true;
    }

    RenderPassReflection::Field& RenderPassReflection::addField(const std::string& name, const std::string& desc, Field::Visibility visibility)
    {
        // See if the field already exists
        for (auto& f : mFields)
        {
            if (f.getName() == name)
            {
                // We can only merge input and output fields, otherwise override the previous field
                bool ioField = is_set(f.getVisibility(), Field::Visibility::Input | Field::Visibility::Output);
                bool ioRequest = is_set(visibility, Field::Visibility::Input | Field::Visibility::Output);
                if (ioField && ioRequest)
                {
                    f.mVisibility |= visibility;
                }
                else if((f.getVisibility() & visibility) != visibility)
                {
                    logError("Trying to add an existing field `" + name + "` to RenderPassReflection, but the visibility flags mismatch. Overriding the previous definition");
                }
                return f;
            }
        }

        mFields.push_back(Field(name, desc, visibility));
        return mFields.back();
    }

    RenderPassReflection::Field& RenderPassReflection::addInput(const std::string& name, const std::string& desc)
    {
        return addField(name, desc, Field::Visibility::Input);
    }

    RenderPassReflection::Field& RenderPassReflection::addOutput(const std::string& name, const std::string& desc)
    {
        return addField(name, desc, Field::Visibility::Output);
    }

    RenderPassReflection::Field& RenderPassReflection::addInputOutput(const std::string& name, const std::string& desc)
    {
        return addField(name, desc, Field::Visibility::Input | Field::Visibility::Output);
    }

    RenderPassReflection::Field& RenderPassReflection::addInternal(const std::string& name, const std::string& desc)
    {
        return addField(name, desc, Field::Visibility::Internal);
    }

    const RenderPassReflection::Field& RenderPassReflection::getField(const std::string& name) const
    {
         for (const auto& field : mFields)
        {
            if (field.getName() == name) return field;
        }
        std::string error = "Can't find a field named `" + name + "` in RenderPassReflection";
        throw std::runtime_error(error.c_str());
    }

}