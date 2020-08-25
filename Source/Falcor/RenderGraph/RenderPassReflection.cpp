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
#include "stdafx.h"
#include "RenderPassReflection.h"

namespace Falcor
{
    RenderPassReflection::Field::Field(const std::string& name, const std::string& desc, Visibility v) : mName(name), mVisibility(v), mDesc(desc)
    {
    }

    RenderPassReflection::Field& RenderPassReflection::Field::rawBuffer(uint32_t size)
    {
        mType = Type::RawBuffer;
        mWidth = size;
        mHeight = mDepth = mArraySize = mMipCount = 0;
        return *this;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::texture1D(uint32_t width, uint32_t mipCount, uint32_t arraySize)
    {
        mType = Type::Texture1D;
        mWidth = width;
        mHeight = 1;
        mDepth = 1;
        mArraySize = arraySize;
        mMipCount = mipCount;
        return *this;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::texture2D(uint32_t width, uint32_t height, uint32_t sampleCount, uint32_t mipCount, uint32_t arraySize)
    {
        mType = Type::Texture2D;
        mWidth = width;
        mHeight = height;
        mSampleCount = sampleCount;
        mDepth = 1;
        mArraySize = arraySize;
        mMipCount = mipCount;
        return *this;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::texture3D(uint32_t width, uint32_t height, uint32_t depth, uint32_t arraySize)
    {
        mType = Type::Texture3D;
        mWidth = width;
        mHeight = height;
        mDepth = depth;
        mArraySize = arraySize;
        return *this;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::textureCube(uint32_t width, uint32_t height, uint32_t mipCount, uint32_t arraySize)
    {
        mType = Type::TextureCube;
        mWidth = width;
        mHeight = height;
        mDepth = 1;
        mMipCount = mipCount;
        mArraySize = arraySize;
        return *this;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::resourceType(RenderPassReflection::Field::Type type, uint32_t width, uint32_t height, uint32_t depth, uint32_t sampleCount, uint32_t mipCount, uint32_t arraySize)
    {
        switch (type)
        {
        case RenderPassReflection::Field::Type::RawBuffer:
            if(height > 0 || depth > 0 || sampleCount > 0) logWarning("RenderPassReflection::Field::resourceType - height, depth, sampleCount for " + to_string(type) + " must be either 0");
            return rawBuffer(width);
        case RenderPassReflection::Field::Type::Texture1D:
            if (height > 1 || depth > 1 || sampleCount > 1) logWarning("RenderPassReflection::Field::resourceType - height, depth, sampleCount for " + to_string(type) + " must be either 0 or 1");
            return texture1D(width, mipCount, arraySize);
        case RenderPassReflection::Field::Type::Texture2D:
            if (depth > 1) logWarning("RenderPassReflection::Field::resourceType - depth for " + to_string(type) + " must be either 0 or 1");
            return texture2D(width, height, sampleCount, mipCount, arraySize);
        case RenderPassReflection::Field::Type::Texture3D:
            if (sampleCount > 1 || mipCount > 1) logWarning("RenderPassReflection::Field::resourceType - sampleCount, mipCount for " + to_string(type) + " must be either 0 or 1");
            return texture3D(width, height, depth, arraySize);
        case RenderPassReflection::Field::Type::TextureCube:
            if (depth > 1 || sampleCount > 1) logWarning("RenderPassReflection::Field::resourceType - depth, sampleCount for " + to_string(type) + " must be either 0 or 1");
            return textureCube(width, height, mipCount, arraySize);
        default:
            throw std::runtime_error("RenderPassReflection::Field::resourceType - " + to_string(type) + " is not a valid Field type");
        }
    }


    RenderPassReflection::Field& RenderPassReflection::Field::format(ResourceFormat f) { mFormat = f; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::bindFlags(Resource::BindFlags flags) { mBindFlags = flags; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::flags(Flags flags) { mFlags = flags; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::visibility(Visibility vis) { mVisibility = vis; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::name(const std::string& name) { mName = name; return *this; }
    RenderPassReflection::Field& RenderPassReflection::Field::desc(const std::string& desc) { mDesc = desc; return *this; }

    bool RenderPassReflection::Field::isValid() const
    {
        if (mSampleCount > 1 && mMipCount > 1)
        {
            logError("Trying to create a multisampled RenderPassReflection::Field '" + mName + "' with mip-count larger than 1. This is illegal.");
            return false;
        }

        if (is_set(mVisibility, Visibility::Internal) && is_set(mFlags, Flags::Optional))
        {
            logError("Internal resource can't be optional, since there will never be a graph edge that forces their creation");
            return false;
        }

        return true;
    }

    RenderPassReflection::Field& RenderPassReflection::addField(const Field& field)
    {
        // See if the field already exists
        for (auto& existingF : mFields)
        {
            if (existingF.getName() == field.getName())
            {
                // We can only merge input and output fields, otherwise override the previous field
                bool ioField = is_set(existingF.getVisibility(), Field::Visibility::Input | Field::Visibility::Output);
                bool ioRequest = is_set(field.getVisibility(), Field::Visibility::Input | Field::Visibility::Output);
                if (ioField && ioRequest)
                {
                    existingF.mVisibility |= field.getVisibility();
                }
                else if ((existingF.getVisibility() & field.getVisibility()) != field.getVisibility())
                {
                    logError("Trying to add an existing field '" + field.getName() + "' to RenderPassReflection, but the visibility flags mismatch. Overriding the previous definition");
                }
                return existingF;
            }
        }

        mFields.push_back(field);
        return mFields.back();
    }

    RenderPassReflection::Field& RenderPassReflection::addField(const std::string& name, const std::string& desc, Field::Visibility visibility)
    {
        return addField(Field(name, desc, visibility));
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

    const RenderPassReflection::Field* RenderPassReflection::getField(const std::string& name) const
    {
        for (const auto& field : mFields)
        {
            if (field.getName() == name) return &field;
        }
        return nullptr;
    }

    RenderPassReflection::Field& RenderPassReflection::Field::merge(const RenderPassReflection::Field& other)
    {
        auto err = [&](const std::string& msg)
        {
            const std::string s = "Can't merge RenderPassReflection::Fields. base(" + getName() + "), newField(" + other.getName() + "). ";
            throw std::exception((s + msg).c_str());
        };

        if (mType != other.mType) err("mismatching types");

        // Default to base dimension
        // If newField property is not 0, retrieve value from newField
        // If both newField and base property is specified, generate warning.
        auto mf = [err](auto& mine, const auto& other, const std::string& name)
        {
            auto none = std::remove_reference_t<decltype(mine)>(0);
            if (other != none)
            {
                if (mine == none) mine = other;
                else if (mine != other) err(name + " already specified with a mismatching value in a different pass");
            }
        };

#define merge_field(f) mf(m##f, other.m##f, #f)
        merge_field(Width);
        merge_field(Height);
        merge_field(Depth);
        merge_field(ArraySize);
        merge_field(MipCount);
        merge_field(SampleCount);
        merge_field(Format);
#undef merge_field

        assert(is_set(other.mVisibility, RenderPassReflection::Field::Visibility::Internal) == false); // We can't alias/merge internal fields
        assert(is_set(mVisibility, RenderPassReflection::Field::Visibility::Internal) == false); // We can't alias/merge internal fields
        mVisibility = mVisibility | other.mVisibility;
        mBindFlags = mBindFlags | other.mBindFlags;
        return *this;
    }

    bool RenderPassReflection::Field::operator==(const Field& other) const
    {
#define check(_f) if(_f != other._f) return false

        check(mType);
        check(mName);
        check(mDesc);
        check(mWidth);
        check(mHeight);
        check(mDepth);
        check(mSampleCount);
        check(mMipCount);
        check(mArraySize);
        check(mFormat);
        check(mBindFlags);
        check(mFlags);
        check(mVisibility);
#undef check
        return true;
    }

    bool RenderPassReflection::operator==(const RenderPassReflection& other) const
    {
        if (other.mFields.size() != mFields.size()) return false;
        auto findField = [](const std::string& name, const auto& fields) -> std::optional<Field>
        {
            for (auto& f : fields) if (f.mName == name) return f;
            return {};
        };

        for (const auto& f : mFields)
        {
            auto otherF = findField(f.mName, other.mFields);
            if (!otherF) return false;
            if (otherF.value() != f) return false;
        }

        return true;
    }
}
