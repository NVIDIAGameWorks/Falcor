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
#include "ResourceCache.h"

namespace Falcor
{
    ResourceCache::SharedPtr ResourceCache::create()
    {
        return SharedPtr(new ResourceCache());
    }

    void ResourceCache::reset()
    {
        mNameToIndex.clear();
        mResourceData.clear();
    }

    const std::shared_ptr<Resource>& ResourceCache::getResource(const std::string& name) const
    {
        static const std::shared_ptr<Resource> pNull;
        auto extIt = mExternalInputs.find(name);

        // Search external resources if not found in render graph resources
        if (extIt == mExternalInputs.end())
        {
            const auto& it = mNameToIndex.find(name);
            if (it == mNameToIndex.end())
            {
                return pNull;
            }

            return mResourceData[it->second].pResource;
        }

        return extIt->second;
    }

    const RenderPassReflection::Field& ResourceCache::getResourceReflection(const std::string& name) const
    {
        uint32_t i = mNameToIndex.at(name);
        return mResourceData[i].field;
    }

    void ResourceCache::registerExternalInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        mExternalInputs[name] = pResource;
    }

    void ResourceCache::removeExternalInput(const std::string& name)
    {
        auto it = mExternalInputs.find(name);
        if (it == mExternalInputs.end())
        {
            logWarning("ResourceCache::removeExternalResource: " + name + " does not exist.");
            return;
        }

        mExternalInputs.erase(it);
    }

    static ResourceBindFlags getBindFlagsFromFormat(Resource::BindFlags flags, ResourceFormat format, RenderPassReflection::Field::Visibility vis)
    {
        auto mask = Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource;
        if (is_set(vis, RenderPassReflection::Field::Visibility::Output)) mask |= Resource::BindFlags::DepthStencil | Resource::BindFlags::RenderTarget;

        auto formatFlags = getFormatBindFlags(format);
        formatFlags = formatFlags & mask;
        return flags | formatFlags;
    }

    /** Overwrite previously unknown/unspecified fields with specified ones.

        If a field property is specified both in the existing cache, as well as the input properties,
        a warning will be logged and the cached properties will not be changed.
    */
    bool mergeFields(RenderPassReflection::Field& base, const RenderPassReflection::Field& newField, const std::string& newFieldName)
    {
        auto warn = [&](const std::string& msg) -> bool
        {
            const std::string warningMsg = "Can't merge RenderPassReflection::Fields. base(" + base.getName() + "), newField(" + newField.getName() + "). ";
            logWarning(warningMsg + msg);
            return false;
        };

        if (base.getType() != newField.getType()) return warn("mismatching types");

        // Default to base dimension
        // If newField property is not 0, retrieve value from newField
        // If both newField and base property is specified, generate warning.
#define get_dim(var, dim) \
    var = base.get##dim(); \
    if (newField.get##dim() != 0) { \
        if (base.get##dim() == 0) var = newField.get##dim(); \
        else if(base.get##dim() != newField.get##dim()) return warn(std::string(#dim) + " already specified. "); }

        uint32_t w = 0, h = 0, d = 0;
        get_dim(w, Width);
        get_dim(h, Height);
        get_dim(d, Depth);
#undef get_dim

        // Merge sample counts
        uint32_t sampleCount = base.getSampleCount();
        if (newField.getSampleCount() != 0)
        {
            // if base field doesn't have anything specified, apply new property
            if (base.getSampleCount() == 0)
            {
                sampleCount = newField.getSampleCount();
            }
            // if they differ in any other way, that is invalid
            else if (base.getSampleCount() != newField.getSampleCount())
            {
                return warn("Cannot merge sample count. Base has " + std::to_string(base.getSampleCount()) + ", new-field has " + std::to_string(newField.getSampleCount()));
            }
        }

        switch (base.getType())
        {
        case RenderPassReflection::Field::Type::Texture1D:
            base.texture1D(w);
            break;
        case RenderPassReflection::Field::Type::Texture2D:
            base.texture2D(w, h, sampleCount);
            break;
        case RenderPassReflection::Field::Type::Texture3D:
            base.texture3D(w, h, d);
            break;
        case RenderPassReflection::Field::Type::TextureCube:
            base.textureCube(w, h);
        default:
            should_not_get_here();
            break;
        }

        // merge array-size
        if (newField.getArraySize() != 0)
        {
            if (base.getArraySize() == 0) base.arraySize(newField.getArraySize());
            else if (base.getArraySize() != newField.getArraySize()) return warn("Mismatching array-sizes");
        }

        // merge mip-levels
        if (newField.getMipLevels() != 0)
        {
            if (base.getMipLevels() == 0) base.mipLevels(newField.getMipLevels());
            else if (base.getMipLevels() != newField.getMipLevels()) return warn("Mismatching mip-levels");
        }

        // Format
        if (newField.getFormat() != ResourceFormat::Unknown)
        {
            if (base.getFormat() == ResourceFormat::Unknown) base.format(newField.getFormat());
            else if (base.getFormat() != newField.getFormat()) return warn("Format already specified");
        }

        // Visibility
        assert(is_set(newField.getVisibility(), RenderPassReflection::Field::Visibility::Internal) == false); // We can't alias/merge internal fields
        assert(is_set(base.getVisibility(), RenderPassReflection::Field::Visibility::Internal) == false); // We can't alias/merge internal fields
        base.visibility(base.getVisibility() | newField.getVisibility());

        Resource::BindFlags baseFlags = base.getBindFlags();
        Resource::BindFlags newFlags = newField.getBindFlags();

        if ((is_set(baseFlags, Resource::BindFlags::RenderTarget) && is_set(newFlags, Resource::BindFlags::DepthStencil)) ||
            (is_set(baseFlags, Resource::BindFlags::DepthStencil) && is_set(newFlags, Resource::BindFlags::RenderTarget)))
        {
            return warn("Usage contained both RenderTarget and DepthStencil bind flags");
        }
        else
        {
            base.bindFlags(baseFlags | newFlags);
        }
        
        return true;
    }

    void mergeTimePoint(uint32_t& minTime, uint32_t& maxTime, uint32_t newTime)
    {
        minTime = min(minTime, newTime);
        maxTime = max(maxTime, newTime);
    }

    void ResourceCache::registerField(const std::string& name, RenderPassReflection::Field field, uint32_t timePoint, const std::string& alias)
    {
        auto nameIt = mNameToIndex.find(name);
        auto aliasIt = mNameToIndex.find(alias);

        // If two fields were registered separately before, but are now aliased together, merge the fields, with alias field being the base
        if ((nameIt != mNameToIndex.end()) && (aliasIt != mNameToIndex.end()) && (nameIt->second != aliasIt->second))
        {
            // Merge data
            auto& baseResData = mResourceData[aliasIt->second];
            auto& newResData = mResourceData[nameIt->second];
            mergeFields(baseResData.field, newResData.field, name);
            mergeTimePoint(baseResData.firstUsed, baseResData.lastUsed, newResData.firstUsed);
            mergeTimePoint(baseResData.firstUsed, baseResData.lastUsed, newResData.lastUsed);

            // Clear data that has been merged
            mResourceData[nameIt->second] = ResourceData();

            // Redirect 'name' to look up the alias field
            nameIt->second = aliasIt->second;
        }

        // If name exists, update time range
        if (mNameToIndex.count(name) > 0)
        {
            uint32_t index = mNameToIndex[name];
            mergeTimePoint(mResourceData[index].firstUsed, mResourceData[index].lastUsed, timePoint);
            return;
        }

        bool addAlias = (alias.empty() == false);
        if (addAlias && mNameToIndex.count(alias) == 0)
        {
            logWarning("ResourceCache::registerField: Field named " + alias + " not found. Cannot add " + name + "as an alias. Creating new entry.");
            addAlias = false;
        }

        // Add a new field
        if (addAlias == false)
        {
            assert(mNameToIndex.count(name) == 0);
            mNameToIndex[name] = (uint32_t)mResourceData.size();
            mResourceData.push_back({ field, true, timePoint, timePoint, nullptr });
        }
        // Add alias
        else
        {
            uint32_t index = mNameToIndex[alias];
            mNameToIndex[name] = index;

            mergeFields(mResourceData[index].field, field, name);
            mergeTimePoint(mResourceData[index].firstUsed, mResourceData[index].lastUsed, timePoint);

            mResourceData[index].dirty = true;
        }
    }

    Texture::SharedPtr createTextureForPass(const ResourceCache::DefaultProperties& params, const RenderPassReflection::Field& field)
    {
        uint32_t width = field.getWidth() ? field.getWidth() : params.width;
        uint32_t height = field.getHeight() ? field.getHeight() : params.height;
        uint32_t depth = field.getDepth() ? field.getDepth() : 1;
        uint32_t sampleCount = field.getSampleCount() ? field.getSampleCount() : 1;
        ResourceFormat format = field.getFormat() == ResourceFormat::Unknown ? params.format : field.getFormat();

        auto bindFlags = getBindFlagsFromFormat(field.getBindFlags(), format, field.getVisibility());

        Texture::SharedPtr pTexture;
        if (depth > 1)
        {
            assert(sampleCount == 1);
            pTexture = Texture::create3D(width, height, depth, format, 1, nullptr, bindFlags);
        }
        else if (height > 1 || sampleCount > 1)
        {
            if (sampleCount > 1)
            {
                pTexture = Texture::create2DMS(width, height, format, sampleCount, 1, bindFlags);
            }
            else
            {
                pTexture = Texture::create2D(width, height, format, 1, 1, nullptr, bindFlags);
            }
        }
        else
        {
            pTexture = Texture::create1D(width, format, 1, 1, nullptr, bindFlags);
        }

        return pTexture;
    }

    void ResourceCache::allocateResources(const DefaultProperties& params)
    {
        for (auto& data : mResourceData)
        {
            if ((data.pResource == nullptr || data.dirty) && data.field.isValid())
            {
                data.pResource = createTextureForPass(params, data.field);
                data.dirty = false;
            }
        }
    }
}
