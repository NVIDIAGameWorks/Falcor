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

    /** Overwrite previously unknown/unspecified fields with specified ones.

        If a field property is specified both in the existing cache, as well as the input properties,
        a warning will be logged and the cached properties will not be changed.
    */
    bool mergeFields(RenderPassReflection::Field& base, const RenderPassReflection::Field& newField, const std::string& newFieldName)
    {
        std::string warningMsg;

        // Default to base dimension
        // If newField property is not 0, retrieve value from newField
        // If both newField and base property is specified, generate warning.
#define get_dim(var, dim) \
    var = base.get##dim(); \
    if (newField.get##dim() != 0) { \
        if (base.get##dim() == 0) var = newField.get##dim(); \
        else if(base.get##dim() != newField.get##dim()) warningMsg += " " + std::string(#dim) + " already specified. "; }

        uint32_t w = 0, h = 0, d = 0;
        get_dim(w, Width);
        get_dim(h, Height);
        get_dim(d, Depth);
        base.setDimensions(w, h, d);
#undef get_dim

        if (newField.getFormat() != ResourceFormat::Unknown)
        {
            if (base.getFormat() == ResourceFormat::Unknown) base.setFormat(newField.getFormat());
            else if (base.getFormat() != newField.getFormat()) warningMsg += " Format already specified. ";
        }

        // Merge sample counts
        if (newField.getSampleCount() != 0)
        {
            // if base field doesn't have anything specified, apply new property
            if (base.getSampleCount() == 0)
            {
                base.setSampleCount(newField.getSampleCount());
            }
            // if they differ in any other way, that is invalid
            else if (base.getSampleCount() != newField.getSampleCount())
            {
                warningMsg += " Cannot merge sample count specified by " + newFieldName + ". ";
                warningMsg += newFieldName + " has sample count " + std::to_string(newField.getSampleCount()) + ", existing merged properties has sample count " + std::to_string(base.getSampleCount());
            }
        }

        Resource::BindFlags baseFlags = base.getBindFlags();
        Resource::BindFlags newFlags = newField.getBindFlags();

        if ((is_set(baseFlags, Resource::BindFlags::RenderTarget) && is_set(newFlags, Resource::BindFlags::DepthStencil)) ||
            (is_set(baseFlags, Resource::BindFlags::DepthStencil) && is_set(newFlags, Resource::BindFlags::RenderTarget)))
        {
            warningMsg += " Usage contained both RenderTarget and DepthStencil bind flags.";
            return false;
        }
        else
        {
            base.setBindFlags(baseFlags | newFlags);
        }

        if (warningMsg.empty() == false)
        {
            logWarning("ResourceCache: Cannot merge field " + newFieldName + ":" + warningMsg);
            return false;
        }

        return true;
    }

    void mergeTimePoint(uint32_t& minTime, uint32_t& maxTime, uint32_t newTime)
    {
        minTime = min(minTime, newTime);
        maxTime = max(maxTime, newTime);
    }

    void ResourceCache::registerField(const std::string& name, const RenderPassReflection::Field& field, uint32_t timePoint, const std::string& alias)
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

        Texture::SharedPtr pTexture;
        if (depth > 1)
        {
            assert(sampleCount == 1);
            pTexture = Texture::create3D(width, height, depth, format, 1, nullptr, field.getBindFlags() | Resource::BindFlags::ShaderResource);
        }
        else if (height > 1 || sampleCount > 1)
        {
            if (sampleCount > 1)
            {
                pTexture = Texture::create2DMS(width, height, format, sampleCount, 1, field.getBindFlags() | Resource::BindFlags::ShaderResource);
            }
            else
            {
                pTexture = Texture::create2D(width, height, format, 1, 1, nullptr, field.getBindFlags() | Resource::BindFlags::ShaderResource);
            }
        }
        else
        {
            pTexture = Texture::create1D(width, format, 1, 1, nullptr, field.getBindFlags() | Resource::BindFlags::ShaderResource);
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
