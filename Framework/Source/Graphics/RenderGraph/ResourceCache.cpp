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
        mFieldMap.clear();
        mResourceData.clear();
        mExternalResources.clear();
    }

    const std::shared_ptr<Resource>& ResourceCache::getResource(const std::string& name) const
    {
        static const std::shared_ptr<Resource> pNull;
        const auto& it = mFieldMap.find(name);
        if (it == mFieldMap.end())
        {
            auto extIt = mExternalResources.find(name);
            if (extIt == mExternalResources.end())
            {
                logWarning("Can't find a resource named `" + name + "` in the resource cache");
                return pNull;
            }

            return extIt->second;
        }

        return mResourceData[it->second].pResource;
    }

    const std::shared_ptr<Falcor::Resource>& ResourceCache::getExternalResource(const std::string& name) const
    {
        static const std::shared_ptr<Resource> pNull;
        const auto& it = mExternalResources.find(name);
        if (it == mExternalResources.end())
        {
            logWarning("Can't find an external resource named `" + name + "` registered in the resource cache");
            return pNull;
        }

        return it->second;
    }

    void ResourceCache::registerExternalResource(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        mExternalResources[name] = pResource;
    }

    void ResourceCache::removeExternalResource(const std::string& name)
    {
        mExternalResources.erase(name);
    }

    void ResourceCache::registerField(const std::string& name, const RenderPassReflection::Field& field, const std::string& alias)
    {
        if (mFieldMap.count(name) > 0)
        {
            logWarning("ResourceCache::registerField: Field named " + name + " already exists. Ignoring operation.");
            return;
        }

        bool addAlias = alias.empty() == false;
        if (addAlias && mFieldMap.count(alias) == 0)
        {
            // Should invalid alias name fail with an error/warning or just add a new one?
            addAlias = false;
        }

        // Add a new field
        if (addAlias == false)
        {
            assert(mFieldMap.count(name) == 0);
            mFieldMap[name] = (uint32_t)mResourceData.size();
            mResourceData.push_back({ field, nullptr });
        }
        else
        {
            uint32_t index = mFieldMap[alias];
            mFieldMap[name] = index;

            // Merge fields, and overwrite previously unknown/unspecified fields with specified ones
            RenderPassReflection::Field& cachedField = mResourceData[index].field;

            uint32_t w = (cachedField.getWidth() == 0 && field.getWidth() != 0) ? field.getWidth() : cachedField.getWidth();
            uint32_t h = (cachedField.getHeight() == 0 && field.getHeight() != 0) ? field.getHeight() : cachedField.getHeight();
            uint32_t d = (cachedField.getDepth() == 0 && field.getDepth() != 0) ? field.getDepth() : cachedField.getDepth();
            cachedField.setDimensions(w, h, d);

            if (cachedField.getFormat() == ResourceFormat::Unknown && field.getFormat() != ResourceFormat::Unknown)
            {
                cachedField.setFormat(field.getFormat());
            }

            // TODO: Output error if fields cannot be merged: Trying to alias incompatible formats, etc. Some of these errors should be caught by RenderGraph::addEdge

        }
    }

    void ResourceCache::allocateResources(const DefaultProperties& params)
    {
        for (auto& data : mResourceData)
        {
            if (data.pResource == nullptr)
            {
                data.pResource = createTextureForPass(params, data.field);
            }
        }
    }

    Texture::SharedPtr ResourceCache::createTextureForPass(const DefaultProperties& params, const RenderPassReflection::Field& field)
    {
        uint32_t width = field.getWidth() ? field.getWidth() : params.width;
        uint32_t height = field.getHeight() ? field.getHeight() : params.height;
        uint32_t depth = field.getDepth() ? field.getDepth() : 1;
        uint32_t sampleCount = field.getSampleCount() ? field.getSampleCount() : 1;
        ResourceFormat format = field.getFormat() == ResourceFormat::Unknown ? params.colorFormat : field.getFormat();
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

}
