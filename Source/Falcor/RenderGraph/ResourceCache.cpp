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
#include "ResourceCache.h"
#include "Core/API/Texture.h"

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

    const Resource::SharedPtr& ResourceCache::getResource(const std::string& name) const
    {
        static const Resource::SharedPtr pNull;
        auto extIt = mExternalResources.find(name);

        // Search external resources if not found in render graph resources
        if (extIt == mExternalResources.end())
        {
            const auto& it = mNameToIndex.find(name);
            if (it == mNameToIndex.end()) return pNull;
            return mResourceData[it->second].pResource;
        }

        return extIt->second;
    }

    const RenderPassReflection::Field& ResourceCache::getResourceReflection(const std::string& name) const
    {
        uint32_t i = mNameToIndex.at(name);
        return mResourceData[i].field;
    }

    void ResourceCache::registerExternalResource(const std::string& name, const Resource::SharedPtr& pResource)
    {
        if(pResource) mExternalResources[name] = pResource;
        else
        {
            auto it = mExternalResources.find(name);
            if (it == mExternalResources.end())
            {
                logWarning("ResourceCache::registerExternalResource: " + name + " does not exist.");
                return;
            }

            mExternalResources.erase(it);
        }
    }

    void mergeTimePoint(std::pair<uint32_t, uint32_t>& range, uint32_t newTime)
    {
        range.first = std::min(range.first, newTime);
        range.second = std::max(range.second, newTime);
    }

    void ResourceCache::registerField(const std::string& name, const RenderPassReflection::Field& field, uint32_t timePoint, const std::string& alias)
    {
        assert(mNameToIndex.find(name) == mNameToIndex.end());

        bool addAlias = (alias.empty() == false);
        if (addAlias && mNameToIndex.count(alias) == 0)
        {
            throw std::exception(("Field named " + alias + " not found. Cannot register " + name + " as an alias.").c_str());
        }

        // Add a new field
        if (addAlias == false)
        {
            assert(mNameToIndex.count(name) == 0);
            mNameToIndex[name] = (uint32_t)mResourceData.size();
            bool resolveBindFlags = (field.getBindFlags() == ResourceBindFlags::None);
            mResourceData.push_back({ field, {timePoint, timePoint}, nullptr, resolveBindFlags, name });
        }
        else // Add alias
        {
            uint32_t index = mNameToIndex[alias];
            mNameToIndex[name] = index;
            mResourceData[index].field.merge(field);
            mergeTimePoint(mResourceData[index].lifetime, timePoint);
            mResourceData[index].pResource = nullptr;
            mResourceData[index].resolveBindFlags = mResourceData[index].resolveBindFlags || (field.getBindFlags() == ResourceBindFlags::None);
        }
    }

    Resource::SharedPtr createResourceForPass(const ResourceCache::DefaultProperties& params, const RenderPassReflection::Field& field, bool resolveBindFlags, const std::string& resourceName)
    {
        uint32_t width = field.getWidth() ? field.getWidth() : params.dims.x;
        uint32_t height = field.getHeight() ? field.getHeight() : params.dims.y;
        uint32_t depth = field.getDepth() ? field.getDepth() : 1;
        uint32_t sampleCount = field.getSampleCount() ? field.getSampleCount() : 1;
        auto bindFlags = field.getBindFlags();
        auto arraySize = field.getArraySize();
        auto mipLevels = field.getMipCount();

        ResourceFormat format = ResourceFormat::Unknown;

        if (field.getType() != RenderPassReflection::Field::Type::RawBuffer)
        {
            format = field.getFormat() == ResourceFormat::Unknown ? params.format : field.getFormat();
            if (resolveBindFlags)
            {
                ResourceBindFlags mask = Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource;
                bool isOutput = is_set(field.getVisibility(), RenderPassReflection::Field::Visibility::Output);
                bool isInternal = is_set(field.getVisibility(), RenderPassReflection::Field::Visibility::Internal);
                if (isOutput || isInternal) mask |= Resource::BindFlags::DepthStencil | Resource::BindFlags::RenderTarget;
                auto supported = getFormatBindFlags(format);
                mask &= supported;
                bindFlags |= mask;
            }
        }
        else // RawBuffer
        {
            if (resolveBindFlags) bindFlags = Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource;
        }
        Resource::SharedPtr pResource;

        switch (field.getType())
        {
        case RenderPassReflection::Field::Type::RawBuffer:
            pResource = Buffer::create(width, bindFlags, Buffer::CpuAccess::None);
            break;
        case RenderPassReflection::Field::Type::Texture1D:
            pResource = Texture::create1D(width, format, arraySize, mipLevels, nullptr, bindFlags);
            break;
        case RenderPassReflection::Field::Type::Texture2D:
            if (sampleCount > 1)
            {
                pResource = Texture::create2DMS(width, height, format, sampleCount, arraySize, bindFlags);
            }
            else
            {
                pResource = Texture::create2D(width, height, format, arraySize, mipLevels, nullptr, bindFlags);
            }
            break;
        case RenderPassReflection::Field::Type::Texture3D:
            pResource = Texture::create3D(width, height, depth, format, mipLevels, nullptr, bindFlags);
            break;
        case RenderPassReflection::Field::Type::TextureCube:
            pResource = Texture::createCube(width, height, format, arraySize, mipLevels, nullptr, bindFlags);
            break;
        default:
            should_not_get_here();
            return nullptr;
        }
        pResource->setName(resourceName);
        return pResource;
    }

    void ResourceCache::allocateResources(const DefaultProperties& params)
    {
        for (auto& data : mResourceData)
        {
            if ((data.pResource == nullptr) && (data.field.isValid()))
            {
                data.pResource = createResourceForPass(params, data.field, data.resolveBindFlags, data.name);
            }
        }
    }
}
