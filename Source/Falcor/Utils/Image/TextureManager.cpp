/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "TextureManager.h"
#include "Core/AssetResolver.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
#include "Utils/NumericRange.h"

#include <execution>

// Temporarily disable asynchronous texture loader until Falcor supports parallel GPU work submission.
// Until then `TextureManager` should only called from the main thread.
#define DISABLE_ASYNC_TEXTURE_LOADER

namespace Falcor
{
namespace
{
const size_t kMaxTextureHandleCount = std::numeric_limits<uint32_t>::max();
static_assert(TextureManager::CpuTextureHandle::kInvalidID >= kMaxTextureHandleCount);
} // namespace

TextureManager::TextureManager(ref<Device> pDevice, size_t maxTextureCount, size_t threadCount)
    : mpDevice(pDevice), mAsyncTextureLoader(pDevice, threadCount), mMaxTextureCount(std::min(maxTextureCount, kMaxTextureHandleCount))
{}

TextureManager::~TextureManager() {}

TextureManager::CpuTextureHandle TextureManager::addTexture(const ref<Texture>& pTexture)
{
    FALCOR_ASSERT(pTexture);
    if (pTexture->getType() != Resource::Type::Texture2D || pTexture->getSampleCount() != 1)
    {
        FALCOR_THROW("Only single-sample 2D textures can be added");
    }

    std::unique_lock<std::mutex> lock(mMutex);
    CpuTextureHandle handle;

    if (auto it = mTextureToHandle.find(pTexture.get()); it != mTextureToHandle.end())
    {
        // Texture is already managed. Return its handle.
        handle = it->second;
    }
    else
    {
        // Texture is not already managed. Add new texture desc.
        TextureDesc desc = {TextureState::Loaded, pTexture};
        handle = addDesc(desc);

        // Add to texture-to-handle map.
        mTextureToHandle[pTexture.get()] = handle;

        // If texture was originally loaded from disk, add to key-to-handle map to avoid loading it again later if requested in
        // loadTexture(). It's possible the user-provided texture has already been loaded by us. In that case, log a warning as the
        // redundant load should be fixed.
        if (!pTexture->getSourcePath().empty())
        {
            bool hasMips = pTexture->getMipCount() > 1;
            bool isSrgb = isSrgbFormat(pTexture->getFormat());
            TextureKey textureKey({pTexture->getSourcePath().string()}, hasMips, isSrgb, pTexture->getBindFlags());

            if (mKeyToHandle.find(textureKey) == mKeyToHandle.end())
            {
                mKeyToHandle[textureKey] = handle;
            }
            else
            {
                logWarning(
                    "TextureManager::addTexture() - Texture loaded from '{}' appears to be identical to an already loaded texture. This "
                    "could be optimized by getting it from TextureManager.",
                    pTexture->getSourcePath()
                );
            }
        }
    }

    return handle;
}

TextureManager::CpuTextureHandle TextureManager::loadUdimTexture(
    const std::filesystem::path& path,
    bool generateMipLevels,
    bool loadAsSRGB,
    ResourceBindFlags bindFlags,
    bool async,
    const AssetResolver* assetResolver,
    size_t* loadedTextureCount
)
{
    std::string filename = path.filename().string();

    // Search UDIMS based on mip0
    auto mipPos = filename.find("<MIP>");
    if (mipPos != std::string::npos)
        filename.replace(mipPos, 5, "mip0");

    auto pos = filename.find("<UDIM>");
    if (pos == std::string::npos)
        return loadTexture(path, generateMipLevels, loadAsSRGB, bindFlags, async);

    std::filesystem::path dirpath = path.parent_path();
    filename.replace(pos, 6, "[1-9][0-9][0-9][0-9]");
    std::regex udimRegex(filename);
    std::vector<std::filesystem::path> texturePaths;
    // Find the first directory containing the pattern, in case the UDIM set lives in multiple available directories
    if (assetResolver)
        texturePaths = assetResolver->resolvePathPattern(dirpath, filename, true /* firstMatchOnly */);
    else
        texturePaths = globFilesInDirectory(dirpath, udimRegex, true /* firstMatchOnly */);

    // nothing found, return an invalid handle
    if (texturePaths.empty())
    {
        logWarning("Can't find UDIM texture files '{}'.", path);
        return CpuTextureHandle();
    }

    // Now load all the files from that directory
    std::filesystem::path loadedDir = texturePaths[0].parent_path();
    texturePaths = globFilesInDirectory(loadedDir, udimRegex);

    if (loadedTextureCount)
        *loadedTextureCount = texturePaths.size();

    std::vector<size_t> udimIndices;
    std::vector<CpuTextureHandle> handles;
    size_t maxIndex = 0;
    for (auto& it : texturePaths)
    {
        std::string textureFilename = it.filename().string();
        std::string udimStr = textureFilename.substr(pos, 4); // the 4 digits
        // Insert the udim number into the original filename (before potentially stripping <MIP>)
        std::string srcStr = path.filename().string();
        std::string newFilename = srcStr.replace(srcStr.find("<UDIM>"), 6, udimStr);
        it = it.parent_path() / newFilename;
        size_t udim = std::stol(udimStr);
        maxIndex = std::max<size_t>(maxIndex, udim);
        udimIndices.push_back(udim);
        handles.push_back(loadTexture(it, generateMipLevels, loadAsSRGB, bindFlags, async));

        FALCOR_CHECK(udim >= 1001, "Texture {} is not a valid UDIM texture, as it violates the valid UDIM range of 1001-9999", it);
    }

    // UDIM range needs to cover all numbers from 1001 to maxIndex inclusive, so 1001, 1002, 1003 needs 3 indices
    size_t rangeStart = getUdimRange(maxIndex - 1001 + 1);

    for (size_t i = 0; i < texturePaths.size(); ++i)
    {
        size_t index = udimIndices[i] - 1001;
        mUdimIndirection[rangeStart + index] = handles[i].getID();
    }

    return TextureManager::CpuTextureHandle(rangeStart, true);
}

TextureManager::CpuTextureHandle TextureManager::loadTexture(
    const std::filesystem::path& path,
    bool generateMipLevels,
    bool loadAsSRGB,
    ResourceBindFlags bindFlags,
    bool async,
    const AssetResolver* assetResolver,
    size_t* loadedTextureCount
)
{
    if (path.string().find("<UDIM>") != std::string::npos)
        return loadUdimTexture(path, generateMipLevels, loadAsSRGB, bindFlags, async, assetResolver, loadedTextureCount);

    std::vector<std::filesystem::path> paths;
    auto addPath = [&](const std::filesystem::path& p)
    {
        // Find the full path to the texture.
        std::filesystem::path fullPath;
        bool found = false;
        if (assetResolver)
        {
            fullPath = assetResolver->resolvePath(p);
            found = !fullPath.empty();
        }
        else
        {
            fullPath = p;
            found = std::filesystem::exists(fullPath);
        }

        if (found)
            paths.emplace_back(std::move(fullPath));

        return found;
    };

    // If we find <MIP> in the filename, locate all mip levels
    std::string filename = path.filename().string();
    auto mipPos = filename.find("<MIP>");
    if (mipPos != std::string::npos)
    {
        while (true)
        {
            std::string basename = std::string(filename).replace(mipPos, 5, "mip" + std::to_string(paths.size()));
            std::filesystem::path mip = path.parent_path() / basename;
            if (!addPath(mip))
                break;
        }
    }
    else
    {
        addPath(path);
    }

    if (loadedTextureCount)
        *loadedTextureCount = paths.empty() ? 0 : 1;

    CpuTextureHandle handle;
    if (paths.empty())
    {
        logWarning("Can't find texture file '{}'.", path);
        return handle;
    }

    std::unique_lock<std::mutex> lock(mMutex);
    const TextureKey textureKey(paths, generateMipLevels, loadAsSRGB, bindFlags);
    if (auto it = mKeyToHandle.find(textureKey); it != mKeyToHandle.end())
    {
        // Texture is already managed. Return its handle.
        handle = it->second;
    }
    else
    {
        if (mUseDeferredLoading)
        {
            // Add new texture desc.
            TextureDesc desc = {TextureState::Referenced, nullptr};
            handle = addDesc(desc);

            // Add to key-to-handle map.
            mKeyToHandle[textureKey] = handle;

            // Return early.
            return handle;
        }

#ifndef DISABLE_ASYNC_TEXTURE_LOADER
        mLoadRequestsInProgress++;

        // Texture is not already managed. Add new texture desc.
        TextureDesc desc = {TextureState::Referenced, nullptr};
        handle = addDesc(desc);

        // Add to key-to-handle map.
        mKeyToHandle[textureKey] = handle;

        // Function called by the async texture loader when loading finishes.
        // It's called by a worker thread so needs to acquire the mutex before changing any state.
        auto callback = [=](ref<Texture> pTexture)
        {
            std::unique_lock<std::mutex> lock(mMutex);

            // Mark texture as loaded.
            auto& desc = getDesc(handle);
            desc.state = TextureState::Loaded;
            desc.pTexture = pTexture;

            // Add to texture-to-handle map.
            if (pTexture)
                mTextureToHandle[pTexture.get()] = handle;

            mLoadRequestsInProgress--;
            mCondition.notify_all();
        };

        // Issue load request to texture loader.
        if (paths.size() > 1)
        {
            mAsyncTextureLoader.loadMippedFromFiles(paths, loadAsSRGB, bindFlags, callback);
        }
        else
        {
            mAsyncTextureLoader.loadFromFile(paths[0], generateMipLevels, loadAsSRGB, bindFlags, callback);
        }
#else
        // Load texture from main thread.
        ref<Texture> pTexture;
        if (paths.size() > 1)
        {
            pTexture = Texture::createMippedFromFiles(mpDevice, paths, loadAsSRGB, bindFlags);
        }
        else
        {
            pTexture = Texture::createFromFile(mpDevice, paths[0], generateMipLevels, loadAsSRGB, bindFlags);
        }

        // Add new texture desc.
        TextureDesc desc = {TextureState::Loaded, pTexture};
        handle = addDesc(desc);

        // Add to key-to-handle map.
        mKeyToHandle[textureKey] = handle;

        // Add to texture-to-handle map.
        if (pTexture)
            mTextureToHandle[pTexture.get()] = handle;

        mCondition.notify_all();
#endif
    }

    lock.unlock();

    if (!mUseDeferredLoading && !async)
    {
        waitForTextureLoading(handle);
    }

    return handle;
}

void TextureManager::waitForTextureLoading(const CpuTextureHandle& handle)
{
    if (!handle)
        return;

    // Acquire mutex and wait for texture state to change.
    std::unique_lock<std::mutex> lock(mMutex);
    mCondition.wait(lock, [&]() { return getDesc(handle).state == TextureState::Loaded; });

    mpDevice->wait();
}

void TextureManager::waitForAllTexturesLoading()
{
    // Acquire mutex and wait for all in-progress requests to finish.
    std::unique_lock<std::mutex> lock(mMutex);
    mCondition.wait(lock, [&]() { return mLoadRequestsInProgress == 0; });

    mpDevice->wait();
}

void TextureManager::beginDeferredLoading()
{
    mUseDeferredLoading = true;
}

void TextureManager::endDeferredLoading()
{
    struct Job
    {
        TextureKey key;
        CpuTextureHandle handle;
    };

    // Get a list of textures to load.
    std::vector<Job> jobs;
    for (auto& [key, handle] : mKeyToHandle)
    {
        auto& desc = getDesc(handle);
        if (desc.state == TextureState::Referenced)
            jobs.push_back(Job{key, handle});
    }

    // Early out if there are no textures to load.
    mUseDeferredLoading = false;
    if (jobs.empty())
        return;

    // Load textures in parallel.
    std::atomic<size_t> texturesLoaded;
    NumericRange<size_t> jobRange(0, jobs.size());
    std::for_each(
        std::execution::par_unseq,
        jobRange.begin(),
        jobRange.end(),
        [&](size_t i)
        {
            const auto& job = jobs[i];
            auto& desc = getDesc(job.handle);
            if (job.key.fullPaths.size() == 1)
            {
                desc.pTexture = Texture::createFromFile(
                    mpDevice, job.key.fullPaths[0], job.key.generateMipLevels, job.key.loadAsSRGB, job.key.bindFlags
                );
                logDebug("Loading texture from '{}'", job.key.fullPaths[0]);
            }
            else
            {
                desc.pTexture = Texture::createMippedFromFiles(mpDevice, job.key.fullPaths, job.key.loadAsSRGB, job.key.bindFlags);
                logDebug("Loading mipped texture from '{}'", job.key.fullPaths[0]);
            }
            if (texturesLoaded.fetch_add(1) % 10 == 9)
            {
                logDebug("Flush");
                std::lock_guard<std::mutex> lock(mpDevice->getGlobalGfxMutex());
                mpDevice->wait();
            }
        }
    );
    mpDevice->wait();

    // Mark loaded textures and add them to lookup table.
    for (const auto& job : jobs)
    {
        auto& desc = getDesc(job.handle);
        desc.state = desc.pTexture ? TextureState::Loaded : TextureState::Invalid;
        mTextureToHandle[desc.pTexture.get()] = job.handle;
    }
}

void TextureManager::removeTexture(const CpuTextureHandle& handle)
{
    if (handle.isUdim())
    {
        removeUdimTexture(handle);
    }
    if (!handle)
        return;

    waitForTextureLoading(handle);

    std::lock_guard<std::mutex> lock(mMutex);

    // Get texture desc. If it's already cleared, we're done.
    auto& desc = getDesc(handle);
    if (!desc.isValid())
        return;

    // Remove handle from maps.
    // Note not all handles exist in key-to-handle map so search for it. This can be optimized if needed.
    auto it = std::find_if(mKeyToHandle.begin(), mKeyToHandle.end(), [handle](const auto& keyVal) { return keyVal.second == handle; });
    if (it != mKeyToHandle.end())
        mKeyToHandle.erase(it);

    if (desc.pTexture)
    {
        FALCOR_ASSERT(mTextureToHandle.find(desc.pTexture.get()) != mTextureToHandle.end());
        mTextureToHandle.erase(desc.pTexture.get());
    }

    // Clear texture desc.
    desc = {};

    // Return handle to the free list.
    mFreeList.push_back(handle);
}

TextureManager::TextureDesc TextureManager::getTextureDesc(const CpuTextureHandle& handle) const
{
    if (!handle)
        return {};

    std::lock_guard<std::mutex> lock(mMutex);
    FALCOR_CHECK(!handle.isUdim(), "Can't lookup texture desc from handle to UDIM texture. Resolve UDIM first.");
    FALCOR_CHECK(handle && handle.getID() < mTextureDescs.size(), "Invalid texture handle.");
    return mTextureDescs[handle.getID()];
}

size_t TextureManager::getTextureDescCount() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mTextureDescs.size();
}

void TextureManager::bindShaderData(const ShaderVar& texturesVar, const size_t descCount, const ShaderVar& udimsVar) const
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (mTextureDescs.size() > descCount)
    {
        FALCOR_THROW("Descriptor array size ({}) is too small for the required number of textures ({})", descCount, mTextureDescs.size());
    }

    ref<Texture> nullTexture;
    for (size_t i = 0; i < mTextureDescs.size(); i++)
    {
        texturesVar[i] = mTextureDescs[i].pTexture;
    }
    for (size_t i = mTextureDescs.size(); i < descCount; i++)
    {
        texturesVar[i] = nullTexture;
    }

    if (mUdimIndirection.empty())
    {
        mpUdimIndirection.reset();
        mUdimIndirectionDirty = false;
        return;
    }

    if (!mpUdimIndirection || mUdimIndirection.size() > mpUdimIndirection->getElementCount())
    {
        mpUdimIndirection = mpDevice->createStructuredBuffer(
            sizeof(int32_t),
            mUdimIndirection.size(),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            mUdimIndirection.data(),
            false
        );
        mUdimIndirectionDirty = false;
    }

    if (mUdimIndirectionDirty)
    {
        mpUdimIndirection->setBlob(mUdimIndirection.data(), 0, mUdimIndirection.size() * sizeof(int32_t));
    }

    udimsVar = mpUdimIndirection;
}

TextureManager::Stats TextureManager::getStats() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    TextureManager::Stats s;
    for (const auto& t : mTextureDescs)
    {
        if (!t.pTexture)
            continue;
        uint64_t texelCount = t.pTexture->getTexelCount();
        uint32_t channelCount = getFormatChannelCount(t.pTexture->getFormat());
        s.textureCount++;
        s.textureTexelCount += texelCount;
        s.textureTexelChannelCount += texelCount * channelCount;
        s.textureMemoryInBytes += t.pTexture->getTextureSizeInBytes();
        if (isCompressedFormat(t.pTexture->getFormat()))
            s.textureCompressedCount++;
    }
    return s;
}

TextureManager::CpuTextureHandle TextureManager::addDesc(const TextureDesc& desc)
{
    CpuTextureHandle handle;

    // Allocate new texture handle and insert desc.
    if (!mFreeList.empty())
    {
        handle = mFreeList.back();
        mFreeList.pop_back();
        getDesc(handle) = desc;
    }
    else
    {
        if (mTextureDescs.size() >= mMaxTextureCount)
        {
            FALCOR_THROW("Out of texture handles");
        }
        handle = CpuTextureHandle{static_cast<uint32_t>(mTextureDescs.size())};
        mTextureDescs.emplace_back(desc);
    }

    return handle;
}

TextureManager::TextureDesc& TextureManager::getDesc(const CpuTextureHandle& handle)
{
    FALCOR_CHECK(!handle.isUdim(), "Can't lookup texture desc from handle to UDIM texture. Resolve UDIM first.");
    FALCOR_CHECK(handle && handle.getID() < mTextureDescs.size(), "Invalid texture handle.");
    return mTextureDescs[handle.getID()];
}

size_t TextureManager::getUdimRange(size_t requiredSize)
{
    // But first look in the freed ranges for the smallest one that we can reuse
    size_t smallestFound = std::numeric_limits<size_t>::max();
    int64_t foundIndex = -1;
    for (size_t i = 0; i < mFreeUdimRanges.size(); ++i)
    {
        size_t rangeSize = mUdimIndirectionSize[mFreeUdimRanges[i]];
        if (requiredSize <= rangeSize && rangeSize < smallestFound)
        {
            foundIndex = i;
            smallestFound = rangeSize;
        }
        // If we found an exact match, no need to look further
        if (smallestFound == requiredSize)
            break;
    }

    // haven't found any reusable, just add a new one
    if (foundIndex == -1)
    {
        const size_t rangeStart = mUdimIndirection.size();
        mUdimIndirection.resize(rangeStart + requiredSize, -1);
        mUdimIndirectionSize.resize(rangeStart + requiredSize, 0);
        mUdimIndirectionSize[rangeStart] = requiredSize;
        return rangeStart;
    }

    mUdimIndirectionDirty = true;

    // Range is already filled with -1 from the deletion
    const size_t rangeStart = mFreeUdimRanges[foundIndex];
    mFreeUdimRanges[foundIndex] = mFreeUdimRanges.back();
    mFreeUdimRanges.pop_back();
    return rangeStart;
}

void TextureManager::freeUdimRange(size_t rangeStart)
{
    mUdimIndirectionDirty = true;
    mFreeUdimRanges.push_back(rangeStart);
}

void TextureManager::removeUdimTexture(const CpuTextureHandle& handle)
{
    size_t rangeStart = handle.getID();
    size_t rangeSize = mUdimIndirectionSize[rangeStart];
    for (size_t i = rangeStart; i < rangeStart + rangeSize; ++i)
    {
        if (mUdimIndirection[i] < 0)
            continue;
        CpuTextureHandle texHandle(mUdimIndirection[i]);
        removeTexture(texHandle);
        mUdimIndirection[i] = -1;
    }

    freeUdimRange(rangeStart);
}

TextureManager::CpuTextureHandle TextureManager::resolveUdimTexture(const CpuTextureHandle& handle, const float2& uv) const
{
    if (!handle.isUdim())
        return handle;

    // Compute which UDIM ID texture coordinate maps to.
    FALCOR_CHECK(uv[0] >= 0.f && uv[0] < 10.f && uv[1] >= 0.f && uv[1] < 10.f, "UDIM texture coordinate ({}) is out of range.", uv);
    uint32_t udimID = 1001 + uint32_t(uv[0]) + 10 * uint32_t(uv[1]);

    return resolveUdimTexture(handle, udimID);
}

TextureManager::CpuTextureHandle TextureManager::resolveUdimTexture(const CpuTextureHandle& handle, const uint32_t udimID) const
{
    if (!handle.isUdim())
        return handle;

    // Check if UDIM ID is valid and within range of the indirection table.
    // udimID = 1001 + u + (10 * v) where u,v in 0..9 => valid IDs are 1001..1100.
    FALCOR_CHECK(udimID >= 1001 && udimID <= 1100, "Illegal UDIM ID ({}).", udimID);
    size_t rangeStart = handle.getID();
    size_t udim = udimID - 1001;
    FALCOR_CHECK(udim < mUdimIndirectionSize[rangeStart], "UDIM ID ({}) is out of range.", udimID);

    if (mUdimIndirection[rangeStart + udim] >= 0)
        return CpuTextureHandle(mUdimIndirection[rangeStart + udim]);
    return CpuTextureHandle();
}
} // namespace Falcor
