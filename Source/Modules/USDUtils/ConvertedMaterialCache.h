/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "Scene/Material/Material.h"

#include "USDUtils/USDUtils.h"
#include "USDUtils/USDHelpers.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdShade/material.h>
END_DISABLE_USD_WARNINGS

#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Falcor
{

/** Cache of materials converted from a UsdShadeShader.
    Minimizes creation of duplicate materials by caching converted materials by UsdShadeShader
    (to trivially reuse created materials), and by a material spec (i.e., parameters used to create a
    material of a given type), which avoids creating identical materials from different UsdShadeShaders.
    Insertion and retrieval are thread-safe.

    The class is templated by:

    T: Material type (e.g., Falcor::StandardMaterial)
    S: Material spec (e.g., Falcor::StandardMaterialSpec)
    H: Hash class over S
*/
template<typename T, typename S, typename H>
class ConvertedMaterialCache
{
public:
    ref<T> get(const pxr::UsdShadeShader& shader);
    ref<T> get(const S& spec, const std::string& shaderName);

    void add(const UsdShadeShader& shader, const S& spec, ref<T> pMaterial);

private:
    //< Spec creation can be expensive, so we cache both by shader (to avoid unnecessary spec creation)
    //< and by spec.

    ///< Map from UsdShadeShader to Falcor material instance. An entry with a null instance
    ///< indicates in-progress conversion.
    std::unordered_map<pxr::UsdPrim, ref<T>, UsdObjHash> mPrimMaterialCache;

    ///< Map from material spec S to Falcor material instance. An entry with a null instance indicates in-progress
    ///< conversion.
    std::unordered_map<S, ref<T>, H> mSpecMaterialCache;

    std::mutex mCacheMutex;                    ///< Mutex controlling access to the unordered_maps.
    std::condition_variable mPrimCacheUpdated; ///< Condition variable for threads waiting on by-prim cache update.
    std::condition_variable mSpecCacheUpdated; ///< Condition variable for threads waiting on by-spec cache update.
};

template<typename T, typename S, typename H>
ref<T> ConvertedMaterialCache<T, S, H>::get(const UsdShadeShader& shader)
{
    std::unique_lock lock(mCacheMutex);

    std::string shaderName = shader.GetPath().GetString();

    while (true)
    {
        auto iter = mPrimMaterialCache.find(shader.GetPrim());
        if (iter == mPrimMaterialCache.end())
        {
            // No entry for this shader. Add one, with a nullptr Falcor material instance, to indicate that work is
            // underway.
            logDebug("No cached material prim for shader '{}'; starting conversion.", shaderName);
            mPrimMaterialCache.insert(std::make_pair(shader.GetPrim(), nullptr));
            return nullptr;
        }

        if (iter->second != nullptr)
        {
            // Found a valid entry
            logDebug("Found a cached material prim for shader '{}'.", shaderName);
            return iter->second;
        }
        /** There is an cache entry, but it is null, indicating that another thread is current performing conversion.
            Wait until we are signaled that a new entry has been added to the cache, and loop again to check for
            existence in the cache. Note that this mechanism requires that conversion never fail to create an entry in
            the cache after calling this function.
        */
        logDebug("In-progress cached entry for shader '{}' detected. Waiting.", shaderName);
        mPrimCacheUpdated.wait(lock);
    }
    FALCOR_UNREACHABLE();
    return nullptr;
}

template<typename T, typename S, typename H>
ref<T> ConvertedMaterialCache<T, S, H>::get(const S& spec, const std::string& shaderName)
{
    std::unique_lock lock(mCacheMutex);

    while (true)
    {
        auto iter = mSpecMaterialCache.find(spec);
        if (iter == mSpecMaterialCache.end())
        {
            // No entry for this shader. Add one, with a nullptr Falcor material instance, to indicate that work is
            // underway.
            logDebug("No cached material for spec '{}'; starting conversion.", shaderName);
            mSpecMaterialCache.insert(std::make_pair(spec, nullptr));
            return nullptr;
        }
        if (iter->second != nullptr)
        {
            // Found a valid entry
            logDebug("Found a cached material spec for surface '{}'.", shaderName);
            return iter->second;
        }
        /** There is an cache entry, but it is null, indicating that another thread is current performing conversion.
            Wait until we are signaled that a new entry has been added to the cache, and loop again to check for
            existence in the cache. Note that this mechanism requires that conversion never fail to create an entry in
            the cache after calling this function.
        */
        logDebug("In-progress cached entry for spec '{}' detected. Waiting.", shaderName);
        mSpecCacheUpdated.wait(lock);
    }
    FALCOR_UNREACHABLE();
    return nullptr;
}

template<typename T, typename S, typename H>
void ConvertedMaterialCache<T, S, H>::add(const UsdShadeShader& shader, const S& spec, ref<T> pMaterial)
{
    {
        std::unique_lock lock(mCacheMutex);
        if (mPrimMaterialCache.find(shader.GetPrim()) == mPrimMaterialCache.end())
        {
            FALCOR_THROW("Expected converted material cache entry for '{}' not found.", shader.GetPath().GetString());
        }
        mPrimMaterialCache[shader.GetPrim()] = pMaterial;

        if (mSpecMaterialCache.find(spec) == mSpecMaterialCache.end())
        {
            FALCOR_THROW("Expected converted material spec cache entry for '{}' not found.", shader.GetPath().GetString());
        }
        mSpecMaterialCache[spec] = pMaterial;
    }
    mPrimCacheUpdated.notify_all();
    mSpecCacheUpdated.notify_all();
}
} // namespace Falcor
