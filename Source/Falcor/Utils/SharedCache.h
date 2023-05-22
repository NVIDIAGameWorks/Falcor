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
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>

namespace Falcor
{

/**
 * Helper class for managing a shared cache.
 *
 * This is used in a few places where Falcor used global statics in the past.
 * Because Falcor now supports multiple devices, global statics don't work anymore.
 * The shared cache is used locally in a file where some resource is shared
 * among all instances of a class. The first instance creates the shared
 * resource, all subsequent instances can reuse the cached data. If all instances
 * are destroyed, the shared resource is automatically released as only the
 * instances hold a shared_ptr to the cached item, the cache itself only
 * holds a weak_ptr. Using a Key type, we can cache multiple versions of the same
 * data, typically used to cache one set for every GPU device instance.
 */
template<typename T, typename Key>
struct SharedCache
{
    std::shared_ptr<T> acquire(Key key, const std::function<std::shared_ptr<T>()>& init)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = cache.find(key);
        if (it != cache.end())
        {
            if (auto data = it->second.lock())
                return data;
            else
                cache.erase(it);
        }

        std::shared_ptr<T> data = init();
        cache.emplace(key, data);
        return data;
    }

    std::mutex mutex;
    std::map<Key, std::weak_ptr<T>> cache;
};

} // namespace Falcor
