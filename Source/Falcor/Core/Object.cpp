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
#include "Object.h"
#include "Error.h"
#include <set>
#include <mutex>

namespace Falcor
{

#if FALCOR_ENABLE_OBJECT_TRACKING
static std::mutex sTrackedObjectsMutex;
static std::set<const Object*> sTrackedObjects;
#endif

void Object::incRef() const
{
    uint32_t refCount = mRefCount.fetch_add(1);
#if FALCOR_ENABLE_OBJECT_TRACKING
    if (refCount == 0)
    {
        std::lock_guard<std::mutex> lock(sTrackedObjectsMutex);
        sTrackedObjects.insert(this);
    }
#endif
}

void Object::decRef(bool dealloc) const noexcept
{
    uint32_t refCount = mRefCount.fetch_sub(1);
    if (refCount <= 0)
    {
        reportFatalErrorAndTerminate("Internal error: Object reference count < 0!");
    }
    else if (refCount == 1)
    {
#if FALCOR_ENABLE_OBJECT_TRACKING
        {
            std::lock_guard<std::mutex> lock(sTrackedObjectsMutex);
            sTrackedObjects.erase(this);
        }
#endif
        if (dealloc)
            delete this;
    }
}

#if FALCOR_ENABLE_OBJECT_TRACKING

void Object::dumpAliveObjects()
{
    std::lock_guard<std::mutex> lock(sTrackedObjectsMutex);
    logInfo("Alive objects:");
    for (const Object* object : sTrackedObjects)
        object->dumpRefs();
}

void Object::dumpRefs() const
{
    logInfo("Object (class={} address={}) has {} reference(s)", getClassName(), fmt::ptr(this), refCount());
#if FALCOR_ENABLE_REF_TRACKING
    std::lock_guard<std::mutex> lock(mRefTrackerMutex);
    for (const auto& it : mRefTrackers)
    {
        logInfo("ref={} count={}\n{}\n", it.first, it.second.count, it.second.origin);
    }
#endif
}

#endif // FALCOR_ENABLE_OBJECT_TRACKING

#if FALCOR_ENABLE_REF_TRACKING

void Object::incRef(uint64_t refId) const
{
    if (mEnableRefTracking)
    {
        std::lock_guard<std::mutex> lock(mRefTrackerMutex);
        auto it = mRefTrackers.find(refId);
        if (it != mRefTrackers.end())
        {
            it->second.count++;
        }
        else
        {
            mRefTrackers.emplace(refId, getStackTrace());
        }
    }

    incRef();
}

void Object::decRef(uint64_t refId, bool dealloc) const noexcept
{
    if (mEnableRefTracking)
    {
        std::lock_guard<std::mutex> lock(mRefTrackerMutex);
        auto it = mRefTrackers.find(refId);
        FALCOR_ASSERT(it != mRefTrackers.end());
        if (--it->second.count == 0)
        {
            mRefTrackers.erase(it);
        }
    }

    decRef();
}

void Object::setEnableRefTracking(bool enable)
{
    mEnableRefTracking = enable;
}

#endif // FALCOR_ENABLE_REF_TRACKING

} // namespace Falcor
