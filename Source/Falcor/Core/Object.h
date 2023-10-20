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

#include "Core/Macros.h"

#include <fmt/core.h>

#include <atomic>
#include <utility>
#include <cstdint>

/**
 * Enable/disable object lifetime tracking.
 * When enabled, each object that derives from Object will have its
 * lifetime tracked. This is useful for debugging memory leaks.
 */
#define FALCOR_ENABLE_OBJECT_TRACKING 0

/**
 * Enable/disable reference tracking.
 * When enabled, all references to object that have reference tracking
 * enabled using setEnableRefTracking() are tracked. Each time the reference
 * count is increased, the current stack trace is stored. This helps identify
 * owners of objects that are not properly releasing their references.
 */
#define FALCOR_ENABLE_REF_TRACKING 0

#if FALCOR_ENABLE_REF_TRACKING
#if !FALCOR_ENABLE_OBJECT_TRACKING
#error "FALCOR_ENABLE_REF_TRACKING requires FALCOR_ENABLE_OBJECT_TRACKING"
#endif
#include <map>
#include <mutex>
#endif

namespace Falcor
{

/**
 * @brief Base class for reference counted objects.
 *
 * This class (in conjunction with the ``ref`` reference counter) is the
 * foundation of an efficient reference-counted object system. The
 * implementation here is an alternative to standard mechanisms for reference
 * counting such as ``std::shared_ptr`` from the STL.
 *
 * There are a few reasons for using a custom reference counting system
 * in Falcor:
 *
 * The reference count is stored in a 32-bit integer and each reference
 * is a single pointer. This is more efficient than the 64-bit reference
 * count and two-pointer reference of ``std::shared_ptr``.
 *
 * Objects are always allocated with a single allocation. With
 * ``std::shared_ptr`` one has to use ``std::make_shared`` to ensure that
 * the object and the reference count are allocated together.
 *
 * We can track references to objects. This is useful for debugging complex
 * ownership scenarios.
 *
 * Finally, if we want to migrate from pybind11 to nanobind at some point in
 * the future, we will need to use a custom reference counting system.
 */
class FALCOR_API Object
{
public:
    /// Default constructor.
    Object() = default;

    /// Copy constructor.
    /// Note: We don't copy the reference counter, so that the new object
    /// starts with a reference count of 0.
    Object(const Object&) {}

    /// Copy assignment.
    /// Note: We don't copy the reference counter, but leave the reference
    /// counts of the two objects unchanged. This results in the same semantics
    /// that we would get if we used `std::shared_ptr` where the reference
    /// counter is stored in a separate place from the object.
    Object& operator=(const Object&) { return *this; }

    /// Make the object non-movable.
    Object(Object&&) = delete;
    Object& operator=(Object&&) = delete;

    /// Destructor.
    virtual ~Object() = default;

    /// Return the name of the class.
    /// Note: This reports the actual class name if FALCOR_OBJECT() is used.
    virtual const char* getClassName() const { return "Object"; }

    /// Return the current reference count.
    int refCount() const { return mRefCount; };

    /// Increase the object's reference count by one.
    void incRef() const;

    /// Decrease the reference count of the object and possibly deallocate it.
    void decRef(bool dealloc = true) const noexcept;

#if FALCOR_ENABLE_OBJECT_TRACKING
    /// Dump all objects that are currently alive.
    static void dumpAliveObjects();

    /// Dump references of this object.
    void dumpRefs() const;
#endif

#if FALCOR_ENABLE_REF_TRACKING
    void incRef(uint64_t refId) const;
    void decRef(uint64_t refId, bool dealloc = true) const noexcept;

    /// Enable/disable reference tracking of this object.
    void setEnableRefTracking(bool enable);
#endif

private:
    mutable std::atomic<uint32_t> mRefCount{0};

#if FALCOR_ENABLE_REF_TRACKING
    struct RefTracker
    {
        uint32_t count;
        std::string origin;
        RefTracker(std::string origin_) : count(1), origin(std::move(origin_)) {}
    };
    mutable std::map<uint64_t, RefTracker> mRefTrackers;
    mutable std::mutex mRefTrackerMutex;
    bool mEnableRefTracking = false;
#endif
};

#if FALCOR_ENABLE_REF_TRACKING
static uint64_t nextRefId()
{
    static std::atomic<uint64_t> sNextId = 0;
    return sNextId.fetch_add(1);
}
#endif

/// Macro to declare the object class name.
#define FALCOR_OBJECT(class_)                 \
public:                                       \
    const char* getClassName() const override \
    {                                         \
        return #class_;                       \
    }

/**
 * @brief Reference counting helper.
 *
 * The @a ref template is a simple wrapper to store a pointer to an object. It
 * takes care of increasing and decreasing the object's reference count as
 * needed. When the last reference goes out of scope, the associated object
 * will be deallocated.
 *
 * This class follows similar semantics to the ``std::shared_ptr`` class from
 * the STL. In particular, we avoid implicit conversion to and from raw
 * pointers.
 */
template<typename T>
class ref
{
public:
    /// Default constructor (nullptr).
    ref() {}

    /// Construct a reference from a nullptr.
    ref(std::nullptr_t) {}

    /// Construct a reference from a convertible pointer.
    template<typename T2 = T>
    explicit ref(T2* ptr) : mPtr(ptr)
    {
        static_assert(std::is_base_of_v<Object, T2>, "Cannot create reference to object not inheriting from Object class.");
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot create reference to object from unconvertible pointer type.");
        if (mPtr)
            incRef((const Object*)(mPtr));
    }

    /// Copy constructor.
    ref(const ref& r) : mPtr(r.mPtr)
    {
        if (mPtr)
            incRef((const Object*)(mPtr));
    }

    /// Construct a reference from a convertible reference.
    template<typename T2 = T>
    ref(const ref<T2>& r) : mPtr(r.mPtr)
    {
        static_assert(std::is_base_of_v<Object, T>, "Cannot create reference to object not inheriting from Object class.");
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot create reference to object from unconvertible reference.");
        if (mPtr)
            incRef((const Object*)(mPtr));
    }

    /// Move constructor.
    ref(ref&& r) noexcept
        : mPtr(r.mPtr)
#if FALCOR_ENABLE_REF_TRACKING
        , mRefId(r.mRefId)
#endif
    {
        r.mPtr = nullptr;
#if FALCOR_ENABLE_REF_TRACKING
        r.mRefId = uint64_t(-1);
#endif
    }

    /// Construct a reference by moving from a convertible reference.
    template<typename T2>
    ref(ref<T2>&& r) noexcept
        : mPtr(r.mPtr)
#if FALCOR_ENABLE_REF_TRACKING
        , mRefId(r.mRefId)
#endif
    {
        static_assert(std::is_base_of_v<Object, T>, "Cannot create reference to object not inheriting from Object class.");
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot create reference to object from unconvertible reference.");
        r.mPtr = nullptr;
#if FALCOR_ENABLE_REF_TRACKING
        r.mRefId = uint64_t(-1);
#endif
    }

    /// Destructor.
    ~ref()
    {
        if (mPtr)
            decRef((const Object*)(mPtr));
    }

    /// Assign another reference into the current one.
    ref& operator=(const ref& r) noexcept
    {
        if (r != *this)
        {
            if (r.mPtr)
                incRef((const Object*)(r.mPtr));
            T* prevPtr = mPtr;
            mPtr = r.mPtr;
            if (prevPtr)
                decRef((const Object*)(prevPtr));
        }
        return *this;
    }

    /// Assign another convertible reference into the current one.
    template<typename T2>
    ref& operator=(const ref<T2>& r) noexcept
    {
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot assign reference to object from unconvertible reference.");
        if (r != *this)
        {
            if (r.mPtr)
                incRef((const Object*)(r.mPtr));
            T* prevPtr = mPtr;
            mPtr = r.mPtr;
            if (prevPtr)
                decRef((const Object*)(prevPtr));
        }
        return *this;
    }

    /// Move another reference into the current one.
    ref& operator=(ref&& r) noexcept
    {
        if (static_cast<void*>(&r) != this)
        {
            if (mPtr)
                decRef((const Object*)(mPtr));
            mPtr = r.mPtr;
            r.mPtr = nullptr;
#if FALCOR_ENABLE_REF_TRACKING
            mRefId = r.mRefId;
            r.mRefId = uint64_t(-1);
#endif
        }
        return *this;
    }

    /// Move another convertible reference into the current one.
    template<typename T2>
    ref& operator=(ref<T2>&& r) noexcept
    {
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot move reference to object from unconvertible reference.");
        if (static_cast<void*>(&r) != this)
        {
            if (mPtr)
                decRef((const Object*)(mPtr));
            mPtr = r.mPtr;
            r.mPtr = nullptr;
#if FALCOR_ENABLE_REF_TRACKING
            mRefId = r.mRefId;
            r.mRefId = uint64_t(-1);
#endif
        }
        return *this;
    }

    /// Overwrite this reference with a pointer to another object
    template<typename T2 = T>
    void reset(T2* ptr = nullptr) noexcept
    {
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot assign reference to object from unconvertible pointer.");
        if (ptr != mPtr)
        {
            if (ptr)
                incRef((const Object*)(ptr));
            T* prevPtr = mPtr;
            mPtr = ptr;
            if (prevPtr)
                decRef((const Object*)(prevPtr));
        }
    }

    /// Compare this reference to another reference.
    template<typename T2 = T>
    bool operator==(const ref<T2>& r) const
    {
        static_assert(
            std::is_convertible_v<T2*, T*> || std::is_convertible_v<T*, T2*>, "Cannot compare references of non-convertible types."
        );
        return mPtr == r.mPtr;
    }

    /// Compare this reference to another reference.
    template<typename T2 = T>
    bool operator!=(const ref<T2>& r) const
    {
        static_assert(
            std::is_convertible_v<T2*, T*> || std::is_convertible_v<T*, T2*>, "Cannot compare references of non-convertible types."
        );
        return mPtr != r.mPtr;
    }

    /// Compare this reference to another reference.
    template<typename T2 = T>
    bool operator<(const ref<T2>& r) const
    {
        static_assert(
            std::is_convertible_v<T2*, T*> || std::is_convertible_v<T*, T2*>, "Cannot compare references of non-convertible types."
        );
        return mPtr < r.mPtr;
    }

    /// Compare this reference to a pointer.
    template<typename T2 = T>
    bool operator==(const T2* ptr) const
    {
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot compare reference to pointer of non-convertible types.");
        return mPtr == ptr;
    }

    /// Compare this reference to a pointer.
    template<typename T2 = T>
    bool operator!=(const T2* ptr) const
    {
        static_assert(std::is_convertible_v<T2*, T*>, "Cannot compare reference to pointer of non-convertible types.");
        return mPtr != ptr;
    }

    /// Compare this reference to a null pointer.
    bool operator==(std::nullptr_t) const { return mPtr == nullptr; }

    /// Compare this reference to a null pointer.
    bool operator!=(std::nullptr_t) const { return mPtr != nullptr; }

    /// Compare this reference to a null pointer.
    bool operator<(std::nullptr_t) const { return mPtr < nullptr; }

    /// Access the object referenced by this reference.
    T* operator->() const { return mPtr; }

    /// Return a C++ reference to the referenced object.
    T& operator*() const { return *mPtr; }

    /// Return a pointer to the referenced object.
    T* get() const { return mPtr; }

    /// Check if the object is defined
    operator bool() const { return mPtr != nullptr; }

    /// Swap this reference with another reference.
    void swap(ref& r) noexcept
    {
        std::swap(mPtr, r.mPtr);
#if FALCOR_ENABLE_REF_TRACKING
        std::swap(mRefId, r.mRefId);
#endif
    }

private:
    inline void incRef(const Object* object)
    {
#if FALCOR_ENABLE_REF_TRACKING
        object->incRef(mRefId);
#else
        object->incRef();
#endif
    }

    inline void decRef(const Object* object)
    {
#if FALCOR_ENABLE_REF_TRACKING
        object->decRef(mRefId);
#else
        object->decRef();
#endif
    }

    T* mPtr{nullptr};
#if FALCOR_ENABLE_REF_TRACKING
    uint64_t mRefId{nextRefId()};
#endif

private:
    template<typename T2>
    friend class ref;
};

template<class T, class... Args>
ref<T> make_ref(Args&&... args)
{
    return ref<T>(new T(std::forward<Args>(args)...));
}

template<class T, class U>
ref<T> static_ref_cast(const ref<U>& r) noexcept
{
    return ref<T>(static_cast<T*>(r.get()));
}

template<class T, class U>
ref<T> dynamic_ref_cast(const ref<U>& r) noexcept
{
    return ref<T>(dynamic_cast<T*>(r.get()));
}

/**
 * @brief Breakable reference counting helper for avoding reference cycles.
 *
 * This helper represents a strong reference (ref<T>) that can be broken.
 * This is accomplished by storing both a strong reference and a raw pointer.
 * When the strong reference is broken, we access the referenced object through
 * the raw pointer.
 *
 * This helper can be used in scenarios where some object holds nested objects
 * that themselves hold a reference to the parent object. In such cases, the
 * nested objects should hold a breakable reference to the parent object.
 * When the nested objects are created, we can immediately break the strong
 * reference to the parent object. This allows the parent object to be destroyed
 * when all of the external references to it are released.
 *
 * This helper can be used in place of a @a ref, but it cannot be reassigned.
 */
template<typename T>
class BreakableReference
{
public:
    BreakableReference(const ref<T>& r) : mStrongRef(r), mWeakRef(mStrongRef.get()) {}
    BreakableReference(ref<T>&& r) : mStrongRef(r), mWeakRef(mStrongRef.get()) {}

    BreakableReference() = delete;
    BreakableReference& operator=(const ref<T>&) = delete;
    BreakableReference& operator=(ref<T>&&) = delete;

    T* get() const { return mWeakRef; }
    T* operator->() const { return get(); }
    T& operator*() const { return *get(); }
    operator ref<T>() const { return ref<T>(get()); }
    operator T*() const { return get(); }
    operator bool() const { return get() != nullptr; }

    void breakStrongReference() { mStrongRef.reset(); }

private:
    ref<T> mStrongRef;
    T* mWeakRef = nullptr;
};

} // namespace Falcor

template<typename T>
struct fmt::formatter<Falcor::ref<T>> : formatter<const void*>
{
    template<typename FormatContext>
    auto format(const Falcor::ref<T>& ref, FormatContext& ctx) const
    {
        return formatter<const void*>::format(ref.get(), ctx);
    }
};

template<typename T>
struct fmt::formatter<Falcor::BreakableReference<T>> : formatter<const void*>
{
    template<typename FormatContext>
    auto format(const Falcor::BreakableReference<T>& ref, FormatContext& ctx) const
    {
        return formatter<const void*>::format(ref.get(), ctx);
    }
};

namespace std
{
template<typename T>
void swap(::Falcor::ref<T>& x, ::Falcor::ref<T>& y) noexcept
{
    return x.swap(y);
}

template<typename T>
struct hash<::Falcor::ref<T>>
{
    constexpr size_t operator()(const ::Falcor::ref<T>& r) const { return std::hash<T*>()(r.get()); }
};

} // namespace std
