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
#include "Core/API/Buffer.h"

#include <vector>

namespace Falcor
{
/**
 * Utility class for memory management of a GPU buffer.
 *
 * The class maintains a dynamically sized backing buffer on the CPU
 * in which memory can be allocated and updated.
 * The GPU buffer is lazily created and updated upon access.
 * The caller should not hold on to pointers into the buffers as the
 * memory may get reallocated at any time.
 *
 * BufferAllocator can enforce various alignment requirements,
 * including minimum byte alignment and (optionally) that allocated
 * objects don't span multiple cache lines if possible.
 * It is assumed that the base pointer of the GPU buffer starts at a
 * cache line. The implementation doesn't provide any alignment
 * guarantees for the CPU side buffer (where it doesn't matter anyway).
 */
class FALCOR_API BufferAllocator
{
public:
    /**
     * Create a buffer allocator.
     * @param[in] alignment Minimum alignment in bytes for any allocation.
     * @param[in] elementSize Element size for structured buffer. If zero a raw buffer is created.
     * @param[in] cacheLineSize Cache line size in bytes. Allocations are placed to not stride cache lines if it can be avoided.
     * @param[in] bindFlags Resource bind flags for the GPU buffer.
     */
    BufferAllocator(
        size_t alignment,
        size_t elementSize,
        size_t cacheLineSize = 128,
        ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    /**
     * Allocates a memory region.
     * @param[in] byteSize Amount of memory in bytes to allocate.
     * @return Offset in bytes to the allocated memory.
     */
    size_t allocate(size_t byteSize);

    /**
     * Allocates memory to hold an array of the given type.
     * @param[in] count Number of array elements.
     * @return Offset in bytes to the allocated memory.
     */
    template<typename T>
    size_t allocate(size_t count = 1)
    {
        return allocate(count * sizeof(T));
    }

    /**
     * Allocates an object of given type and copies the data.
     * @param[in] obj The object to copy.
     * @return Offset in bytes to the allocated object.
     */
    template<typename T>
    size_t pushBack(const T& obj)
    {
        const size_t byteSize = sizeof(T);
        computeAndAllocatePadding(byteSize);
        size_t byteOffset = allocInternal(byteSize);
        T* ptr = reinterpret_cast<T*>(mBuffer.data() + byteOffset);
        *ptr = obj;
        markAsDirty(byteOffset, byteSize);
        return byteOffset;
    }

    /**
     * Allocates an object of given type and executes its constructor.
     * @param[in] args Arguments to pass to the constructor.
     * @return Offset in bytes to the allocated object.
     */
    template<typename T, typename... Args>
    size_t emplaceBack(Args&&... args)
    {
        const size_t byteSize = sizeof(T);
        computeAndAllocatePadding(byteSize);
        size_t byteOffset = allocInternal(byteSize);
        void* ptr = mBuffer.data() + byteOffset;
        new (ptr) T(std::forward<Args>(args)...);
        markAsDirty(byteOffset, byteSize);
        return byteOffset;
    }

    /**
     * Set data into a memory region.
     * @param[in] pData Pointer to the source data.
     * @param[in] byteOffset Offset in bytes to the destination memory region.
     * @param[in] byteSize Size in bytes to copy.
     */
    void setBlob(const void* pData, size_t byteOffset, size_t byteSize);

    /**
     * Set an object of the given type. The memory must have been previously allocated.
     * @param[in] obj The object to set.
     * @param[in] byteOffset Offset in bytes of the object.
     */
    template<typename T>
    void set(size_t byteOffset, const T& obj)
    {
        setBlob(&obj, byteOffset, sizeof(T));
    }

    /**
     * Get an object of the given type.
     * @param[in] byteOffset Offset in bytes of the object.
     */
    template<typename T>
    const T& get(size_t byteOffset) const
    {
        return *reinterpret_cast<const uint8_t*>(mBuffer.data() + byteOffset);
    }

    /**
     * Mark memory region as modified. The GPU buffer will get updated.
     * @param[in] byteOffset Offset in bytes.
     * @param[in] byteSize Size in bytes.
     */
    void modified(size_t byteOffset, size_t byteSize);

    /**
     * Mark an object of the given type as modified. The GPU buffer will get updated.
     * @param[in] byteOffset Offset in bytes to the object.
     */
    template<typename T>
    void modified(size_t byteOffset)
    {
        modified(byteOffset, sizeof(T));
    }

    /**
     * Returns the pointer to the start of the allocated CPU buffer (read/write).
     * The pointer is transient and only valid until the next allocation operation.
     * This is for low-level access only. Use with care and call `modified` to mark any changes for the GPU buffer to get updated correctly.
     * @return Pointer to CPU side buffer.
     */
    uint8_t* getStartPointer() { return mBuffer.data(); }

    /**
     * Returns the pointer to the start of the allocated CPU buffer (read only).
     * The pointer is transient and only valid until the next allocation operation.
     * @return Pointer to CPU side buffer.
     */
    const uint8_t* getStartPointer() const { return mBuffer.data(); }

    /**
     * Get size of the buffer in bytes.
     * @return Size in bytes.
     */
    size_t getSize() const { return mBuffer.size(); }

    /**
     * Clear buffer. This removes all allocations.
     */
    void clear();

    /**
     * Get GPU buffer. The buffer is updated and ready for use.
     * The buffer is transient and only valid until the next allocation operation.
     */
    ref<Buffer> getGPUBuffer(ref<Device> pDevice);

private:
    void computeAndAllocatePadding(size_t byteSize);
    size_t allocInternal(size_t byteSize);

    struct Range
    {
        size_t start = 0;
        size_t end = 0;
        Range(){};
        Range(size_t s, size_t e) : start(s), end(e) {}
    };

    void markAsDirty(const Range& range);
    void markAsDirty(size_t byteOffset, size_t byteSize) { markAsDirty(Range(byteOffset, byteOffset + byteSize)); }

    /// Minimum alignment for allocations from base address. A value of zero means no aligment is performed.
    const size_t mAlignment;

    /// Element size for structured buffers. A value of zero means a raw buffer is created.
    const size_t mElementSize;

    /// Allocation are aligned to not span multiple cache lines (if possible). A value of zero means do not care about cache line alignment.
    const size_t mCacheLineSize;

    /// Bind flags for the GPU buffer.
    const ResourceBindFlags mBindFlags;

    /// Range of buffer that is dirty and needs to be updatd on the GPU. We track a single dirty range for now to minimize the number of
    /// uploads, but this could be changed.
    Range mDirty;

    std::vector<uint8_t> mBuffer; ///< CPU buffer holding a copy of the data.
    ref<Buffer> mpGpuBuffer;      ///< GPU buffer holding the data.
};
} // namespace Falcor
