/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

#include <filesystem>

namespace Falcor
{

/**
 * Helper class abstracting lock files.
 * Uses LockFileEx() on Windows systems and flock() on POSIX systems.
 */
class FALCOR_API LockFile
{
public:
    enum class LockType
    {
        Exclusive,
        Shared,
    };

    LockFile() = default;

    /**
     * Construct and open the loc file. This will create the file if it doesn't exist yet.
     * @note Use isOpen() to check if the file was successfully opened.
     * @param path File path.
     */
    LockFile(const std::filesystem::path& path);

    ~LockFile();

    /**
     * Open the lock file. This will create the file if it doesn't exist yet.
     * @param path File path.
     * @return True if successful.
     */
    bool open(const std::filesystem::path& path);

    /// Closes the lock file.
    void close();

    /// Returns true if the lock file is open.
    bool isOpen() const { return mIsOpen; }

    /**
     * Acquire the lock in non-blocking mode.
     * @param lockType Lock type (Exclusive or Shared).
     * @return True if successful.
     */
    bool tryLock(LockType lockType = LockType::Exclusive);

    /**
     * Acquire the lock in blocking mode.
     * @param lockType Lock type (Exclusive or Shared).
     * @return True if successful.
     */
    bool lock(LockType lockType = LockType::Exclusive);

    /**
     * Release the lock.
     * @return True if successful.
     */
    bool unlock();

private:
    LockFile(const LockFile&) = delete;
    LockFile(LockFile&) = delete;
    LockFile& operator=(const LockFile&) = delete;
    LockFile& operator=(const LockFile&&) = delete;

#if FALCOR_WINDOWS
    using FileHandle = void*;
#elif FALCOR_LINUX
    using FileHandle = int;
#else
#error "Unknown OS"
#endif

    FileHandle mFileHandle;
    bool mIsOpen = false;
};

} // namespace Falcor
