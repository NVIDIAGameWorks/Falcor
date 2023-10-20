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
#include "LockFile.h"

#if FALCOR_WINDOWS
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#elif FALCOR_LINUX
#include <unistd.h>
#include <sys/types.h>
#include <sys/file.h>
#else
#error "Unknown OS"
#endif

namespace Falcor
{

LockFile::LockFile(const std::filesystem::path& path)
{
    open(path);
}

LockFile::~LockFile()
{
    close();
}

bool LockFile::open(const std::filesystem::path& path)
{
#if FALCOR_WINDOWS
    mFileHandle = ::CreateFileW(
        path.c_str(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
        NULL
    );
    mIsOpen = mFileHandle != INVALID_HANDLE_VALUE;
#elif FALCOR_LINUX
    mFileHandle = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);
    mIsOpen = mFileHandle != -1;
#endif
    return mIsOpen;
}

void LockFile::close()
{
    if (!mIsOpen)
        return;

#if FALCOR_WINDOWS
    ::CloseHandle(mFileHandle);
#elif FALCOR_LINUX
    ::close(mFileHandle);
#endif
    mIsOpen = false;
}

bool LockFile::tryLock(LockType lockType)
{
    if (!mIsOpen)
        return false;

    bool success = false;
#if FALCOR_WINDOWS
    OVERLAPPED overlapped = {0};
    DWORD flags = lockType == LockType::Shared ? LOCKFILE_FAIL_IMMEDIATELY : (LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY);
    success = ::LockFileEx(mFileHandle, flags, DWORD(0), ~DWORD(0), ~DWORD(0), &overlapped);
#elif FALCOR_LINUX
    int operation = lockType == LockType::Shared ? (LOCK_SH | LOCK_NB) : (LOCK_EX | LOCK_NB);
    success = ::flock(mFileHandle, operation) == 0;
#endif
    return success;
}

bool LockFile::lock(LockType lockType)
{
    if (!mIsOpen)
        return false;

    bool success = false;
#if FALCOR_WINDOWS
    OVERLAPPED overlapped = {0};
    overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL);
    DWORD flags = lockType == LockType::Shared ? 0 : LOCKFILE_EXCLUSIVE_LOCK;
    success = ::LockFileEx(mFileHandle, flags, DWORD(0), ~DWORD(0), ~DWORD(0), &overlapped);
    if (!success)
    {
        auto err = ::GetLastError();
        if (err == ERROR_IO_PENDING)
        {
            DWORD bytes;
            if (::GetOverlappedResult(mFileHandle, &overlapped, &bytes, TRUE))
                success = true;
        }
    }
    ::CloseHandle(overlapped.hEvent);
#elif FALCOR_LINUX
    int operation = lockType == LockType::Shared ? LOCK_SH : LOCK_EX;
    success = ::flock(mFileHandle, operation) == 0;
#endif
    return success;
}

bool LockFile::unlock()
{
    if (!mIsOpen)
        return false;

    bool success = false;
#if FALCOR_WINDOWS
    ::OVERLAPPED overlapped = {0};
    success = ::UnlockFileEx(mFileHandle, DWORD(0), ~DWORD(0), ~DWORD(0), &overlapped);
#elif FALCOR_LINUX
    success = ::flock(mFileHandle, LOCK_UN) == 0;
#endif
    return success;
}

} // namespace Falcor
