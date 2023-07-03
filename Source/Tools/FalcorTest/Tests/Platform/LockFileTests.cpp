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
#include "Testing/UnitTest.h"
#include "Core/Platform/OS.h"
#include "Core/Platform/LockFile.h"

#include <atomic>
#include <future>
#include <thread>
#include <vector>

namespace Falcor
{
CPU_TEST(LockFile_Closed)
{
    LockFile file;
    EXPECT_FALSE(file.isOpen());

    // Allowed to close a closed file.
    file.close();
    EXPECT_FALSE(file.isOpen());
}

CPU_TEST(LockFile_OpenClose)
{
    const std::filesystem::path path = "test_lock_file_1";

    {
        LockFile file;
        EXPECT_TRUE(file.open(path));
        EXPECT_TRUE(file.isOpen());
        EXPECT_TRUE(std::filesystem::exists(path));
        file.close();
        EXPECT_FALSE(file.isOpen());
    }

    {
        LockFile file(path);
        EXPECT_TRUE(file.isOpen());
        EXPECT_TRUE(std::filesystem::exists(path));
        file.close();
        EXPECT_FALSE(file.isOpen());
    }

    // Cleanup.
    std::filesystem::remove(path);
    ASSERT_FALSE(std::filesystem::exists(path));
}

CPU_TEST(LockFile_ExclusiveLock)
{
    static const std::filesystem::path path = "test_lock_file_2";
    static std::atomic<uint32_t> lockCounter;
    static std::atomic<uint32_t> unlockCounter;

    struct LockTask
    {
        std::thread thread;
        std::promise<void> startPromise;
        std::future<void> startFuture;
        LockFile lockFile;
        bool openResult = false;
        bool tryLockSharedResult = false;
        bool tryLockExclusiveResult = false;
        bool lockResult = false;
        bool unlockResult = false;
        uint32_t lockIteration = 0;
        uint32_t unlockIteration = 0;

        LockTask() : startFuture(startPromise.get_future()), lockFile(path) { openResult = lockFile.isOpen(); }

        void run()
        {
            tryLockSharedResult = lockFile.tryLock(LockFile::LockType::Shared);
            tryLockExclusiveResult = lockFile.tryLock(LockFile::LockType::Exclusive);
            startPromise.set_value();
            lockResult = lockFile.lock(LockFile::LockType::Exclusive);
            lockIteration = lockCounter.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            unlockIteration = unlockCounter.fetch_add(1);
            unlockResult = lockFile.unlock();
        }
    };

    // Acquire lock from main thread.
    LockFile lockFile(path);
    EXPECT_TRUE(lockFile.isOpen());
    EXPECT_TRUE(lockFile.lock(LockFile::LockType::Exclusive));

    // Make sure we cannot acquire the lock in non-blocking mode from a second instance.
    LockFile lockFile2(path);
    EXPECT_TRUE(lockFile2.isOpen());
    EXPECT_FALSE(lockFile2.tryLock(LockFile::LockType::Shared));
    EXPECT_FALSE(lockFile2.tryLock(LockFile::LockType::Exclusive));

    // Start a number of threads and wait for them to start up.
    // Each thread immediately tries to acquire the lock in non-blocking mode (expected to fail).
    // Next each thread acquires the lock in blocking mode.
    std::vector<LockTask> tasks(32);
    for (auto& task : tasks)
    {
        task.thread = std::thread(&LockTask::run, &task);
        task.startFuture.wait();
    }

    // Make sure none of the threads were able to acquire the lock yet.
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_EQ(lockCounter.load(), 0);

    // Release the lock from the main thread. This will allow all the other
    // threads to acquire the lock, one after the other.
    EXPECT_TRUE(lockFile.unlock());

    // Wait for all threads to finish and make sure they behaved as expected.
    std::vector<bool> lockIterationUsed(tasks.size(), false);
    std::vector<bool> unlockIterationUsed(tasks.size(), false);
    for (auto& task : tasks)
    {
        task.thread.join();

        EXPECT_TRUE(task.openResult);
        EXPECT_FALSE(task.tryLockSharedResult);
        EXPECT_FALSE(task.tryLockExclusiveResult);
        EXPECT_TRUE(task.lockResult);
        EXPECT_TRUE(task.unlockResult);
        ASSERT_LT(task.lockIteration, lockIterationUsed.size());
        ASSERT_LT(task.unlockIteration, unlockIterationUsed.size());
        EXPECT_EQ(task.unlockIteration, task.lockIteration);
        EXPECT_FALSE(lockIterationUsed[task.lockIteration]);
        EXPECT_FALSE(unlockIterationUsed[task.unlockIteration]);
        lockIterationUsed[task.lockIteration] = true;
        unlockIterationUsed[task.unlockIteration] = true;
    }

    // Ensure all threads did manage to acquire the lock.
    EXPECT_EQ(lockCounter.load(), tasks.size());
    EXPECT_EQ(unlockCounter.load(), tasks.size());

    // Check that we can now acquire the lock in non-blocking mode.
    EXPECT_TRUE(lockFile2.tryLock(LockFile::LockType::Exclusive));
    EXPECT_TRUE(lockFile2.unlock());

    // Close all lock files.
    lockFile.close();
    lockFile2.close();
    tasks.clear();

    // Cleanup.
    std::filesystem::remove(path);
    ASSERT_FALSE(std::filesystem::exists(path));
}

} // namespace Falcor
