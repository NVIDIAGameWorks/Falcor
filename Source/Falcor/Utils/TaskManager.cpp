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
#include "TaskManager.h"

namespace Falcor
{

TaskManager::TaskManager(bool startPaused)
{
    if (startPaused)
        mThreadPool.pause();
}

void TaskManager::addTask(CpuTask&& task)
{
    std::lock_guard<std::mutex> l(mTaskMutex);
    ++mCurrentlyScheduled;
    mThreadPool.push_task(
        [task = std::move(task), this]() mutable
        {
            ++mCurrentlyRunning;
            --mCurrentlyScheduled;
            executeCpuTask(std::move(task));
            size_t running = --mCurrentlyRunning;
            // If nothing is running, lets wake up and try to exit.
            if (running == 0)
                mGpuTaskCond.notify_all();
        }
    );
}

void TaskManager::addTask(GpuTask&& task)
{
    std::lock_guard<std::mutex> l(mTaskMutex);
    ++mCurrentlyScheduled;
    mGpuTasks.push_back(std::move(task));
    mGpuTaskCond.notify_all();
}

void TaskManager::finish(RenderContext* renderContext)
{
    mThreadPool.unpause();
    while (true)
    {
        while (true)
        {
            std::unique_lock<std::mutex> l(mTaskMutex);
            if (mGpuTasks.empty())
                break;
            auto task = std::move(mGpuTasks.back());
            mGpuTasks.pop_back();
            l.unlock();
            ++mCurrentlyRunning;
            --mCurrentlyScheduled;
            task(renderContext);
            --mCurrentlyRunning;
        }

        std::unique_lock<std::mutex> l(mTaskMutex);
        while (true)
        {
            // If there are GPU tasks, go do them
            if (!mGpuTasks.empty())
                break;
            // If there are absolutely no tasks, go finish
            if (mCurrentlyRunning == 0 && mCurrentlyScheduled == 0)
                break;
            // Otherwise wait for either a new GPU task, or last running to notify us to check
            mGpuTaskCond.wait(l);
        }

        if (mCurrentlyRunning == 0 && mCurrentlyScheduled == 0)
            break;
    }
    rethrowException();
}

void TaskManager::storeException()
{
    std::lock_guard<std::mutex> l(mExceptionMutex);
    mException = std::current_exception();
}

void TaskManager::rethrowException()
{
    std::lock_guard<std::mutex> l(mExceptionMutex);
    if (mException)
        std::rethrow_exception(mException);
}

void TaskManager::executeCpuTask(CpuTask&& task)
{
    try
    {
        task();
    }
    catch (...)
    {
        storeException();
    }
}

} // namespace Falcor
