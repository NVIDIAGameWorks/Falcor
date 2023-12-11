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

#include <BS_thread_pool.hpp>

#include <functional>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include <exception>

namespace Falcor
{
class RenderContext;
class FALCOR_API TaskManager
{
public:
    using CpuTask = std::function<void()>;
    using GpuTask = std::function<void(RenderContext* renderContext)>;

public:
    TaskManager(bool startPaused = false);

    /// Adds a CPU only task to the manager, if unpaused, the task starts right away
    void addTask(CpuTask&& task);
    /// Adds a GPU task to the manager, GPU tasks only start in the finish call and are sequential
    void addTask(GpuTask&& task);

    /// Unpauses and waits for all tasks to finish.
    /// The renderContext might be needed even if the TaskManager contains no GPU tasks,
    /// as those could be spawned from the CPU tasks
    void finish(RenderContext* renderContext);

private:
    /// Thread safe way to store an exception
    void storeException();
    /// Thread safe way to retrow a stored exception
    void rethrowException();
    /// CPU task execution wrapped so it stores exception if the task throws
    void executeCpuTask(CpuTask&& task);

private:
    BS::thread_pool mThreadPool;
    std::atomic_size_t mCurrentlyRunning{0};
    std::atomic_size_t mCurrentlyScheduled{0};

    std::mutex mTaskMutex;
    std::condition_variable mGpuTaskCond;
    std::vector<GpuTask> mGpuTasks;

    std::mutex mExceptionMutex;
    std::exception_ptr mException;
};

} // namespace Falcor
