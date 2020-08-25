/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <thread>

namespace Falcor
{
    class dlldecl Threading
    {
    public:
        const static uint32_t kDefaultThreadCount = 16;

        /** Handle to a dispatched task

            TODO: Implementation
        */
        class Task
        {
        public:
            /** Check if task is still executing
            */
            bool isRunning();

            /** Wait for task to finish executing
            */
            void finish();

        private:
            Task();
            friend class Threading;
        };

        /** Initializes the global thread pool
            \param[in] threadCount Number of threads in the pool
        */
        static void start(uint32_t threadCount = kDefaultThreadCount);

        /** Waits for all currently executing threads to finish
        */
        static void finish();

        /** Waits for all currently executing threads to finish and shuts down the thread pool
        */
        static void shutdown();

        /** Returns the maximum number of concurrent threads supported by the hardware
        */
        static uint32_t getLogicalThreadCount() { return std::thread::hardware_concurrency(); }

        /** Starts a task on an available thread.
            \return Handle to the task
        */
        static Task dispatchTask(const std::function<void(void)>& func);
    };
}
