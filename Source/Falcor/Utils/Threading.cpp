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
#include "stdafx.h"
#include "Threading.h"

namespace Falcor
{
    namespace
    {
        struct ThreadingData
        {
            bool initialized = false;
            std::vector<std::thread> threads;
            uint32_t current;
        } gData;
    }

    void Threading::start(uint32_t threadCount)
    {
        if (gData.initialized) return;

        gData.threads.resize(threadCount);
        gData.initialized = true;
    }

    void Threading::shutdown()
    {
        for (auto& t : gData.threads)
        {
            if (t.joinable()) t.join();
        }

        gData.initialized = false;
    }

    Threading::Task Threading::dispatchTask(const std::function<void(void)>& func)
    {
        assert(gData.initialized);

        std::thread& t = gData.threads[gData.current];
        if (t.joinable()) t.join();
        t = std::thread(func);
        gData.current = (gData.current + 1) % gData.threads.size();

        return Task();
    }

    void Threading::finish()
    {
        for (auto& t : gData.threads)
        {
            if (t.joinable()) t.join();
        }
    }

    Threading::Task::Task()
    {
    }

    bool Threading::Task::isRunning()
    {
        logError("Threading::Task::isRunning() not implemented");
        return true;
    }

    void Threading::Task::finish()
    {
        logError("Threading::Task::finish() not implemented");
    }
}
