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
#include "Profiler.h"
#include "Core/API/GpuTimer.h"
#include <sstream>
#include <fstream>
#define USE_PIX
#include "WinPixEventRuntime/Include/WinPixEventRuntime/pix3.h"

namespace Falcor
{
    bool gProfileEnabled = false;

    std::unordered_map<std::string, Profiler::EventData*> Profiler::sProfilerEvents;
    std::vector<Profiler::EventData*> Profiler::sRegisteredEvents;
    std::string curEventName = "";
    uint32_t Profiler::sCurrentLevel = 0;
    uint32_t Profiler::sGpuTimerIndex = 0;

    void Profiler::initNewEvent(EventData *pEvent, const std::string& name)
    {
        pEvent->name = name;
        sProfilerEvents[curEventName] = pEvent;
    }

    Profiler::EventData* Profiler::createNewEvent(const std::string& name)
    {
        EventData *pData = new EventData;
        initNewEvent(pData, name);
        return pData;
    }

    Profiler::EventData* Profiler::isEventRegistered(const std::string& name)
    {
        auto event = sProfilerEvents.find(name);
        return (event == sProfilerEvents.end()) ? nullptr : event->second;
    }

    Profiler::EventData* Profiler::getEvent(const std::string& name)
    {
        auto event = isEventRegistered(name);
        return event ? event : createNewEvent(name);
    }

    void Profiler::startEvent(const std::string& name, Flags flags, bool showInMsg)
    {
        if (gProfileEnabled && is_set(flags, Flags::Internal))
        {
            curEventName = curEventName + "#" + name;
            EventData* pData = getEvent(curEventName);
            pData->triggered++;
            if (pData->triggered > 1)
            {
                logWarning("Profiler event '" + name + "' was triggered while it is already running. Nesting profiler events with the same name is disallowed and you should probably fix that. Ignoring the new call");
                return;
            }

            pData->showInMsg = showInMsg;
            pData->level = sCurrentLevel;
            pData->cpuStart = CpuTimer::getCurrentTimePoint();
            EventData::FrameData& frame = pData->frameData[sGpuTimerIndex];
            if (frame.currentTimer >= frame.pTimers.size())
            {
                frame.pTimers.push_back(GpuTimer::create());
            }
            frame.pTimers[frame.currentTimer]->begin();
            pData->callStack.push(frame.currentTimer);
            frame.currentTimer++;
            sCurrentLevel++;

            if (!pData->registered)
            {
                sRegisteredEvents.push_back(pData);
                pData->registered = true;
            }
        }
        if (is_set(flags, Flags::Pix))
        {
            PIXBeginEvent((ID3D12GraphicsCommandList*)gpDevice->getRenderContext()->getLowLevelData()->getCommandList(), PIX_COLOR(0, 0, 0), name.c_str());
        }
    }

    void Profiler::endEvent(const std::string& name, Flags flags)
    {
        if (gProfileEnabled && is_set(flags, Flags::Internal))
        {
            assert(isEventRegistered(curEventName));
            EventData* pData = getEvent(curEventName);
            pData->triggered--;
            if (pData->triggered != 0) return;

            pData->cpuEnd = CpuTimer::getCurrentTimePoint();
            pData->cpuTotal += CpuTimer::calcDuration(pData->cpuStart, pData->cpuEnd);

            pData->frameData[sGpuTimerIndex].pTimers[pData->callStack.top()]->end();
            pData->callStack.pop();

            sCurrentLevel--;
            curEventName.erase(curEventName.find_last_of("#"));
        }
        if (is_set(flags, Flags::Pix))
        {
            PIXEndEvent((ID3D12GraphicsCommandList*)gpDevice->getRenderContext()->getLowLevelData()->getCommandList());
        }
    }

    double Profiler::getEventGpuTime(const std::string& name)
    {
        const auto& pEvent = getEvent(name);
        return pEvent ? getGpuTime(pEvent) : 0;
    }

    double Profiler::getEventCpuTime(const std::string& name)
    {
        const auto& pEvent = getEvent(name);
        return pEvent ? getCpuTime(pEvent) : 0;
    }

    double Profiler::getGpuTime(const EventData* pData)
    {
        double gpuTime = 0;
        for (size_t i = 0; i < pData->frameData[1 - sGpuTimerIndex].currentTimer; i++)
        {
            gpuTime += pData->frameData[1 - sGpuTimerIndex].pTimers[i]->getElapsedTime();
        }
        return gpuTime;
    }

    double Profiler::getCpuTime(const EventData* pData)
    {
        return pData->cpuTotal;
    }

    std::string Profiler::getEventsString()
    {
        std::string results("Name\t\t\t\t\tCPU time(ms)\t\t  GPU time(ms)\n");

        for (EventData* pData : sRegisteredEvents)
        {
            assert(pData->triggered == 0);
            if(pData->showInMsg == false) continue;

            double gpuTime = getGpuTime(pData);
            assert(pData->callStack.empty());

            char event[1000];
            uint32_t nameIndent = pData->level * 2 + 1;
            uint32_t cpuIndent = 30 - (nameIndent + (uint32_t)pData->name.substr(pData->name.find_last_of("#") + 1).size());
            snprintf(event, 1000, "%*s%s %*.2f (%.2f) %14.2f (%.2f)\n", nameIndent, " ", pData->name.substr(pData->name.find_last_of("#") + 1).c_str(), cpuIndent, getCpuTime(pData),
                     pData->cpuRunningAverageMS, gpuTime, pData->gpuRunningAverageMS);
#if _PROFILING_LOG == 1
            pData->cpuMs[pData->stepNr] = (float)pData->cpuTotal;
            pData->gpuMs[pData->stepNr] = (float)gpuTime;
            pData->stepNr++;
            if (pData->stepNr == _PROFILING_LOG_BATCH_SIZE)
            {
                std::ostringstream logOss, fileOss;
                logOss << "dumping " << "profile_" << pData->name << "_" << pData->filesWritten;
                logInfo(logOss.str());
                fileOss << "profile_" << pData->name << "_" << pData->filesWritten++;
                std::ofstream out(fileOss.str().c_str());
                for (int i = 0; i < _PROFILING_LOG_BATCH_SIZE; ++i)
                {
                    out << pData->cpuMs[i] << " " << pData->gpuMs[i] << "\n";
                }
                pData->stepNr = 0;
            }
#endif
            results += event;
        }

        return results;
    }

    void Profiler::endFrame()
    {
        for (EventData* pData : sRegisteredEvents)
        {
            // Update CPU/GPU time running averages.
            const double cpuTime = getCpuTime(pData);
            const double gpuTime = getGpuTime(pData);
            // With sigma = 0.98, then after 100 frames, a given value's contribution is down to ~1.7% of
            // the running average, which seems to provide a reasonable trade-off of temporal smoothing
            // versus setting in to a new value when something has changed.
            const double sigma = .98;
            if (pData->cpuRunningAverageMS < 0.) pData->cpuRunningAverageMS = cpuTime;
            else pData->cpuRunningAverageMS = sigma * pData->cpuRunningAverageMS + (1. - sigma) * cpuTime;
            if (pData->gpuRunningAverageMS < 0.) pData->gpuRunningAverageMS = gpuTime;
            else pData->gpuRunningAverageMS = sigma * pData->gpuRunningAverageMS + (1. - sigma) * gpuTime;

            pData->showInMsg = false;
            pData->cpuTotal = 0;
            pData->triggered = 0;
            pData->frameData[1 - sGpuTimerIndex].currentTimer = 0;
            pData->registered = false;
        }
        sRegisteredEvents.clear();
        sGpuTimerIndex = 1 - sGpuTimerIndex;
    }

#if _PROFILING_LOG == 1
    void Profiler::flushLog()
    {
        for (EventData* pData : sRegisteredEvents)
        {
            std::ostringstream logOss, fileOss;
            logOss << "dumping " << "profile_" << pData->name << "_" << pData->filesWritten;
            logInfo(logOss.str());
            fileOss << "profile_" << pData->name << "_" << pData->filesWritten++;
            std::ofstream out(fileOss.str().c_str());
            for (int i = 0; i < pData->stepNr; ++i)
            {
                out << pData->cpuMs[i] << " " << pData->gpuMs[i] << "\n";
            }
            pData->stepNr = 0;
        }
    }
#endif

    void Profiler::clearEvents()
    {
        for (EventData* pData : sRegisteredEvents)
        {
            delete pData;
        }
        sProfilerEvents.clear();
        sRegisteredEvents.clear();
        sCurrentLevel = 0;
        sGpuTimerIndex = 0;
        curEventName = "";
    }
}
