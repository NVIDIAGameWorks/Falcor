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
    void Profiler::initNewEvent(EventData *pEvent, const std::string& name)
    {
        pEvent->name = name;
        mEvents[mCurEventName] = pEvent;
    }

    Profiler::EventData* Profiler::createNewEvent(const std::string& name)
    {
        EventData *pData = new EventData;
        initNewEvent(pData, name);
        return pData;
    }

    Profiler::EventData* Profiler::isEventRegistered(const std::string& name)
    {
        auto event = mEvents.find(name);
        return (event == mEvents.end()) ? nullptr : event->second;
    }

    Profiler::EventData* Profiler::getEvent(const std::string& name)
    {
        auto event = isEventRegistered(name);
        return event ? event : createNewEvent(name);
    }

    void Profiler::startEvent(const std::string& name, Flags flags, bool showInMsg)
    {
        if (mEnabled && is_set(flags, Flags::Internal))
        {
            mCurEventName = mCurEventName + "#" + name;
            EventData* pData = getEvent(mCurEventName);
            pData->triggered++;
            if (pData->triggered > 1)
            {
                logWarning("Profiler event '" + name + "' was triggered while it is already running. Nesting profiler events with the same name is disallowed and you should probably fix that. Ignoring the new call");
                return;
            }

            pData->showInMsg = showInMsg;
            pData->level = mCurrentLevel;
            pData->cpuStart = CpuTimer::getCurrentTimePoint();
            EventData::FrameData& frame = pData->frameData[mGpuTimerIndex];
            if (frame.currentTimer >= frame.pTimers.size())
            {
                frame.pTimers.push_back(GpuTimer::create());
            }
            frame.pTimers[frame.currentTimer]->begin();
            pData->callStack.push(frame.currentTimer);
            frame.currentTimer++;
            mCurrentLevel++;

            if (!pData->registered)
            {
                mRegisteredEvents.push_back(pData);
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
        if (mEnabled && is_set(flags, Flags::Internal))
        {
            assert(isEventRegistered(mCurEventName));
            EventData* pData = getEvent(mCurEventName);
            pData->triggered--;
            if (pData->triggered != 0) return;

            pData->cpuEnd = CpuTimer::getCurrentTimePoint();
            pData->cpuTotal += CpuTimer::calcDuration(pData->cpuStart, pData->cpuEnd);

            pData->frameData[mGpuTimerIndex].pTimers[pData->callStack.top()]->end();
            pData->callStack.pop();

            mCurrentLevel--;
            mCurEventName.erase(mCurEventName.find_last_of("#"));
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
        for (size_t i = 0; i < pData->frameData[1 - mGpuTimerIndex].currentTimer; i++)
        {
            gpuTime += pData->frameData[1 - mGpuTimerIndex].pTimers[i]->getElapsedTime();
        }
        return gpuTime;
    }

    double Profiler::getCpuTime(const EventData* pData)
    {
        return pData->cpuTotal;
    }

    std::string Profiler::getEventsString()
    {
        std::string results("Name                                    CPU time (ms)         GPU time (ms)\n");

        for (EventData* pData : mRegisteredEvents)
        {
            assert(pData->triggered == 0);
            if(pData->showInMsg == false) continue;

            double gpuTime = getGpuTime(pData);
            assert(pData->callStack.empty());

            char event[1000];
            std::string name = pData->name.substr(pData->name.find_last_of("#") + 1);
            uint32_t nameIndent = pData->level * 2 + 1;
            uint32_t cpuIndent = 45 - (nameIndent + (uint32_t)name.size());
            snprintf(event, 1000, "%*s%s %*.2f (%.2f) %14.2f (%.2f)\n", nameIndent, " ", name.c_str(), cpuIndent, getCpuTime(pData),
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
        for (EventData* pData : mRegisteredEvents)
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
            pData->frameData[1 - mGpuTimerIndex].currentTimer = 0;
            pData->registered = false;
        }
        mLastFrameEvents = std::move(mRegisteredEvents);
        mGpuTimerIndex = 1 - mGpuTimerIndex;
    }

#if _PROFILING_LOG == 1
    void Profiler::flushLog()
    {
        for (EventData* pData : mRegisteredEvents)
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
        for (auto& [_, pData] : mEvents) delete pData;
        mEvents.clear();
        mRegisteredEvents.clear();
        mLastFrameEvents.clear();
        mCurrentLevel = 0;
        mGpuTimerIndex = 0;
        mCurEventName = "";
    }

    const Profiler::SharedPtr& Profiler::instancePtr()
    {
        static Profiler::SharedPtr pInstance;
        if (!pInstance) pInstance = std::make_shared<Profiler>();
        return pInstance;
    }

    pybind11::dict Profiler::EventData::toPython() const
    {
        pybind11::dict d;

        d["name"] = name;
        d["cpuTime"] = cpuRunningAverageMS / 1000.f;
        d["gpuTime"] = gpuRunningAverageMS / 1000.f;

        return d;
    }

    SCRIPT_BINDING(Profiler)
    {
        auto getEvents = [] (Profiler* pProfiler) {
            pybind11::dict d;

            for (const Profiler::EventData* pData : pProfiler->getLastFrameEvents())
            {
                auto name = pData->name;
                d[name.c_str()] = pData->toPython();
            }

            return d;
        };

        pybind11::class_<Profiler, Profiler::SharedPtr> profiler(m, "Profiler");
        profiler.def_property("enabled", &Profiler::isEnabled, &Profiler::setEnabled);
        profiler.def_property_readonly("events", getEvents);
        profiler.def("clearEvents", &Profiler::clearEvents);
    }
}
