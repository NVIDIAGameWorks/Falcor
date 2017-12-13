/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "Framework.h"
#include "Profiler.h"
#include "API/GpuTimer.h"
#include "API/LowLevel/FencedPool.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

namespace Falcor
{
    bool gProfileEnabled = false;

    std::map<size_t, Profiler::EventData*> Profiler::sProfilerEvents;
    uint32_t Profiler::sCurrentLevel = 0;
    uint32_t Profiler::sGpuTimerIndex = 0;
    std::vector<Profiler::EventData*> Profiler::sProfilerVector;
    
    std::hash<std::string> HashedString::hashFunc;

	void Profiler::initNewEvent(EventData *pEvent, const HashedString& name)
    {
	    pEvent->name = name.str;
        pEvent->level = sCurrentLevel;
		sProfilerEvents[name.hash] = pEvent;
        sProfilerVector.push_back(pEvent);
	}

    Profiler::EventData* Profiler::createNewEvent(const HashedString& name)
    {
        EventData *pData = new EventData;
		initNewEvent(pData, name);
        return pData;
    }

    Profiler::EventData* Profiler::isEventRegistered(const HashedString& name)
	{
        auto event = sProfilerEvents.find(name.hash);
        if(event == sProfilerEvents.end())
		{
			return nullptr;
		}
		else
		{
			return event->second;
		}
	}

    Profiler::EventData* Profiler::getEvent(const HashedString& name)
    {
        auto event = isEventRegistered(name);
        if(event)
		{
			return event;
		}
		else
        {
            return createNewEvent(name);
        }
    }

    void Profiler::startEvent(const HashedString& name, EventData* pData)
    {
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
    }

	void Profiler::endEvent(const HashedString& name, EventData* pData)
    {
        pData->cpuEnd = CpuTimer::getCurrentTimePoint();
        pData->cpuTotal += CpuTimer::calcDuration(pData->cpuStart, pData->cpuEnd);

        pData->frameData[sGpuTimerIndex].pTimers[pData->callStack.top()]->end();
        pData->callStack.pop();

        sCurrentLevel--;
    }

    void Profiler::endFrame(std::string& profileResults)
    {
        profileResults = "Name\t\t\tCPU time(ms)\t\t\tGPU time(ms)\n";

		for (EventData* pData : sProfilerVector)
		{
            double gpuTime = 0;
            for(size_t i = 0 ; i < pData->frameData[1 - sGpuTimerIndex].currentTimer ; i++)
            {
                gpuTime += pData->frameData[1 - sGpuTimerIndex].pTimers[i]->getElapsedTime();
            }

            pData->frameData[1 - sGpuTimerIndex].currentTimer = 0;
            assert(pData->callStack.empty());

			char event[1000];
			uint32_t nameIndent = pData->level * 2 + 1;
			uint32_t cpuIndent = 32 - (nameIndent + (uint32_t)pData->name.size());
			std::snprintf(event, 1000, "%*s%s %*.3f %36.3f\n", nameIndent, " ", pData->name.c_str(), cpuIndent, pData->cpuTotal, gpuTime);
#if _PROFILING_LOG == 1
			pData->cpuMs[pData->stepNr] = pData->cpuTotal;
			pData->gpuMs[pData->stepNr] = gpuTime;
			pData->stepNr++;
			if (pData->stepNr == _PROFILING_LOG_BATCH_SIZE)
			{
				std::ostringstream logOss, fileOss;
				logOss << "dumping " << "profile_" << pData->name << "_" << pData->filesWritten;
				Logger::log(Logger::Level::Info, logOss.str());
				fileOss << "profile_" << pData->name << "_" << pData->filesWritten++;
				std::ofstream out(fileOss.str().c_str());
				for (int i = 0; i < _PROFILING_LOG_BATCH_SIZE; ++i)
				{
				 	out << pData->cpuMs[i] << " " << pData->gpuMs[i] << "\n";
				}
				pData->stepNr = 0;
			}
#endif
            pData->cpuTotal = 0;
			pData->gpuTotal = 0;
            profileResults += event;
        }

        sGpuTimerIndex = 1 - sGpuTimerIndex;
    }

#if _PROFILING_LOG == 1
	void Profiler::flushLog() {
		for (EventData* pData : sProfilerVector)
		{
				std::ostringstream logOss, fileOss;
				logOss << "dumping " << "profile_" << pData->name << "_" << pData->filesWritten;
				Logger::log(Logger::Level::Info, logOss.str());
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
        for (EventData* pData : sProfilerVector)
        {
            delete pData;
        }
        sProfilerEvents.clear();
        sProfilerVector.clear();
        sCurrentLevel = 0;
        sGpuTimerIndex = 0;
    }
}