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
#pragma once
#include <string>
#include <map>
#include <functional>
#include <vector>
#include "API/GpuTimer.h"
#include "Utils/CpuTimer.h"
#include "FalcorConfig.h"
#include <stack>

namespace Falcor
{
    extern bool gProfileEnabled;

    class GpuTimer;

    /** Basic wrapper to calculate and store a string's hash value.
    */
    struct HashedString
    {
        static std::hash<std::string> hashFunc;

        HashedString(const std::string& s) : str(s), hash(hashFunc(s)) {}
        HashedString(const char* c) : str(c), hash(hashFunc(c)) {}

        const std::string str;
        const size_t hash;
    };

    /** Container class for CPU/GPU profiling.
        This class uses the most accurately available CPU and GPU timers to profile given events. It automatically creates event hierarchies based on the order of the calls made.
        This class uses a double-buffering scheme for GPU profiling to avoid GPU stalls.
        CProfilerEvent is a wrapper class which together with scoping can simplify event profiling.
    */
    class Profiler
    {
    public:

#if _PROFILING_LOG == 1
        static void flushLog();
#endif

        struct EventData
        {
            virtual ~EventData() {}
            std::string name;
            struct FrameData
            {
                std::vector<GpuTimer::SharedPtr> pTimers;
                size_t currentTimer = 0;
            };
            FrameData frameData[2]; // Double-buffering, to avoid GPU flushes
            bool showInMsg;
            std::stack<size_t> callStack;
            CpuTimer::TimePoint cpuStart;
            CpuTimer::TimePoint cpuEnd;
            float cpuTotal = 0;
            uint32_t level;
#if _PROFILING_LOG == 1
            int stepNr = 0;
            int filesWritten = 0;
            float cpuMs[_PROFILING_LOG_BATCH_SIZE];
            float gpuMs[_PROFILING_LOG_BATCH_SIZE];
#endif
        };

        /** Start profiling a new event and update the events hierarchies.
            \param[in] name The event name.
        */
        static void startEvent(const HashedString& name, bool showInMsg = true);

        /** Finish profiling a new event and update the events hierarchies.
            \param[in] name The event name.
        */
        static void endEvent(const HashedString& name);

        /** Finish profiling for the entire frame.
            Due to the double-buffering nature of the profiler, the results returned are for the previous frame.
            \param[out] profileResults A string containing the the profiling results.
        */
        static void endFrame();

        /** Get a string with the current frame results
        */
        static std::string getEventsString();

        /** Create a new event and register and initialize it using \ref initNewEvent.
            \param[in] name The event name.
        */
        static EventData* createNewEvent(const HashedString& name);
        
        /** Initialize a previously generated event.
            Used to do the default initialization without creating the actual event instance, to support derived event types. See \ref Cuda::Profiler::EventData.
            \param[out] pEvent Event to initialize
            \param[in] name New event name
        */
        static void initNewEvent(EventData *pEvent, const HashedString& name);

        /** Get the event, or create a new one if the event does not yet exist.
            This is a public interface to facilitate more complicated construction of event names and finegrained control over the profiled region.
        */
        static EventData* getEvent(const HashedString& name);

        /** Get the event, or create a new one if the event does not yet exist.
        This is a public interface to facilitate more complicated construction of event names and finegrained control over the profiled region.
        */
        static double getEventCpuTime(const HashedString& name);

        /** Get the event, or create a new one if the event does not yet exist.
        This is a public interface to facilitate more complicated construction of event names and finegrained control over the profiled region.
        */
        static double getEventGpuTime(const HashedString& name);

        /** Returns the event or \c nullptr if the event is not known.
            Can be used as a predicate.
        */
        static EventData* isEventRegistered(const HashedString& name);

        /** Clears all the events. 
            Useful if you want to start profiling a different technique with different events.
        */
        static void clearEvents();

    private:
        static double getGpuTime(const EventData* pData);
        static double getCpuTime(const EventData* pData);

        static std::map<size_t, EventData*> sProfilerEvents;
        static std::vector<EventData*> sProfilerVector;
        static uint32_t sCurrentLevel;
        static uint32_t sGpuTimerIndex;
    };

    /** Helper class for starting and ending profiling events.
        The class C'tor and D'tor call Profiler#StartEvent() and Profiler#EndEvent(). This class can be used with scoping to simplify event creation.\n
        The PROFILE macro wraps creation of local CProfilerEvent objects when profiling is enabled, and does nothing when profiling is disabled, so should be used instead of directly creating CProfilerEvent objects.
    */
    class ProfilerEvent
    {
    public:
        /** C'tor
        */
        ProfilerEvent(const HashedString& name) : mName(name) { if(gProfileEnabled) { Profiler::startEvent(name); } }
        /** D'tor
        */
        ~ProfilerEvent() { if(gProfileEnabled) {Profiler::endEvent(mName); }}

    private:
        const HashedString mName;
    };

#if _PROFILING_ENABLED
#define PROFILE(_name) static const Falcor::HashedString hashed ## _name(#_name); Falcor::ProfilerEvent _profileEvent(hashed ## _name);
#else
#define PROFILE(_name)
#endif
}
