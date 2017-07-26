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

#include <Utils/Profiler.h>
#include <Cuda/CudaInterop.h>
#include <Cuda/CudaContext.h>

namespace Falcor
{
	namespace Cuda
	{
		/** A Cuda-specific adaption of \ref Falcor::Profiler.
		    The derived event structure provides integration of Cuda timer queries while keeping all Cuda related functionality to the Faclor::Cuda module.
			The interface is adapted such that a non-Cuda profiler can be changed to a Cuda-awere profiler simply by adding \c Cuda:: in the code.
		*/
		class Profiler
		{
		public:
			/** Cuda version of \ref Falcor::Profiler::EventData.
			*/
			struct EventData : public Falcor::Profiler::EventData
			{
				/** These are the two members that should not be introduced in the basic Falcor module.
				*/
				CUevent startEvent,
						stopEvent;
				EventData();
				~EventData();
			};

			static EventData* createNewCudaEvent(const HashedString& name);
			static void       startEvent(const HashedString& name, EventData *pEvent);
	        static void       endEvent(const HashedString& name, EventData *pEvent);
			static void       startEvent(const HashedString& name);
	        static void       endEvent(const HashedString& name);
			static EventData* getEvent(const HashedString& name);
			static EventData* isEventRegistered(const HashedString& name);

		};

		/** Same as Falcor::ProfilerEvent, just with creating Cuda events.
		*/
	    class ProfilerEvent
	    {
	    public:
		    ProfilerEvent(const HashedString& name) : mName(name) { if(gProfileEnabled) { Profiler::startEvent(name); } }
			~ProfilerEvent() { if(gProfileEnabled) {Profiler::endEvent(mName); }}
	    private:
		    const HashedString mName;
		};
	}
}
#if _PROFILING_ENABLED
#define PROFILE_CUDA(_name) static const Falcor::HashedString hashed ## _name(#_name); Falcor::Cuda::ProfilerEvent _profileEvent(hashed ## _name);
#else
#define PROFILE_CUDA(_name)
#endif