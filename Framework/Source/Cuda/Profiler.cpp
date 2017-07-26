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
#include "Profiler.h"

namespace Falcor
{
	namespace Cuda
	{
	    Falcor::Cuda::Profiler::EventData* Profiler::getEvent(const HashedString& name)
		{
			auto event = Falcor::Profiler::isEventRegistered(name);
			if(!event)
			{
	            return createNewCudaEvent(name);
			}
			else
			{
				Falcor::Cuda::Profiler::EventData *cuEv = dynamic_cast<Falcor::Cuda::Profiler::EventData*>(event);
				if (!cuEv)
					throw std::logic_error(((std::string)"The event " + name.str + " is not a Cuda-Event").c_str());
	            return cuEv;
		    }
		}

		Falcor::Cuda::Profiler::EventData* Profiler::createNewCudaEvent(const HashedString& name)
		{
			EventData *data = new EventData;
			Falcor::Profiler::initNewEvent(data, name);
			return data;
		}

		void Profiler::startEvent(const HashedString& name)
		{
			Profiler::EventData *pEvent = Profiler::isEventRegistered(name);
			if (!pEvent)
			{
				pEvent = createNewCudaEvent(name);
			}
			Falcor::Cuda::Profiler::startEvent(name, pEvent);
		}

        void Profiler::endEvent(const HashedString& name)
		{
			Falcor::Cuda::Profiler::endEvent(name, Falcor::Cuda::Profiler::getEvent(name));
		}


        void Profiler::startEvent(const HashedString& name, EventData *pData)
		{
			Falcor::Profiler::startEvent(pData->name, pData);
			cuEventRecord(pData->startEvent, 0);
		}

        void Profiler::endEvent(const HashedString& name, EventData *pData)
		{
			cuEventRecord(pData->stopEvent, 0);
			cuEventSynchronize(pData->stopEvent);
			float ms = -FLT_MAX;
			cuEventElapsedTime(&ms, pData->startEvent, pData->stopEvent);
			Falcor::Profiler::endEvent(name);
			pData->gpuTotal += ms;
		}

		Profiler::EventData* Profiler::isEventRegistered(const HashedString& name)
		{
			return dynamic_cast<Falcor::Cuda::Profiler::EventData*>(Falcor::Profiler::isEventRegistered(name));
		}

		Profiler::EventData::EventData()
		{
			cuEventCreate(&startEvent, CU_EVENT_BLOCKING_SYNC);
			cuEventCreate(&stopEvent, CU_EVENT_BLOCKING_SYNC);
		}
		Profiler::EventData::~EventData()
		{
			cuEventDestroy(startEvent);
			cuEventDestroy(stopEvent);
		}
	}
}
