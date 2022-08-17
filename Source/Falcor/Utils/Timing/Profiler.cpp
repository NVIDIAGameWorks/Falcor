/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Profiler.h"
#include "Core/Renderer.h"
#include "Core/API/Device.h"
#include "Core/API/GpuTimer.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <fstream>

#ifdef FALCOR_D3D12
#include <WinPixEventRuntime/pix3.h>
#endif

namespace Falcor
{
    namespace
    {
        // With sigma = 0.98, then after 100 frames, a given value's contribution is down to ~1.7% of
        // the running average, which seems to provide a reasonable trade-off of temporal smoothing
        // versus setting in to a new value when something has changed.
        const float kSigma = 0.98f;

        // Size of the event history. The event history is keeping track of event times to allow
        // for computing statistics (min, max, mean, stddev) over the recent history.
        const size_t kMaxHistorySize = 512;
    }

    // Profiler::Stats

    pybind11::dict Profiler::Stats::toPython() const
    {
        pybind11::dict d;

        d["min"] = min;
        d["max"] = max;
        d["mean"] = mean;
        d["stdDev"] = stdDev;

        return d;
    }

    Profiler::Stats Profiler::Stats::compute(const float* data, size_t len)
    {
        if (len == 0) return {};

        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();
        double sum = 0.0;
        double sum2 = 0.0;

        for (size_t i = 0; i < len; ++i)
        {
            float value = data[i];
            min = std::min(min, value);
            max = std::max(max, value);
            sum += value;
            sum2 += value * value;
        }

        double mean = sum / len;
        double mean2 = sum2 / len;
        double variance = mean2 - mean * mean;
        double stdDev = std::sqrt(variance);

        return { min, max, (float)mean, (float)stdDev };
    }

    // Profiler::Event

    Profiler::Event::Event(const std::string& name)
        : mName(name)
        , mCpuTimeHistory(kMaxHistorySize, 0.f)
        , mGpuTimeHistory(kMaxHistorySize, 0.f)
    {}

    Profiler::Stats Profiler::Event::computeCpuTimeStats() const
    {
        return Stats::compute(mCpuTimeHistory.data(), mHistorySize);
    }

    Profiler::Stats Profiler::Event::computeGpuTimeStats() const
    {
        return Stats::compute(mGpuTimeHistory.data(), mHistorySize);
    }

    void Profiler::Event::start(uint32_t frameIndex)
    {
        if (++mTriggered > 1)
        {
            logWarning("Profiler event '{}' was triggered while it is already running. Nesting profiler events with the same name is disallowed and you should probably fix that. Ignoring the new call.", mName);
            return;
        }

        auto& frameData = mFrameData[frameIndex % 2];

        // Update CPU time.
        frameData.cpuStartTime = CpuTimer::getCurrentTimePoint();

        // Update GPU time.
        FALCOR_ASSERT(frameData.pActiveTimer == nullptr);
        FALCOR_ASSERT(frameData.currentTimer <= frameData.pTimers.size());
        if (frameData.currentTimer == frameData.pTimers.size())
        {
            frameData.pTimers.push_back(GpuTimer::create());
        }
        frameData.pActiveTimer = frameData.pTimers[frameData.currentTimer++].get();
        frameData.pActiveTimer->begin();
        frameData.valid = false;
    }

    void Profiler::Event::end(uint32_t frameIndex)
    {
        if (--mTriggered != 0) return;

        auto& frameData = mFrameData[frameIndex % 2];

        // Update CPU time.
        frameData.cpuTotalTime += (float)CpuTimer::calcDuration(frameData.cpuStartTime, CpuTimer::getCurrentTimePoint());

        // Update GPU time.
        FALCOR_ASSERT(frameData.pActiveTimer != nullptr);
        frameData.pActiveTimer->end();
        frameData.pActiveTimer = nullptr;
        frameData.valid = true;
    }

    void Profiler::Event::endFrame(uint32_t frameIndex)
    {
        // Resolve GPU timers for the current frame measurements.
        // This is necessary before we readback of results next frame.
        {
            auto& frameData = mFrameData[frameIndex % 2];
            for (auto& pTimer : frameData.pTimers)
            {
                pTimer->resolve();
            }
        }

        // Update CPU/GPU time from last frame measurement.
        auto& frameData = mFrameData[(frameIndex + 1) % 2];

        // Skip update if there are no measurements last frame.
        if (!frameData.valid) return;

        mCpuTime = frameData.cpuTotalTime;
        mGpuTime = 0.f;
        for (size_t i = 0; i < frameData.currentTimer; ++i) mGpuTime += (float)frameData.pTimers[i]->getElapsedTime();
        frameData.cpuTotalTime = 0.f;
        frameData.currentTimer = 0;

        // Update EMA.
        mCpuTimeAverage = mCpuTimeAverage < 0.f ? mCpuTime : (kSigma * mCpuTimeAverage + (1.f - kSigma) * mCpuTime);
        mGpuTimeAverage = mGpuTimeAverage < 0.f ? mGpuTime : (kSigma * mGpuTimeAverage + (1.f - kSigma) * mGpuTime);

        // Update history.
        mCpuTimeHistory[mHistoryWriteIndex] = mCpuTime;
        mGpuTimeHistory[mHistoryWriteIndex] = mGpuTime;
        mHistoryWriteIndex = (mHistoryWriteIndex + 1) % kMaxHistorySize;
        mHistorySize = std::min(mHistorySize + 1, kMaxHistorySize);

        mTriggered = 0;
    }

    // Profiler::Capture

    pybind11::dict Profiler::Capture::toPython() const
    {
        pybind11::dict pyCapture;
        pybind11::dict pyEvents;

        pyCapture["frameCount"] = mFrameCount;
        pyCapture["events"] = pyEvents;

        for (const auto& lane : mLanes)
        {
            pybind11::dict pyLane;
            pyLane["name"] = lane.name;
            pyLane["stats"] = lane.stats.toPython();
            pyLane["records"] = lane.records;
            pyEvents[lane.name.c_str()] = pyLane;
        }

        return pyCapture;
    }

    std::string Profiler::Capture::toJsonString() const
    {
        using namespace pybind11::literals;

        // We use pythons JSON encoder to encode the python dictionary to a JSON string.
        pybind11::module json = pybind11::module::import("json");
        pybind11::object dumps = json.attr("dumps");
        return pybind11::cast<std::string>(dumps(toPython(), "indent"_a = 2));
    }

    void Profiler::Capture::writeToFile(const std::filesystem::path& path) const
    {
        auto json = toJsonString();
        std::ofstream ofs(path);
        ofs.write(json.data(), json.size());
    }

    Profiler::Capture::Capture(size_t reservedEvents, size_t reservedFrames)
        : mReservedFrames(reservedFrames)
    {
        // Speculativly allocate event record storage.
        mLanes.resize(reservedEvents * 2);
        for (auto& lane : mLanes) lane.records.reserve(reservedFrames);
    }

    Profiler::Capture::SharedPtr Profiler::Capture::create(size_t reservedEvents, size_t reservedFrames)
    {
        return SharedPtr(new Capture(reservedEvents, reservedFrames));
    }

    void Profiler::Capture::captureEvents(const std::vector<Event*>& events)
    {
        if (events.empty()) return;

        // Initialize on first capture.
        if (mEvents.empty())
        {
            mEvents = events;
            mLanes.resize(mEvents.size() * 2);
            for (size_t i = 0; i < mEvents.size(); ++i)
            {
                auto& pEvent = mEvents[i];
                mLanes[i * 2].name = pEvent->getName() + "/cpuTime";
                mLanes[i * 2].records.reserve(mReservedFrames);
                mLanes[i * 2 + 1].name = pEvent->getName() + "/gpuTime";
                mLanes[i * 2 + 1].records.reserve(mReservedFrames);
            }
            return; // Exit as no data is available on first capture.
        }

        // Record CPU/GPU timing on subsequent captures.
        for (size_t i = 0; i < mEvents.size(); ++i)
        {
            auto& pEvent = mEvents[i];
            mLanes[i * 2].records.push_back(pEvent->getCpuTime());
            mLanes[i * 2 + 1].records.push_back(pEvent->getGpuTime());
        }

        ++mFrameCount;
    }

    void Profiler::Capture::finalize()
    {
        FALCOR_ASSERT(!mFinalized);

        for (auto& lane : mLanes)
        {
            lane.stats = Stats::compute(lane.records.data(), lane.records.size());
        }

        mFinalized = true;
    }

    // Profiler

    void Profiler::startEvent(const std::string& name, Flags flags)
    {
        if (mEnabled && is_set(flags, Flags::Internal))
        {
            // '/' is used as a "path delimiter", so it cannot be used in the event name.
            if (name.find('/') != std::string::npos)
            {
                logWarning("Profiler event names must not contain '/'. Ignoring this profiler event.");
                return;
            }

            mCurrentEventName = mCurrentEventName + "/" + name;

            Event* pEvent = getEvent(mCurrentEventName);
            FALCOR_ASSERT(pEvent != nullptr);
            if (!mPaused) pEvent->start(mFrameIndex);

            if (std::find(mCurrentFrameEvents.begin(), mCurrentFrameEvents.end(), pEvent) == mCurrentFrameEvents.end())
            {
                mCurrentFrameEvents.push_back(pEvent);
            }
        }
        if (is_set(flags, Flags::Pix))
        {
#ifdef FALCOR_D3D12
            PIXBeginEvent((ID3D12GraphicsCommandList*)gpDevice->getRenderContext()->getLowLevelData()->getD3D12CommandList(), PIX_COLOR(0, 0, 0), name.c_str());
#else
            gpDevice->getRenderContext()->getLowLevelData()->beginDebugEvent(name.c_str());
#endif
        }
    }

    void Profiler::endEvent(const std::string& name, Flags flags)
    {
        if (mEnabled && is_set(flags, Flags::Internal))
        {
            // '/' is used as a "path delimiter", so it cannot be used in the event name.
            if (name.find('/') != std::string::npos) return;

            Event* pEvent = getEvent(mCurrentEventName);
            FALCOR_ASSERT(pEvent != nullptr);
            if (!mPaused) pEvent->end(mFrameIndex);

            mCurrentEventName.erase(mCurrentEventName.find_last_of("/"));
        }

        if (is_set(flags, Flags::Pix))
        {
#ifdef FALCOR_D3D12
            PIXEndEvent((ID3D12GraphicsCommandList*)gpDevice->getRenderContext()->getLowLevelData()->getD3D12CommandList());
#else
            gpDevice->getRenderContext()->getLowLevelData()->endDebugEvent();
#endif
        }
    }

    Profiler::Event* Profiler::getEvent(const std::string& name)
    {
        auto event = findEvent(name);
        return event ? event : createEvent(name);
    }

    void Profiler::endFrame()
    {
        if (mPaused) return;

        // Wait for GPU timings to be available from last frame.
        // We use a single fence here instead of one per event, which gets too inefficient.
        // TODO: This code should refactored to batch the resolve and readback of timestamps.
        if (mFenceValue != uint64_t(-1)) mpFence->syncCpu();

        for (Event* pEvent : mCurrentFrameEvents)
        {
            pEvent->endFrame(mFrameIndex);
        }

        // Flush and insert signal for synchronization of GPU timings.
        auto pRenderContext = gpFramework->getRenderContext();
        pRenderContext->flush(false);
        mFenceValue = mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());

        if (mpCapture) mpCapture->captureEvents(mCurrentFrameEvents);

        mLastFrameEvents = std::move(mCurrentFrameEvents);
        ++mFrameIndex;
    }

    void Profiler::startCapture(size_t reservedFrames)
    {
        setEnabled(true);
        mpCapture = Capture::create(mLastFrameEvents.size(), reservedFrames);
    }

    Profiler::Capture::SharedPtr Profiler::endCapture()
    {
        Capture::SharedPtr pCapture;
        std::swap(pCapture, mpCapture);
        if (pCapture) pCapture->finalize();
        return pCapture;
    }

    bool Profiler::isCapturing() const
    {
        return mpCapture != nullptr;
    }

    pybind11::dict Profiler::getPythonEvents() const
    {
        pybind11::dict result;

        auto addLane = [&result] (std::string name, float value, float average, const Stats& stats)
        {
            pybind11::dict d;
            d["name"] = name;
            d["value"] = value;
            d["average"] = average;
            d["stats"] = stats.toPython();
            result[name.c_str()] = d;
        };

        for (const Profiler::Event* pEvent : getEvents())
        {
            addLane(pEvent->getName() + "/cpuTime", pEvent->getCpuTime(), pEvent->getCpuTimeAverage(), pEvent->computeCpuTimeStats());
            addLane(pEvent->getName() + "/gpuTime", pEvent->getGpuTime(), pEvent->getGpuTimeAverage(), pEvent->computeGpuTimeStats());
        }

        return result;
    }

    const Profiler::SharedPtr& Profiler::instancePtr()
    {
        static Profiler::SharedPtr pInstance;
        if (!pInstance) pInstance = std::make_shared<Profiler>();
        return pInstance;
    }

    Profiler::Profiler()
    {
        mpFence = GpuFence::create();
    }

    Profiler::Event* Profiler::createEvent(const std::string& name)
    {
        auto pEvent = std::shared_ptr<Event>(new Event(name));
        mEvents.emplace(name, pEvent);
        return pEvent.get();
    }

    Profiler::Event* Profiler::findEvent(const std::string& name)
    {
        auto event = mEvents.find(name);
        return (event == mEvents.end()) ? nullptr : event->second.get();
    }

    FALCOR_SCRIPT_BINDING(Profiler)
    {
        using namespace pybind11::literals;

        auto endCapture = [] (Profiler* pProfiler) {
            std::optional<pybind11::dict> result;
            auto pCapture = pProfiler->endCapture();
            if (pCapture) result = pCapture->toPython();
            return result;
        };

        pybind11::class_<Profiler, Profiler::SharedPtr> profiler(m, "Profiler");
        profiler.def_property("enabled", &Profiler::isEnabled, &Profiler::setEnabled);
        profiler.def_property("paused", &Profiler::isPaused, &Profiler::setPaused);
        profiler.def_property_readonly("isCapturing", &Profiler::isCapturing);
        profiler.def_property_readonly("events", &Profiler::getPythonEvents);
        profiler.def("startCapture", &Profiler::startCapture, "reservedFrames"_a = 1000);
        profiler.def("endCapture", endCapture);
    }
}
