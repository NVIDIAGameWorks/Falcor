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
#include "Profiler.h"
#include "Core/API/Device.h"
#include "Core/API/GpuTimer.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"

#include <fstream>

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

pybind11::dict toPython(const Profiler::Stats& stats)
{
    pybind11::dict d;
    d["min"] = stats.min;
    d["max"] = stats.max;
    d["mean"] = stats.mean;
    d["std_dev"] = stats.stdDev;
    return d;
}

pybind11::dict toPython(const Profiler::Capture& capture)
{
    pybind11::dict pyCapture;
    pybind11::dict pyEvents;

    pyCapture["frame_count"] = capture.getFrameCount();
    pyCapture["events"] = pyEvents;

    for (const auto& lane : capture.getLanes())
    {
        pybind11::dict pyLane;
        pyLane["name"] = lane.name;
        pyLane["stats"] = toPython(lane.stats);
        pyLane["records"] = lane.records;
        pyEvents[lane.name.c_str()] = pyLane;
    }

    return pyCapture;
}

pybind11::dict toPython(const std::vector<Profiler::Event*>& events)
{
    pybind11::dict result;

    auto addLane = [&result](std::string name, float value, float average, const Profiler::Stats& stats)
    {
        pybind11::dict d;
        d["name"] = name;
        d["value"] = value;
        d["average"] = average;
        d["stats"] = toPython(stats);
        result[name.c_str()] = d;
    };

    for (const Profiler::Event* pEvent : events)
    {
        addLane(pEvent->getName() + "/cpu_time", pEvent->getCpuTime(), pEvent->getCpuTimeAverage(), pEvent->computeCpuTimeStats());
        addLane(pEvent->getName() + "/gpu_time", pEvent->getGpuTime(), pEvent->getGpuTimeAverage(), pEvent->computeGpuTimeStats());
    }

    return result;
}
} // namespace

// Profiler::Stats

Profiler::Stats Profiler::Stats::compute(const float* data, size_t len)
{
    if (len == 0)
        return {};

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

    return {min, max, (float)mean, (float)stdDev};
}

// Profiler::Event

Profiler::Event::Event(const std::string& name) : mName(name), mCpuTimeHistory(kMaxHistorySize, 0.f), mGpuTimeHistory(kMaxHistorySize, 0.f)
{}

Profiler::Stats Profiler::Event::computeCpuTimeStats() const
{
    return Stats::compute(mCpuTimeHistory.data(), mHistorySize);
}

Profiler::Stats Profiler::Event::computeGpuTimeStats() const
{
    return Stats::compute(mGpuTimeHistory.data(), mHistorySize);
}

void Profiler::Event::start(Profiler& profiler, uint32_t frameIndex)
{
    if (++mTriggered > 1)
    {
        logWarning(
            "Profiler event '{}' was triggered while it is already running. Nesting profiler events with the same name is disallowed and "
            "you should probably fix that. Ignoring the new call.",
            mName
        );
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
        ref<GpuTimer> timer = GpuTimer::create(profiler.mpDevice);
        timer->breakStrongReferenceToDevice();
        frameData.pTimers.push_back(timer);
    }
    frameData.pActiveTimer = frameData.pTimers[frameData.currentTimer++].get();
    frameData.pActiveTimer->begin();
    frameData.valid = false;
}

void Profiler::Event::end(uint32_t frameIndex)
{
    if (--mTriggered != 0)
        return;

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
    if (!frameData.valid)
        return;

    mCpuTime = frameData.cpuTotalTime;
    mGpuTime = 0.f;
    for (size_t i = 0; i < frameData.currentTimer; ++i)
        mGpuTime += (float)frameData.pTimers[i]->getElapsedTime();
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

std::string Profiler::Capture::toJsonString() const
{
    using namespace pybind11::literals;

    // We use pythons JSON encoder to encode the python dictionary to a JSON string.
    pybind11::module json = pybind11::module::import("json");
    pybind11::object dumps = json.attr("dumps");
    return pybind11::cast<std::string>(dumps(toPython(*this), "indent"_a = 2));
}

void Profiler::Capture::writeToFile(const std::filesystem::path& path) const
{
    auto json = toJsonString();
    std::ofstream ofs(path);
    ofs.write(json.data(), json.size());
}

Profiler::Capture::Capture(size_t reservedEvents, size_t reservedFrames) : mReservedFrames(reservedFrames)
{
    // Speculativly allocate event record storage.
    mLanes.resize(reservedEvents * 2);
    for (auto& lane : mLanes)
        lane.records.reserve(reservedFrames);
}

void Profiler::Capture::captureEvents(const std::vector<Event*>& events)
{
    if (events.empty())
        return;

    // Initialize on first capture.
    if (mEvents.empty())
    {
        mEvents = events;
        mLanes.resize(mEvents.size() * 2);
        for (size_t i = 0; i < mEvents.size(); ++i)
        {
            auto& pEvent = mEvents[i];
            mLanes[i * 2].name = pEvent->getName() + "/cpu_time";
            mLanes[i * 2].records.reserve(mReservedFrames);
            mLanes[i * 2 + 1].name = pEvent->getName() + "/gpu_time";
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

Profiler::Profiler(ref<Device> pDevice) : mpDevice(pDevice)
{
    mpFence = mpDevice->createFence();
    mpFence->breakStrongReferenceToDevice();
}

void Profiler::startEvent(RenderContext* pRenderContext, const std::string& name, Flags flags)
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
        if (!mPaused)
            pEvent->start(*this, mFrameIndex);

        if (std::find(mCurrentFrameEvents.begin(), mCurrentFrameEvents.end(), pEvent) == mCurrentFrameEvents.end())
        {
            mCurrentFrameEvents.push_back(pEvent);
        }
    }
    if (is_set(flags, Flags::Pix))
    {
        FALCOR_ASSERT(pRenderContext);
        pRenderContext->getLowLevelData()->beginDebugEvent(name.c_str());
    }
}

void Profiler::endEvent(RenderContext* pRenderContext, const std::string& name, Flags flags)
{
    if (mEnabled && is_set(flags, Flags::Internal))
    {
        // '/' is used as a "path delimiter", so it cannot be used in the event name.
        if (name.find('/') != std::string::npos)
            return;

        Event* pEvent = getEvent(mCurrentEventName);
        FALCOR_ASSERT(pEvent != nullptr);
        if (!mPaused)
            pEvent->end(mFrameIndex);

        mCurrentEventName.erase(mCurrentEventName.find_last_of("/"));
    }

    if (is_set(flags, Flags::Pix))
    {
        FALCOR_ASSERT(pRenderContext)
        pRenderContext->getLowLevelData()->endDebugEvent();
    }
}

Profiler::Event* Profiler::getEvent(const std::string& name)
{
    auto event = findEvent(name);
    return event ? event : createEvent(name);
}

void Profiler::endFrame(RenderContext* pRenderContext)
{
    if (mPaused)
        return;

    // Wait for GPU timings to be available from last frame.
    // We use a single fence here instead of one per event, which gets too inefficient.
    // TODO: This code should refactored to batch the resolve and readback of timestamps.
    if (mFenceValue != uint64_t(-1))
        mpFence->wait();

    for (Event* pEvent : mCurrentFrameEvents)
    {
        pEvent->endFrame(mFrameIndex);
    }

    // Flush and insert signal for synchronization of GPU timings.
    pRenderContext->submit(false);
    mFenceValue = pRenderContext->signal(mpFence.get());

    if (mpCapture)
        mpCapture->captureEvents(mCurrentFrameEvents);

    mLastFrameEvents = std::move(mCurrentFrameEvents);
    ++mFrameIndex;
}

void Profiler::startCapture(size_t reservedFrames)
{
    setEnabled(true);
    mpCapture = std::make_shared<Capture>(mLastFrameEvents.size(), reservedFrames);
}

std::shared_ptr<Profiler::Capture> Profiler::endCapture()
{
    std::shared_ptr<Capture> pCapture;
    std::swap(pCapture, mpCapture);
    if (pCapture)
        pCapture->finalize();
    return pCapture;
}

bool Profiler::isCapturing() const
{
    return mpCapture != nullptr;
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

void Profiler::breakStrongReferenceToDevice()
{
    mpDevice.breakStrongReference();
}

ScopedProfilerEvent::ScopedProfilerEvent(RenderContext* pRenderContext, const std::string& name, Profiler::Flags flags)
    : mpRenderContext(pRenderContext), mName(name), mFlags(flags)
{
    FALCOR_ASSERT(mpRenderContext);
    mpRenderContext->getProfiler()->startEvent(mpRenderContext, mName, mFlags);
}

ScopedProfilerEvent::~ScopedProfilerEvent()
{
    mpRenderContext->getProfiler()->endEvent(mpRenderContext, mName, mFlags);
}

/// Implements a Python context manager for profiling events.
class PythonProfilerEvent
{
public:
    PythonProfilerEvent(RenderContext* pRenderContext, std::string_view name) : mpRenderContext(pRenderContext), mName(name) {}
    void enter() { mpRenderContext->getProfiler()->startEvent(mpRenderContext, mName); }
    void exit(pybind11::object, pybind11::object, pybind11::object) { mpRenderContext->getProfiler()->endEvent(mpRenderContext, mName); }

private:
    RenderContext* mpRenderContext;
    std::string mName;
};

FALCOR_SCRIPT_BINDING(Profiler)
{
    FALCOR_SCRIPT_BINDING_DEPENDENCY(RenderContext)

    using namespace pybind11::literals;

    auto endCapture = [](Profiler* pProfiler)
    {
        std::optional<pybind11::dict> result;
        auto pCapture = pProfiler->endCapture();
        if (pCapture)
            result = toPython(*pCapture);
        return result;
    };

    pybind11::class_<Profiler> profiler(m, "Profiler");
    profiler.def_property("enabled", &Profiler::isEnabled, &Profiler::setEnabled);
    profiler.def_property("paused", &Profiler::isPaused, &Profiler::setPaused);
    profiler.def_property_readonly("is_capturing", &Profiler::isCapturing);
    profiler.def_property_readonly("events", [](const Profiler& profiler) { return toPython(profiler.getEvents()); });
    profiler.def("start_capture", &Profiler::startCapture, "reserved_frames"_a = 1000);
    profiler.def("end_capture", endCapture);

    pybind11::class_<PythonProfilerEvent>(m, "ProfilerEvent")
        .def(pybind11::init<RenderContext*, std::string_view>())
        .def("__enter__", &PythonProfilerEvent::enter)
        .def("__exit__", &PythonProfilerEvent::exit);

    profiler.def(
        "event", [](Profiler& self, std::string_view name) { return PythonProfilerEvent(self.getDevice()->getRenderContext(), name); }
    );
}
} // namespace Falcor
