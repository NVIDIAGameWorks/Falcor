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
#include "Clock.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Scripting/ScriptWriter.h"

namespace Falcor
{
    namespace
    {
        constexpr char kTime[] = "time";
        constexpr char kFrame[] = "frame";
        constexpr char kFramerate[] = "framerate";
        constexpr char kTimeScale[] = "timeScale";
        constexpr char kExitTime[] = "exitTime";
        constexpr char kExitFrame[] = "exitFrame";
        constexpr char kStartTime[] = "startTime";
        constexpr char kEndTime[] = "endTime";
        constexpr char kPause[] = "pause";
        constexpr char kPlay[] = "play";
        constexpr char kStop[] = "stop";
        constexpr char kStep[] = "step";

        std::optional<uint32_t> fpsDropdown(Gui::Window& w, uint32_t curVal)
        {
            static const uint32_t commonFps[] = { 0, 24, 25, 30, 48, 50, 60, 75, 90, 120, 144, 200, 240, 360, 480 };
            static const uint32_t kCustom = uint32_t(-1);

            static auto dropdown = []()
            {
                Gui::DropdownList d;
                for (auto f : commonFps) d.push_back({ f, f == 0 ? "Disabled" : std::to_string(f) });
                d.push_back({ kCustom, "Custom" });
                return d;
            }();

            uint32_t index = [curVal]()
            {
                for (auto f : commonFps) if (f == curVal) return f;
                return kCustom;
            }();

            bool changed = w.dropdown("FPS", dropdown, index);
            if (index == kCustom)
            {
                changed = w.var<uint32_t>("Custom##fps", curVal, 0u, std::numeric_limits<uint32_t>::max(), 1u, false, nullptr);
            }
            else curVal = index;

            return changed ? std::optional(curVal) : std::nullopt;
        }

        struct ClockTextures
        {
            Texture::SharedPtr pPlay;
            Texture::SharedPtr pPause;
            Texture::SharedPtr pRewind;
            Texture::SharedPtr pStop;
            Texture::SharedPtr pNextFrame;
            Texture::SharedPtr pPrevFrame;
        } gClockTextures;

        constexpr uint64_t kTicksPerSecond = 14400 * (1 << 16); // 14400 is a common multiple of our supported frame-rates. 2^16 gives 64K intra-frame steps

        double timeFromFrame(uint64_t frame, uint64_t ticksPerFrame)
        {
            return double(frame * ticksPerFrame) / (double)kTicksPerSecond;
        }

        uint64_t frameFromTime(double seconds, uint64_t ticksPerFrame)
        {
            return uint64_t(seconds * (double)kTicksPerSecond) / ticksPerFrame;
        }
    }

    Clock::Clock() { setTime(0); }

    Clock& Clock::setFramerate(uint32_t fps)
    {
        mFramerate = fps;
        mTicksPerFrame = 0;
        if(fps)
        {
            if (kTicksPerSecond % fps) logWarning("Clock::setFramerate() - requested FPS can't be accurately represented. Expect rounding errors");
            mTicksPerFrame = kTicksPerSecond / fps;
        }

        if(!mDeferredFrameID && !mDeferredTime) setTime(mTime.now);
        return *this;
    }

    Clock& Clock::setExitTime(double seconds)
    {
        mExitTime = seconds;
        mExitFrame = 0;
        return *this;
    }

    Clock& Clock::setExitFrame(uint64_t frame)
    {
        mExitFrame = frame;
        mExitTime = 0.0;
        return *this;
    }

    bool Clock::shouldExit() const
    {
        return ((mExitTime && getTime() >= mExitTime) || (mExitFrame && getFrame() >= mExitFrame));
    }

    Clock& Clock::tick()
    {
        if (mDeferredFrameID) setFrame(mDeferredFrameID.value());
        else if (mDeferredTime) setTime(mDeferredTime.value());
        else if(!mPaused) step();
        return *this;
    }

    bool Clock::setStartTime(double time)
    {
        if (time <= 0.)
        {
            mStartTime = 0.;
            return true;
        }
        if (mEndTime < 0. || time < mEndTime)
        {
            mStartTime = time;
            return true;
        }
        return false;
    }

    bool Clock::setEndTime(double time)
    {
        if (time < 0. || mStartTime <= 0.f || time > mStartTime)
        {
            mEndTime = time;
            return true;
        }
        return false;
    }

    void Clock::updateTimer()
    {
        mTimer.update();
        mRealtime.update(mRealtime.now + mTimer.delta());
    }

    void Clock::resetDeferredObjects()
    {
        mDeferredTime = std::nullopt;
        mDeferredFrameID = std::nullopt;
    }

    Clock& Clock::setTime(double seconds, bool deferToNextTick)
    {
        resetDeferredObjects();

        seconds = clampTime(seconds);

        if (deferToNextTick)
        {
            mDeferredTime = seconds;
        }
        else
        {
            updateTimer();
            if (mFramerate)
            {
                mFrames = frameFromTime(seconds, mTicksPerFrame);
                seconds = timeFromFrame(mFrames, mTicksPerFrame);
            }
            else mFrames = 0;
            if (mTime.delta < 0) mTime.delta = 0;

            mTime.delta = mTime.now - seconds;
            mTime.now = seconds;
        }
        return *this;
    }

    Clock& Clock::setFrame(uint64_t f, bool deferToNextTick)
    {
        resetDeferredObjects();

        if (deferToNextTick)
        {
            mDeferredFrameID = f;
        }
        else
        {
            updateTimer();
            mFrames = f;
            if (mFramerate)
            {
                double orgSecs = timeFromFrame(mFrames, mTicksPerFrame);
                // TODO: The clamping really should be on ticks, as should everything else
                // except when we actually ask for the actual floating time (e.g., for interpolation).
                // Otherwise the rounding will be a terrible mess.
                double newSecs = clampTime(orgSecs);
                if (newSecs != orgSecs)
                    mFrames = frameFromTime(newSecs, mTicksPerFrame);

                mTime.delta = mTime.now - newSecs;
                if (mTime.delta < 0) mTime.delta = 0;
                mTime.now = newSecs;
            }
        }
        return *this;
    }

    Clock& Clock::play()
    {
        updateTimer();
        mPaused = false;
        return *this;
    }

    Clock& Clock::step(int64_t frames)
    {
        if (frames < 0 && uint64_t(-frames) > mFrames) mFrames = 0;
        else mFrames += frames;

        updateTimer();
        double t = isSimulatingFps() ? timeFromFrame(mFrames, mTicksPerFrame) : ((mTimer.delta() * mScale) + mTime.now);
        t = clampTime(t);
        mTime.update(t);
        return *this;
    }

    void Clock::renderUI(Gui::Window& w)
    {
        const auto& tex = gClockTextures;

        float time = (float)getTime();
        float scale = (float)getTimeScale();
        if (w.var("Time##Cur", time, 0.f, FLT_MAX, 0.001f, false, "%.3f")) setTime(time);
        if (!isSimulatingFps() && w.var("Scale", scale)) setTimeScale(scale);
        bool showStep = mPaused && isSimulatingFps();

        float indent = showStep ? 10.0f : 60.0f;
        w.indent(indent);
        static const uint2 iconSize = { 25, 25 };
        if (w.imageButton("Rewind", tex.pRewind, iconSize)) setTime(0);
        if (showStep && w.imageButton("PrevFrame", tex.pPrevFrame, iconSize, true, true)) step(-1);
        if (w.imageButton("Stop", tex.pStop, iconSize, true, true)) stop();
        auto pTex = mPaused ? tex.pPlay : tex.pPause;
        if (w.imageButton("PlayPause", pTex, iconSize, true, true)) mPaused ? play() : pause();
        if (showStep && w.imageButton("NextFrame", tex.pNextFrame, iconSize, true, true)) step();

        w.indent(-indent);

        w.separator(2);
        w.text("Framerate Simulation");
        w.tooltip("Simulate a constant frame rate. The time will advance by 1/FPS each frame, regardless of the actual frame rendering time");

        auto fps = fpsDropdown(w, mFramerate);
        if (fps) setFramerate(fps.value());

        if (isSimulatingFps())
        {
            uint64_t curFrame = getFrame();
            if (w.var("Frame ID", curFrame)) setFrame(curFrame);
        }
    }

    FALCOR_SCRIPT_BINDING(Clock)
    {
        using namespace pybind11::literals;

        pybind11::class_<Clock> clock(m, "Clock");

        auto setTime = [](Clock* pClock, double t) {pClock->setTime(t, true); };
        clock.def_property(kTime, &Clock::getTime, setTime);
        auto setFrame = [](Clock* pClock, uint64_t f) {pClock->setFrame(f, true); };
        clock.def_property(kFrame, &Clock::getFrame, setFrame);
        clock.def_property(kFramerate, &Clock::getFramerate, &Clock::setFramerate);
        clock.def_property(kTimeScale, &Clock::getTimeScale, &Clock::setTimeScale);
        clock.def_property(kStartTime, &Clock::getStartTime, &Clock::setStartTime);
        clock.def_property(kEndTime, &Clock::getEndTime, &Clock::setEndTime);
        clock.def_property(kExitTime, &Clock::getExitTime, &Clock::setExitTime);
        clock.def_property(kExitFrame, &Clock::getExitFrame, &Clock::setExitFrame);

        clock.def(kPause, &Clock::pause);
        clock.def(kPlay, &Clock::play);
        clock.def(kStop, &Clock::stop);
        clock.def(kStep, &Clock::step, "frames"_a = 1);
    }

    void Clock::start()
    {
        auto loadTexture = [](const std::string& tex)
        {
            auto pTex = Texture::createFromFile("Framework/Textures/" + tex, false, true);
            if (!pTex) throw RuntimeError("Failed to load texture");
            return pTex;
        };
        gClockTextures.pRewind = loadTexture("Rewind.jpg");
        gClockTextures.pPlay = loadTexture("Play.jpg");
        gClockTextures.pPause = loadTexture("Pause.jpg");
        gClockTextures.pStop = loadTexture("Stop.jpg");
        gClockTextures.pNextFrame = loadTexture("NextFrame.jpg");
        gClockTextures.pPrevFrame = loadTexture("PrevFrame.jpg");
    }

    void Clock::shutdown()
    {
        gClockTextures = {};
    }

    std::string Clock::getScript(const std::string& var) const
    {
        std::string s;
        s += ScriptWriter::makeSetProperty(var, kTime, 0);
        s += ScriptWriter::makeSetProperty(var, kFramerate, mFramerate);
        if (mExitTime) s += ScriptWriter::makeSetProperty(var, kExitTime, mExitTime);
        if (mExitFrame) s += ScriptWriter::makeSetProperty(var, kExitFrame, mExitFrame);
        s += std::string("# If ") + kFramerate + " is not zero, you can use the frame property to set the start frame\n";
        s += "# " + ScriptWriter::makeSetProperty(var, kFrame, 0);
        if (mPaused) s += ScriptWriter::makeMemberFunc(var, kPause);
        return s;
    }
}
