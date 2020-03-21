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
 **************************************************************************/
#include "stdafx.h"
#include "Clock.h"

namespace Falcor
{
    namespace
    {
        constexpr char kNow[] = "now";
        constexpr char kPause[] = "pause";
        constexpr char kPlay[] = "play";
        constexpr char kStop[] = "stop";
        constexpr char kSimFps[] = "fpsSim";
        constexpr char kFrame[] = "frame";
        constexpr char kStep[] = "step";
        constexpr char kFramerate[] = "framerate";

        std::optional<uint32_t> fpsDropdown(Gui::Window& w, uint32_t curVal)
        {
            static const uint32_t commonFps[] = { 0, 24, 25, 30, 48, 50, 60, 75, 90, 120, 144, 200, 240, 360, 480 };
            static const uint32_t kCustom = uint32_t(-1);

            static auto dropdown = []()
            {
                Gui::DropdownList d;
                for (auto f : commonFps) d.push_back({ f, f == 0 ? "Disabled" : to_string(f) });
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
                changed = w.var<uint32_t>("Custom##fps", curVal, 0u, UINT32_MAX, 1u, false, nullptr);
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

    Clock::Clock() { now(0); }

    Clock& Clock::framerate(uint32_t fps)
    {
        mFramerate = fps;
        mTicksPerFrame = 0;
        if(fps)
        {
            if (kTicksPerSecond % fps) logWarning("Clock::framerate() - requesetd FPS can't be accurately representated. Expect roudning errors");
            mTicksPerFrame = kTicksPerSecond / fps;
        }

        if(!mDeferredFrameID && !mDeferredTime) now(mTime.now);
        return *this;
    }

    Clock& Clock::tick()
    {
        if (mDeferredFrameID) frame(mDeferredFrameID.value());
        else if (mDeferredTime) now(mDeferredTime.value());
        else if(!mPaused) step();
        return *this;
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

    Clock& Clock::now(double seconds, bool deferToNextTick)
    {
        resetDeferredObjects();

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

            mTime.delta = mTime.now - seconds;
            mTime.now = seconds;
        }
        return *this;
    }

    Clock& Clock::frame(uint64_t f, bool deferToNextTick)
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
                double secs = timeFromFrame(mFrames, mTicksPerFrame);
                mTime.delta = mTime.now - secs;
                mTime.now = secs;
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
        double t = simulatingFps() ? timeFromFrame(mFrames, mTicksPerFrame) : ((mTimer.delta() * mScale) + mTime.now);
        mTime.update(t);
        return *this;
    }

    void Clock::renderUI(Gui::Window& w)
    {
        const auto& tex = gClockTextures;

        float time = (float)now();
        float scale = (float)timeScale();
        if (w.var("Time##Cur", time, 0.f, FLT_MAX, 0.001f, false, "%.3f")) now(time);
        if (!simulatingFps() && w.var("Scale", scale)) timeScale(scale);
        bool showStep = mPaused && simulatingFps();

        float indent = showStep ? 10.0f : 60.0f;
        w.indent(indent);
        static const uvec2 iconSize = { 25, 25 };
        if (w.imageButton("Rewind", tex.pRewind, iconSize)) now(0);
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
        if (fps) framerate(fps.value());

        if (simulatingFps())
        {
            uint64_t curFrame = frame();
            if (w.var("Frame ID", curFrame)) frame(curFrame);
        }
    }

    SCRIPT_BINDING(Clock)
    {
        auto c = m.regClass(Clock);
        c.func_(kNow, ScriptBindings::overload_cast<>(&Clock::now, ScriptBindings::const_));

        auto now = [](Clock* pClock, double secs) {pClock->now(secs, true); };
        c.func_(kNow, now, "seconds"_a);

        c.func_(kFrame, ScriptBindings::overload_cast<>(&Clock::frame, ScriptBindings::const_));
        auto frame = [](Clock* pClock, uint64_t f) {pClock->frame(f, true); };
        c.func_(kFrame, frame, "frameID"_a);

        c.func_(kPause, &Clock::pause);
        c.func_(kPlay, &Clock::play);
        c.func_(kStop, &Clock::stop);
        c.func_(kPause, &Clock::pause);
        c.func_(kStep, &Clock::step, "frames"_a = 1);
        c.func_(kFramerate, ScriptBindings::overload_cast<uint32_t>(&Clock::framerate));
        c.func_(kFramerate, ScriptBindings::overload_cast<>(&Clock::framerate, ScriptBindings::const_));
    }

    void Clock::start()
    {
        auto loadTexture = [](const std::string& tex)
        {
            auto pTex = Texture::createFromFile("Framework/Textures/" + tex, false, true);
            if (!pTex) throw std::exception("Failed to load texture");
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
        s += Scripting::makeMemberFunc(var, kNow, 0);
        s += Scripting::makeMemberFunc(var, kFramerate, mFramerate);
        s += std::string("# If ") + kFramerate + "() is not zero, you can use the following function to set the start frame\n";
        s += "# " + Scripting::makeMemberFunc(var, kFrame, 0);
        if (mPaused) s += Scripting::makeMemberFunc(var, kPause);
        return s;
    }
}
