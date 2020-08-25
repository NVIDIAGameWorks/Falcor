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
#pragma once
#include "CpuTimer.h"
#include "Utils/UI/Gui.h"

namespace Falcor
{
    /** A clock. This class supports both real-time clock (based on the system's clock) and a fixed time-step clock (based on tick count)
    */
    class dlldecl Clock
    {
    public:
        Clock();

        /** Start the system
        */
        static void start();

        /** End the system
        */
        static void shutdown();

        /** Set the current time
            \param[in] seconds The time in seconds
            \param[in] deferToNextTick Apply the change on the next tick. No changes will be made to the clock until the next tick
        */
        Clock& setTime(double seconds, bool deferToNextTick = false);

        deprecate("4.0.1", "Use setTime() instead.")
        Clock& now(double seconds, bool deferToNextTick = false) { return setTime(seconds, deferToNextTick); }

        /** Get the time of the last `tick()` call
        */
        double getTime() const { return mTime.now; }

        deprecate("4.0.1", "Use getTime() instead.")
        double now() const { return getTime(); }

        /** Get the time delta between the 2 previous ticks. This function respects the FPS simulation setting
            Note that due to floating-point precision, this function won't necessarily return exactly (1/FPS) when simulating framerate.
            This function will potentially return a negative number, for example when resetting the time to zero
        */
        double getDelta() const { return mTime.delta; }

        deprecate("4.0.1", "Use getDelta() instead.")
        double delta() const { return getDelta(); }

        /** Set the current frame ID. Calling this will cause the next `tick()` call to be skipped
            When running in real-time mode, it will only change the frame number without affecting the time
            When simulating FPS, it will change the time to match the current frame ID
            \param[in] seconds The frame ID
            \param[in] deferToNextTick Apply the change on the next tick. No changes will be made to the clock until the next tick
        */
        Clock& setFrame(uint64_t f, bool deferToNextTick = false);

        deprecate("4.0.1", "Use setFrame() instead.")
        Clock& frame(uint64_t f, bool deferToNextTick = false) { return setFrame(f, deferToNextTick); }

        /** Get the current frame ID.
            When running in real-time mode, this is the number of frames since the last time the time was set.
            When simulating FPS, the number of frames according to the time
        */
        uint64_t getFrame() const { return mFrames; }

        deprecate("4.0.1", "Use getFrame() instead.")
        uint64_t frame() const { return getFrame(); }

        /** Get the real-time delta between the 2 previous ticks.
            This function returns the actual time that passed between the 2 `tick()` calls. It doesn't any time-manipulation setting like time-scaling and FPS simulation
        */
        double getRealTimeDelta() const { return mRealtime.delta; }

        deprecate("4.0.1", "Use getRealTimeDelta() instead.")
        double realTimeDelta() const { return getRealTimeDelta(); }

        /** Set the time at which to terminate the application.
        */
        Clock& setExitTime(double seconds);

        /** Get the time at which to terminate the application.
        */
        double getExitTime() const { return mExitTime; }

        /** Set the frame at which to terminate the application.
        */
        Clock& setExitFrame(uint64_t frame);

        /** Get the frame at which to terminate the application.
        */
        uint64_t getExitFrame() const { return mExitFrame; }

        /** Check if the application should be terminated.
        */
        bool shouldExit() const;

        /** Tick the clock. Calling this function has no effect if the clock is paused
        */
        Clock& tick();

        /** Set the requested FPS to simulate, or disable FPS simulation.
            When enabling FPS simulation, calls to tick() will change the time by `1/FPS` seconds.
            If FPS simulation is disabled, calling `tick()` will add the actual time that passed since the previous `tick()` call
        */
        Clock& setFramerate(uint32_t fps);

        deprecate("4.0.1", "Use setFramerate() instead.")
        Clock& framerate(uint32_t fps) { return setFramerate(fps); }

        /** Get the requested FPS value
        */
        uint32_t getFramerate() const { return mFramerate; }

        deprecate("4.0.1", "Use getFramerate() instead.")
        uint32_t framerate() const { return getFramerate(); }

        /** Pause the clock
        */
        Clock& pause() { mPaused = true; return *this; }

        /** Resume the clock
        */
        Clock& play();

        /** Stop the clock (pause + reset)
        */
        Clock& stop() { setTime(0); return pause(); }

        /** Step forward or backward. Ignored if the Clock is running or not in FPS simulation mode
            \param[in] frames The number of frames to step. Can be negative
            The function will not step backward beyond frame zero
        */
        Clock& step(int64_t frames = 1);

        /** Set the time scale. This value is ignored when simulating FPS
        */
        Clock& setTimeScale(double scale) { mScale = scale; return *this; }

        deprecate("4.0.1", "Use setTimeScale() instead.")
        Clock& timeScale(double scale) { return setTimeScale(scale); }

        /** Get the scale
        */
        double getTimeScale() const { return mScale; }

        deprecate("4.0.1", "Use getTimeScale() instead.")
        double timeScale() const { return getTimeScale(); }

        /** Check if the clock is paused
        */
        bool isPaused() const { return mPaused; }

        /** Check if the clock is in real-time mode
        */
        bool isSimulatingFps() const { return mFramerate != 0; }

        deprecate("4.0.1", "Use isSimulatingFps() instead.")
        bool simulatingFps() const { return isSimulatingFps(); }

        /** Render the UI
        */
        void renderUI(Gui::Window& w);

        /** Get the script string
        */
        std::string getScript(const std::string& var) const;
    private:
        struct Time
        {
            double now = 0;
            double delta = 0;
            void update(double time)
            {
                delta = time - now;
                now = time;
            }
        } mRealtime, mTime;

        uint32_t mFramerate = 0;
        uint64_t mFrames = 0;
        uint64_t mTicksPerFrame = 0;
        CpuTimer mTimer;

        bool mPaused = false;
        double mScale = 1;
        std::optional<double> mDeferredTime;
        std::optional<uint64_t> mDeferredFrameID;

        double mExitTime = 0.0;
        uint64_t mExitFrame = 0;

        void updateTimer();
        void resetDeferredObjects();
    };
}
