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
#pragma once
#include "Clock.h"
#include "Core/Macros.h"
#include <algorithm>
#include <string>
#include <vector>

namespace Falcor
{
    /** Framerate calculator
    */
    class FALCOR_API FrameRate
    {
    public:
        FrameRate()
        {
            mFrameTimes.resize(kFrameWindow);
            reset();
        }

        /** Resets the FPS
            After this call it will appear as if the application had just started. Useful in cases a new scene is loaded, since it will display a more accurate FPS.
        */
        void reset()
        {
            mFrameCount = 0;
            mClock.setTime(0).tick();
        }

        /** Tick the timer.
            It is assumed that this is called once per frame, since this frequency is assumed when calculating FPS.
        */
        void newFrame()
        {
            mFrameCount++;
            mFrameTimes[mFrameCount % kFrameWindow] = mClock.tick().getRealTimeDelta();
            mClock.setTime(0).tick();
        }

        /** Get the time in ms it took to render a frame
        */
        double getAverageFrameTime() const
        {
            uint64_t frames = std::min(mFrameCount, kFrameWindow);
            double elapsedTime = 0;
            for(uint64_t i = 0; i < frames; i++) elapsedTime += mFrameTimes[i];
            double time = elapsedTime / double(frames) * 1000;
            return time;
        }

        /** Get the time that it took to render the last frame
        */
        double getLastFrameTime() const
        {
            return mFrameTimes[mFrameCount % kFrameWindow];
        }

        /** Get the frame count (= number of times newFrame() has been called).
        */
        uint64_t getFrameCount() const { return mFrameCount; }

        /** Get a message with the FPS
        */
        std::string getMsg(bool vsyncOn = false) const;

    private:
        Clock mClock;
        std::vector<double> mFrameTimes;
        uint64_t mFrameCount = 0;
        static constexpr uint64_t kFrameWindow = 60;
    };

    inline std::string to_string(const FrameRate& fr, bool vsyncOn = false) { return fr.getMsg(vsyncOn); }
}
