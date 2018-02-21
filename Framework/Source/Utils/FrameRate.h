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
#include <chrono>
#include <vector>
#include "CpuTimer.h"

namespace Falcor
{
    /** Framerate calculator
    */
    class FrameRate
    {
    public:
        FrameRate()
        {
            mFrameTimes.resize(sFrameWindow);
            resetClock();
        }

        /** Resets the calculator.
            After this call it will appear as if the application had just started. Useful in cases a new scene is loaded, since it will display a more accurate FPS.
        */
        void resetClock()
        {
            newFrame();
            mFrameCount = 0;
        }

        /** Tick the timer.
            It is assumed that this is called once per frame, since this frequency is assumed when calculating FPS.
        */
        void newFrame()
        {
            mFrameCount++;
            mTimer.update();
            mFrameTimes[mFrameCount % sFrameWindow] = mTimer.getElapsedTime();
        }

        /** Get the time in ms it took to render a frame
        */
        float getAverageFrameTime() const
        {
            uint64_t frames = min(mFrameCount, sFrameWindow);
            double elapsedTime = 0;
            for(uint64_t i = 0; i < frames; i++)
            {
                elapsedTime += mFrameTimes[i];
            }

            double time = elapsedTime / double(frames) * 1000;
            return float(time);
        }

        /** Get the time that passed from the last NewFrame() call to the one before that.
        */
        float getLastFrameTime() const
        {
            return mFrameTimes[mFrameCount % sFrameWindow];
        }

        /** Get the numer of frames passed from the last resetClock() call.
        */
        uint64_t getFrameCount() const
        {
            return mFrameCount;
        }
    private:

        CpuTimer mTimer;
        std::vector<float> mFrameTimes;
        uint64_t mFrameCount;
        static const uint64_t sFrameWindow = 60;
    };
}