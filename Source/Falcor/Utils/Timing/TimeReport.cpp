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
#include "TimeReport.h"
#include "Utils/Logger.h"
#include "Utils/StringUtils.h"
#include <numeric>

namespace Falcor
{
    TimeReport::TimeReport()
    {
        reset();
    }

    void TimeReport::reset()
    {
        mLastMeasureTime = CpuTimer::getCurrentTimePoint();
        mMeasurements.clear();
        mTotal = 0.0;
    }

    void TimeReport::resetTimer()
    {
        mLastMeasureTime = CpuTimer::getCurrentTimePoint();
        mTotal = 0.0;
    }

    void TimeReport::printToLog()
    {
        for (const auto& [task, duration] : mMeasurements)
        {
            logInfo(padStringToLength(task + ":", 25) + " " + std::to_string(duration) + " s" + (mTotal > 0.0 && !mMeasurements.empty() ? ", " + std::to_string(100.0 * duration / mTotal) + "% of total" : ""));
        }
    }

    void TimeReport::measure(const std::string& name)
    {
        auto currentTime = CpuTimer::getCurrentTimePoint();
        std::chrono::duration<double> duration = currentTime - mLastMeasureTime;
        mLastMeasureTime = currentTime;
        mMeasurements.push_back({name, duration.count()});
    }

    void TimeReport::addTotal(const std::string name)
    {
        mTotal = std::accumulate(mMeasurements.begin(), mMeasurements.end(), 0.0, [] (double t, auto &&m) { return t + m.second; });
        mMeasurements.push_back({"Total", mTotal});
    }
}
