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
#include "CpuTimer.h"
#include "Core/Macros.h"
#include <string>
#include <utility>
#include <vector>

namespace Falcor
{
    /** Utility class to record a number of timing measurements and print them afterwards.
        This is mainly intended for measuring longer running tasks on the CPU.
    */
    class FALCOR_API TimeReport
    {
    public:
        TimeReport();

        /** Resets the recorded measurements and the internal timer.
        */
        void reset();

        /** Resets the the internal timer but not the recoreded measurements.
        */
        void resetTimer();

        /** Prints the recorded measurements to the logfile.
        */
        void printToLog();

        /** Records a time measurement.
            Measures time since last call to reset() or measure(), whichever happened more recently.
            \param[in] name Name of the record.
        */
        void measure(const std::string& name);

        /** Add a record containing the total of all measurements.
            \param[in] name Name of the record.
        */
        void addTotal(const std::string name = "Total");

    private:
        CpuTimer::TimePoint mLastMeasureTime;
        std::vector<std::pair<std::string, double>> mMeasurements;
        double mTotal = 0.0;
    };
}
