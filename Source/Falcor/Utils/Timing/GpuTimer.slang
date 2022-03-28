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
#include "Utils/NVAPI.slangh" // We need this for accessing the global timer.

/** GPU timer based on NVAPI.
    Uses only the lower 32-bits of the global timer. This should be sufficient to do
    most measurements but care needs to be taken for handling wrap around.
*/
struct GpuTimer
{
    typedef uint TimePoint;

    TimePoint startTime;    ///< Time when timer was last started.

    /** Starts the timer.
    */
    [mutating] void start()
    {
        startTime = getCurrentTimePoint();
    }

    /** Returns the elapsed time since the last call to start().
        \return Returns the elapsed time.
    */
    uint getElapsed()
    {
        TimePoint currentTime = getCurrentTimePoint();
        return calcDuration(startTime, currentTime);
    }

    /** Get the current time point.
    */
    static TimePoint getCurrentTimePoint()
    {
#if FALCOR_NVAPI_AVAILABLE
        return (TimePoint)NvGetSpecial(NV_SPECIALOP_GLOBAL_TIMER_LO);
#else
        return 0;
#endif
    }

    /** Compute the duration between two time points accounting for at most one overflow.
        \param[in] start Start time point
        \param[in] end End time point
        \return Returns the duration.
    */
    static uint calcDuration(TimePoint start, TimePoint end)
    {
        return end - start;
    }
};
