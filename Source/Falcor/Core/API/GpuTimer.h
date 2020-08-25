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
#include "Core/API/LowLevelContextData.h"
#include "Core/API/QueryHeap.h"
#include "Core/API/Buffer.h"

namespace Falcor
{
    /** Abstracts GPU timer queries. \n
        This class provides mechanism to get elapsed time in milliseconds between a pair of Begin()/End() calls.
    */
    class dlldecl GpuTimer : public std::enable_shared_from_this<GpuTimer>
    {
    public:
        using SharedPtr = std::shared_ptr<GpuTimer>;
        using SharedConstPtr = std::shared_ptr<const GpuTimer>;

        /** Create a new timer object.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create();

        /** Destroy a new object
        */
        ~GpuTimer();

        /** Begin the capture window. \n
            If begin() is called in the middle of a begin()/end() pair, it will be ignored and a warning will be logged.
        */
        void begin();

        /** Begin the capture window. \n
            If end() is called before a begin() was called, it will be ignored and a warning will be logged.
        */
        void end();

        /** Get the elapsed time in milliseconds between a pair of Begin()/End() calls. \n
            If this function called not after a Begin()/End() pair, zero will be returned and a warning will be logged.
        */
        double getElapsedTime();

    private:
        GpuTimer();

        enum class Status
        {
            Begin,
            End,
            Idle
        };

        static std::weak_ptr<QueryHeap> spHeap;
        LowLevelContextData::SharedPtr mpLowLevelData;
        Status mStatus = Status::Idle;
        uint32_t mStart;
        uint32_t mEnd;
        double mElapsedTime;
        void apiBegin();
        void apiEnd();
        void apiResolve(uint64_t result[2]);

#ifdef FALCOR_D3D12
        Buffer::SharedPtr mpResolveBuffer; // Yes, I know it's against my policy to put API specific code in common headers, but it's not worth the complications
#endif
    };
}
