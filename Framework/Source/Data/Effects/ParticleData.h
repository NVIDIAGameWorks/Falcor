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
#ifndef PARTICLEDATA_H
#define PARTICLEDATA_H

#include "Data/HostDeviceData.h"

#define EMIT_THREADS 64
#define SORT_THREADS 1024

struct Particle
{
    float3 pos;
    float scale;
    float3 vel;    
    float life;
    float3 accel;
    float growth;
    float rot; 
    float rotVel;
    float2 padding1;
    //id?
};

struct EmitData
{
    uint numEmit;
    uint maxParticles;
    float2 padding;
};

struct SortData
{
    int index;
    float depth;
};

struct ColorInterpPsPerFrame
{
    float4 color1;
    float4 color2;
    float colorT1;
    float colorT2;
};

struct SimulatePerFrame
{
    float dt;
    uint maxParticles;
};

struct VSPerFrame
{
    float4x4 view;
    float4x4 proj;
};

struct SimulateWithSortPerFrame
{
    float4x4 view;
    float dt;
    uint maxParticles;
    float2 padding;
};

#ifndef HOST_CODE
uint getParticleIndex(uint groupIDx, uint threadsPerGroup, uint groupIndex)
{
    return groupIDx * threadsPerGroup + groupIndex;
}
#endif

#endif //particleData_h
