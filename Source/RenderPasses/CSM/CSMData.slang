/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class CsmFilter
{
    Point,
    HwPcf,
    FixedPcf,
    Vsm,
    Evsm2,
    Evsm4,
    StochasticPcf,
};

struct CsmData
{
    static const uint32_t kMaxCascades = 8;

    float4x4 globalMat;
    float4 cascadeScale[kMaxCascades];
    float4 cascadeOffset[kMaxCascades];

    //Only uses xy
    float4 cascadeRange[kMaxCascades];  // In camera clip-space

    float depthBias = 0.005f;
    uint32_t cascadeCount = 4;
    uint32_t filterMode = (uint32_t)CsmFilter::HwPcf; // TODO: Make the field an enum when that works in Slang
    int32_t pcfKernelWidth = 5;

    float3 lightDir;
    float lightBleedingReduction = 0;

    float2 evsmExponents = float2(5.54f, 3.0f); // posExp, negExp
    float cascadeBlendThreshold = 0.0f;
    uint32_t padding;

#ifndef HOST_CODE
    Texture2DArray shadowMap;
    SamplerState csmSampler;
#endif
};

#ifdef HOST_CODE
static_assert(sizeof(CsmData) % sizeof(float4) == 0, "CsmData size should be aligned on float4 size");
#endif

END_NAMESPACE_FALCOR
