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
#ifndef CSMDATA_H
#define CSMDATA_H

#include "Data/HostDeviceData.h"

#define CSM_MAX_CASCADES 8

#define CsmFilterPoint 0
#define CsmFilterHwPcf 1
#define CsmFilterFixedPcf 2
#define CsmFilterVsm   3
#define CsmFilterEvsm2 4
#define CsmFilterEvsm4 5
#define CsmFilterStochasticPcf 6

struct CsmData
{
    float4x4 globalMat;
    float4 cascadeScale[CSM_MAX_CASCADES];
    float4 cascadeOffset[CSM_MAX_CASCADES];

    //Only uses xy
    float4 cascadeRange[CSM_MAX_CASCADES];  // In camera clip-space

    float depthBias DEFAULTS(0.005f);
    int cascadeCount DEFAULTS(4);
    uint32_t filterMode DEFAULTS(CsmFilterHwPcf);
    int32_t pcfKernelWidth DEFAULTS(5);
    
    float3 lightDir;
    float lightBleedingReduction DEFAULTS(0);

    float2 evsmExponents DEFAULTS(float2(5.54f, 3.0f)); // posExp, negExp
    float cascadeBlendThreshold DEFAULTS(0.0f);
    uint32_t padding;

#ifndef HOST_CODE
    Texture2DArray shadowMap;
    SamplerState csmSampler;
#endif
};

#ifdef HOST_CODE
static_assert(sizeof(CsmData) % sizeof(float4) == 0, "CsmData size should be aligned on float4 size");
#endif
#endif //CSMDATA_H
