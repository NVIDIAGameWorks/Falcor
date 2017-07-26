/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THEa
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

#include "SceneEditorCommon.slang.h"

#ifdef DEBUG_DRAW

float4 main(DebugDrawVSOut vOut) : SV_TARGET
{
    return float4(vOut.color, 1);
}

#else //////////////////////////////////////////////////////////////////////////////////////

#ifndef PICKING
cbuffer ConstColorCB : register(b0)
{
    float3 gColor;
};
#endif

// PS Output
#ifdef PICKING
#define PSOut uint
#else
#define PSOut float4
#endif

PSOut main(EditorVSOut vOut) : SV_TARGET
{
    float3 toCamera = normalize(gCam.position - vOut.vOut.posW);

#ifdef CULL_REAR_SECTION
    if(dot(toCamera, vOut.toVertex) < -0.1)
    {
        discard;
    }
#endif

#ifdef PICKING
    return vOut.drawID;
#else
    #ifdef SHADING
        float shading = lerp(0.2f, 1.0f, dot(toCamera, normalize(vOut.vOut.normalW)));
        return float4(gColor * shading, 1);
    #else
        return float4(gColor, 1);
    #endif
#endif
}

#endif
