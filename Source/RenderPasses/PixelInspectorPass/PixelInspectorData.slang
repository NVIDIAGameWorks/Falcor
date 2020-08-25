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
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Pixel data read out by the InspectorPass.
*/
struct PixelData
{
    static const uint kInvalidIndex = 0xffffffff;

    // Geometry data
    float3 posW;
    float texCoordU;

    float3 normal;
    float texCoordV;

    float3 tangent;
    float NdotV;

    float3 bitangent;
    float _pad0;

    float3 faceNormal;
    float _pad3;

    float3 view;
    float _pad1;

    // Material data
    float3 diffuse;
    float opacity;

    float3 specular;
    float linearRoughness;

    float3 emissive;
    float ggxAlpha;

    float IoR;
    uint doubleSided;
    float luminance;
    float _pad2;

    // Output data
    float4 linearColor;
    float4 outputColor;

    // Visibility data
    uint meshInstanceID;
    uint triangleIndex;
    float2 barycentrics;

#ifdef HOST_CODE
    PixelData()
    {
        init();
    }
#endif

#ifdef HOST_CODE
    void init()
#else
    [mutating] void init()
#endif
    {
        posW = float3(0, 0, 0);
        texCoordU = 0;

        normal = float3(0, 0, 0);
        texCoordV = 0;

        tangent = float3(0, 0, 0);
        NdotV = 0;

        bitangent = float3(0, 0, 0);
        _pad0 = 0;

        faceNormal = float3(0, 0, 0);
        _pad3 = 0;

        view = float3(0, 0, 0);
        _pad1 = 0;

        diffuse = float3(0, 0, 0);
        opacity = 0;

        specular = float3(0, 0, 0);
        linearRoughness = 0;

        emissive = float3(0, 0, 0);
        ggxAlpha = 0;

        IoR = 0;
        doubleSided = 0;
        luminance = 0;
        _pad2 = 0;

        linearColor = float4(0, 0, 0, 0);
        outputColor = float4(0, 0, 0, 0);

        meshInstanceID = kInvalidIndex;
        triangleIndex = kInvalidIndex;
        barycentrics = float2(0, 0);
    }
};

END_NAMESPACE_FALCOR
