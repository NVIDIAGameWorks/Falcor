/***************************************************************************
 # Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/

Texture2D<float4>   gReference;
Texture2D<float4>   gSource;
Texture2D<float4>   gWorldPosition;
RWTexture2D<float4> gResult;

cbuffer PerFrameCB
{
    uint2   gResolution;
    uint    gIgnoreBackground;
    uint    gComputeDiffSqr;
};

[numthreads(16, 16, 1)]
void main( uint3 DTid : SV_DispatchThreadID )
{
    const uint2 pixel = DTid.xy;
    if (any(pixel >= gResolution)) return;

    // Determine if we should include this pixel or not.
    const float4 worldPos = gWorldPosition[pixel]; 
    const bool isForeground = worldPos.w != 0.0f;  // We're using the w-component to identify valid pixels.

    const bool isPixelValid = !gIgnoreBackground || isForeground;

    // Compute error based on the current options.
    float3 diff = isPixelValid ? abs(gSource[pixel].rgb - gReference[pixel].rgb) : float3(0.0f);
    if (gComputeDiffSqr) diff *= diff;

    gResult[pixel] = float4(diff, 0.0f);
}
