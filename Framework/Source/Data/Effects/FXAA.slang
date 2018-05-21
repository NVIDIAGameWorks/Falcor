/***************************************************************************
Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
***************************************************************************/
Texture2D gSrc;
SSamplerState gSampler;

float4 ps(float2 texC : TEXCOORD) : SV_TARGET
{
    return gSrc.Sample(gSampler, texC);
}