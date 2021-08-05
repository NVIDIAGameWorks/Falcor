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
import RenderPasses.DebugPasses.ColorMapPass.ColorMapParams;
import Utils.Color.ColorMap;

#define FORMAT_FLOAT 0
#define FORMAT_UINT  1
#define FORMAT_SINT  2

#if _FORMAT == FORMAT_FLOAT
Texture2D<float4> gTexture;
#elif _FORMAT == FORMAT_UINT
Texture2D<uint4> gTexture;
#elif _FORMAT == FORMAT_SINT
Texture2D<int4> gTexture;
#else
#error "Invalid format"
#endif

static const uint kColorMap = _COLOR_MAP;
static const uint kChannel = _CHANNEL;

cbuffer StaticCB
{
    ColorMapParams gParams;
};

float3 applyColorMap(float value)
{
    switch (kColorMap)
    {
        case ColorMap::Grey:
            return colormapGray(value);
        case ColorMap::Jet:
            return colormapJet(value);
        case ColorMap::Viridis:
            return colormapViridis(value);
        case ColorMap::Plasma:
            return colormapPlasma(value);
        case ColorMap::Magma:
            return colormapMagma(value);
        case ColorMap::Inferno:
            return colormapInferno(value);
        default:
            return 0.f;
    }
}

float4 main(float2 texC : TEXCOORD, float4 pos : SV_Position) : SV_Target0
{
    int2 pixelPos = int2(pos.xy);

    // Read raw value from selected channel.
    float value = gTexture.Load(int3(pixelPos, 0))[_CHANNEL];

    // Normalize and colorize value.
    float normalized = gParams.minValue == gParams.maxValue ? 0.f : saturate((value - gParams.minValue) / (gParams.maxValue - gParams.minValue));
    float3 colorized = applyColorMap(normalized);

    return float4(colorized, 1.f);
}
