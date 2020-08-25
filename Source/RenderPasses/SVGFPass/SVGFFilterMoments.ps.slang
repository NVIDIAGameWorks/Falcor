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
import Scene.ShadingData;
import Utils.Helpers;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import SVGFCommon;

cbuffer PerImageCB
{
    Texture2D   gIllumination;
    Texture2D   gMoments;
    Texture2D   gHistoryLength;
    Texture2D   gLinearZAndNormal;

    float       gPhiColor;
    float       gPhiNormal;
};

float4 main(FullScreenPassVsOut vsOut) : SV_TARGET0
{
    float4 posH = vsOut.posH;
    int2 ipos = int2(posH.xy);

    float h = gHistoryLength[ipos].x;
    int2 screenSize = getTextureDims(gHistoryLength, 0);

    if (h < 4.0) // not enough temporal history available
    {
        float sumWIllumination   = 0.0;
        float3 sumIllumination   = float3(0.0, 0.0, 0.0);
        float2 sumMoments  = float2(0.0, 0.0);

        const float4 illuminationCenter = gIllumination[ipos];
        const float lIlluminationCenter = luminance(illuminationCenter.rgb);

        const float2 zCenter = gLinearZAndNormal[ipos].xy;
        if (zCenter.x < 0)
        {
            // current pixel does not a valid depth => must be envmap => do nothing
            return illuminationCenter;
        }
        const float3 nCenter = oct_to_ndir_snorm(gLinearZAndNormal[ipos].zw);
        const float phiLIllumination   = gPhiColor;
        const float phiDepth     = max(zCenter.y, 1e-8) * 3.0;

        // compute first and second moment spatially. This code also applies cross-bilateral
        // filtering on the input illumination.
        const int radius = 3;

        for (int yy = -radius; yy <= radius; yy++)
        {
            for (int xx = -radius; xx <= radius; xx++)
            {
                const int2 p     = ipos + int2(xx, yy);
                const bool inside = all(p >= int2(0,0)) && all(p < screenSize);
                const bool samePixel = (xx ==0 && yy == 0);
                const float kernel = 1.0;

                if (inside)
                {
                    const float3 illuminationP = gIllumination[p].rgb;
                    const float2 momentsP      = gMoments[p].xy;
                    const float lIlluminationP = luminance(illuminationP.rgb);
                    const float zP = gLinearZAndNormal[p].x;
                    const float3 nP = oct_to_ndir_snorm(gLinearZAndNormal[p].zw);

                    const float w = computeWeight(
                        zCenter.x, zP, phiDepth * length(float2(xx, yy)),
                        nCenter, nP, gPhiNormal,
                        lIlluminationCenter, lIlluminationP, phiLIllumination);

                    sumWIllumination += w;
                    sumIllumination  += illuminationP * w;
                    sumMoments += momentsP * w;
                }
            }
        }

        // Clamp sum to >0 to avoid NaNs.
        sumWIllumination = max(sumWIllumination, 1e-6f);

        sumIllumination   /= sumWIllumination;
        sumMoments  /= sumWIllumination;

        // compute variance using the first and second moments
        float variance = sumMoments.g - sumMoments.r * sumMoments.r;

        // give the variance a boost for the first frames
        variance *= 4.0 / h;

        return float4(sumIllumination, variance.r);
    }
    else
    {
        // do nothing, pass data unmodified
        return gIllumination[ipos];
    }
}
