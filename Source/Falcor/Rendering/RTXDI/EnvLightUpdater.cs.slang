/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Color.ColorHelpers;

/** Update the env light luminance and PDF textures by sampling the scene's env map.
*/
struct EnvLightUpdater
{
    RWTexture2D<float> envLightLuminance;
    RWTexture2D<float> envLightPdf;

    uint2 texDim;

    void execute(uint2 texelID)
    {
        if (any(texelID >= texDim)) return;

        // Compute UV coordinates in env map.
        float2 uv = (texelID + float2(0.5f)) / texDim;

        // Evaluate the env map luminance.
        float L = max(0.f, luminance(gScene.envMap.eval(uv)));

        // Write luminance.
        envLightLuminance[texelID] = L;

        // Compute relative solid angle to account for compression at the poles.
        float elevation = (((float(texelID.y) + 0.5f) / float(texDim.y)) - 0.5f) * M_PI;
        float relativeSolidAngle = cos(elevation);

        // Write PDF.
        envLightPdf[texelID] = relativeSolidAngle * L;
    }
};

ParameterBlock<EnvLightUpdater> gEnvLightUpdater;

[numthreads(16, 16, 1)]
void main(uint2 threadID : SV_DispatchThreadID)
{
    gEnvLightUpdater.execute(threadID);
}
