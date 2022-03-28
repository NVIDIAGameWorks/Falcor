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

/** Fullscreen pass to draw the scene as background behind the probes.
*/
import LoadShadingData;
import VisualizerDefines;

cbuffer CB
{
    uint2 gFrameDim;
    uint2 gRadianceTexDim;
    uint gVisualizerSceneMode; // TODO: Replace by VisualizerSceneMode enum when enums can be placed in structs 
}

Texture2D<float4> gRadianceTex;
RWTexture2D<float4> gOutput;


[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= gFrameDim)) return;

    float3 output = float3(0);

    if (gVisualizerSceneMode == (uint)VisualizerSceneMode::RadianceTexture)
    {
        output = all(pixel < gRadianceTexDim) ? gRadianceTex[pixel].rgb : float3(0.1f);
    }
    else
    {
        let lod = createTextureSampler(pixel);

        ShadingData sd;
        if (loadShadingData(pixel, gFrameDim, gScene.camera, lod, sd))
        {
            if (gVisualizerSceneMode == (uint)VisualizerSceneMode::Albedo)
            {
                // Create BSDF instance.
                let bsdf = gScene.materials.getBSDF(sd, lod);

                // We assume a perfect Lambertian diffuse lobe.
                // Visualize the albedo here for debugging purposes.
                output = bsdf.getProperties(sd).diffuseReflectionAlbedo;
            }
            else if (gVisualizerSceneMode == (uint)VisualizerSceneMode::Normals)
            {
                output = saturate(sd.N);
            }
            else output = float3(1, 0, 0); // Should not happen
        }
    }

    gOutput[pixel] = float4(output, 1.f);
}
