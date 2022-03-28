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
import Scene.Raster;
import Utils.Math.MathHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EnvMapLighting;
import Utils.Sampling.TinyUniformSampleGenerator;

cbuffer PerFrameCB
{
    float2 gRenderTargetDim;
    float  gFrameCount;
};

EnvMapLighting gEnvMapLighting;
SamplerState gSampler;
Texture2D visibilityBuffer;
static VSOut vsData;

VSOut vsMain(VSIn vIn)
{
    VSOut vsOut;
    vsOut = defaultVS(vIn);
#ifdef _OUTPUT_MOTION_VECTORS
    vsOut.prevPosH.xy += vsOut.prevPosH.w * 2 * float2(gScene.camera.data.jitterX, gScene.camera.data.jitterY);
#endif
    return vsOut;
}

struct PsOut
{
    float4 color : SV_TARGET0;
    float4 normal : SV_TARGET1;
#ifdef _OUTPUT_MOTION_VECTORS
    float2 motion : SV_TARGET2;
#endif
};

PsOut psMain(VSOut vsOut, uint triangleIndex : SV_PrimitiveID)
{
    PsOut psOut;

    let lod = ImplicitLodTextureSampler();
    if (alphaTest(vsOut, triangleIndex, lod)) discard;

    float3 viewDir = normalize(gScene.camera.getPosition() - vsOut.posW);
    ShadingData sd = prepareShadingData(vsOut, triangleIndex, viewDir, lod);

    // Create BSDF instance and query its properties.
    let bsdf = gScene.materials.getBSDF(sd, lod);
    let bsdfProperties = bsdf.getProperties(sd);

    float4 finalColor = float4(0, 0, 0, 1);

    // Direct lighting from analytic light sources
    const uint2 pixel = vsOut.posH.xy;
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gFrameCount);
    for (int i = 0; i < gScene.getLightCount(); i++)
    {
        float shadowFactor = 1;
        if (i == 0)
        {
            shadowFactor = visibilityBuffer.Load(int3(vsOut.posH.xy, 0)).r;
            shadowFactor *= sd.opacity;
        }

        AnalyticLightSample ls;
        evalLightApproximate(sd.posW, gScene.getLight(i), ls);
        finalColor.rgb += bsdf.eval(sd, ls.dir, sg) * ls.Li * shadowFactor;
    }

    // Add the emissive component
    finalColor.rgb += bsdf.getProperties(sd).emission;
    finalColor.a = sd.opacity;

#ifdef _USE_ENV_MAP
    // Add lighting from environment map
    finalColor.rgb += gEnvMapLighting.evalDiffuse(sd, bsdfProperties);
    finalColor.rgb += gEnvMapLighting.evalSpecular(sd, bsdfProperties, reflect(-sd.V, sd.N));
#endif

    psOut.color = finalColor;
    psOut.normal = float4(sd.N * 0.5f + 0.5f, 1.0f);

#ifdef _OUTPUT_MOTION_VECTORS
    // Using vsOut.posH.xy as pixel coordinate since it has the SV_Position semantic.
    psOut.motion = calcMotionVector(vsOut.posH.xy, vsOut.prevPosH, gRenderTargetDim);
#endif

#if defined(_VISUALIZE_CASCADES) && defined(_ENABLE_SHADOWS)
    float3 cascadeColor = visibilityBuffer.Load(int3(vsOut.posH.xy, 0)).gba;
    psOut.color.rgb *= cascadeColor;
#endif
    return psOut;
}
