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
 **************************************************************************/
__exported import Scene.ShadingData;
import Scene.Lights.Lights;
import Scene.Material.BRDF;

/** Shading result struct
*/
struct ShadingResult
{
    float3 diffuseBrdf;             // The result of the diffuse BRDF
    float3 specularBrdf;            // The result of the specular BRDF
    float3 diffuse;                 // The diffuse component of the result
    float3 specular;                // The specular component of the result
    float4 color;                   // The final color. Alpha holds the opacity valua
};

ShadingResult initShadingResult()
{
    ShadingResult sr;
    sr.diffuse = 0;
    sr.color.rgb = 0;
    sr.color.a = 1;
    sr.specular = 0;
    sr.diffuseBrdf = 0;
    sr.specularBrdf = 0;
    return sr;
}

ShadingResult evalMaterial(ShadingData sd, LightData light, float shadowFactor)
{
    ShadingResult sr = initShadingResult();
    LightSample ls = evalLight(light, sd);

    // If the light doesn't hit the surface or we are viewing the surface from the back, return
    if (ls.NdotL <= 0) return sr;
    sd.NdotV = saturate(sd.NdotV);

    // Calculate the diffuse term
    sr.diffuseBrdf = evalDiffuseBrdf(sd, ls);
    sr.diffuse = ls.diffuse * sr.diffuseBrdf * ls.NdotL;
    sr.color.rgb = sr.diffuse;
    sr.color.a = sd.opacity;

    // Calculate the specular term
    sr.specularBrdf = evalSpecularBrdf(sd, ls);
    sr.specular = ls.specular * sr.specularBrdf * ls.NdotL;
    sr.color.rgb += sr.specular;

    // Apply the shadow factor
    sr.color.rgb *= shadowFactor;

    return sr;
};

ShadingResult evalMaterial(ShadingData sd, LightProbeData probe)
{
    ShadingResult sr = initShadingResult();
    LightSample ls = evalLightProbe(probe, sd);

    sr.diffuse = ls.diffuse;
    sr.color.rgb = sr.diffuse;
    sr.specular = ls.specular;
    sr.color.rgb += sr.specular;
    return sr;
}
