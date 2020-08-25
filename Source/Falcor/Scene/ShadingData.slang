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
import Utils.Helpers;

/** This struct holds all the data for shading a specific hit point. This consists of:
    - Geometric data
    - Preprocessed material properties (fetched from constants/textures)
    - BSDF lobes to be evaluated/sampled
*/
struct ShadingData
{
    float3  posW;                   ///< Shading hit position in world space
    float3  V;                      ///< Direction to the eye at shading hit
    float3  N;                      ///< Shading normal at shading hit
    float3  T;                      ///< Shading tangent at shading hit
    float3  B;                      ///< Shading bitangent at shading hit
    float2  uv;                     ///< Texture mapping coordinates
    float   NdotV;                  // Unclamped, can be negative.

    // Primitive data
    float3  faceN;                  ///< Face normal in world space, always on the front-facing side.
    bool    frontFacing;            ///< True if primitive seen from the front-facing side.
    bool    doubleSided;            ///< Material double-sided flag, if false only shade front face.

    // Pre-loaded texture data
    uint    materialID;             ///< Material ID at shading location.
    float3  diffuse;                ///< Diffuse albedo.
    float   opacity;
    float3  specular;               ///< Specular albedo.
    float   linearRoughness;        ///< This is the original roughness, before re-mapping.
    float   ggxAlpha;               ///< DEPRECATED: This is the re-mapped roughness value, which should be used for GGX computations. It equals `roughness^2`
    float3  emissive;
    float4  occlusion;
    float   IoR;                    ///< Index of refraction of the medium "behind" the surface.
    float   metallic;               ///< Metallic parameter, blends between dielectric and conducting BSDFs.
    float   specularTransmission;   ///< Specular transmission, blends between opaque dielectric BRDF and specular transmissive BSDF.
    float   eta;                    ///< Relative index of refraction (incident IoR / transmissive IoR).

    // Sampling/evaluation data
    uint    activeLobes;            ///< BSDF lobes to include for sampling and evaluation. See LobeType in BxDFTypes.slang.

    [mutating] void setActiveLobes(uint lobes) { activeLobes = lobes; }

    // Utility functions

    /** Computes new ray origin based on the hit point to avoid self-intersection.
        The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
        Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.
        \param[in] viewside True if the origin should be on the view side (reflection) or false otherwise (transmission).
        \return Ray origin of the new ray.
    */
    float3 computeNewRayOrigin(bool viewside = true)
    {
        return computeRayOrigin(posW, (frontFacing == viewside) ? faceN : -faceN);
    }
};
