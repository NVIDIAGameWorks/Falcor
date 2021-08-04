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
#include "Scene/Material/MaterialDefines.slangh"

import Experimental.Scene.Material.IBxDF;
import Scene.Material.MaterialData;
import Utils.Geometry.GeometryHelpers;

/** This struct holds all the data for shading a specific hit point. This consists of:
    - Geometric data
    - Material data (type/ID, BSDF lobes to be evaluated/sampled etc)
    - Preprocessed material properties (fetched from constants/textures)
*/
struct ShadingData
{
    float3  posW;                   ///< Shading hit position in world space.
    float3  V;                      ///< Direction to the eye at shading hit.
    float3  N;                      ///< Shading normal at shading hit.
    float3  T;                      ///< Shading tangent at shading hit.
    float3  B;                      ///< Shading bitangent at shading hit.
    float2  uv;                     ///< Texture mapping coordinates.
    float   NdotV;                  ///< Unclamped, can be negative.

    // Primitive data
    float3  faceN;                  ///< Face normal in world space, always on the front-facing side.
    bool    frontFacing;            ///< True if primitive seen from the front-facing side.

    // Material data
    uint    materialID;             ///< Material ID at shading location.
    uint    flags;                  ///< Flags storing additional fields. Use the accessor functions below.

    // Pre-loaded material properties
    float3  diffuse;                ///< Diffuse albedo.
    float   opacity;
    float3  specular;               ///< Specular albedo.
    float   linearRoughness;        ///< This is the original roughness, before re-mapping.
    float3  transmission;           ///< Transmission color.
    float   ggxAlpha;               ///< DEPRECATED: This is the re-mapped roughness value, which should be used for GGX computations. It equals `roughness^2`
    float3  emissive;
    float   IoR;                    ///< Index of refraction of the medium "behind" the surface.
    float   metallic;               ///< Metallic parameter, blends between dielectric and conducting BSDFs. It is used for storing the curve sphere radius when it refers to shading curves.
    float   diffuseTransmission;    ///< Diffuse transmission, blends between diffuse reflection and transmission lobes.
    float   specularTransmission;   ///< Specular transmission, blends between opaque dielectric BRDF and specular transmissive BSDF.
    float   eta;                    ///< Relative index of refraction (incident IoR / transmissive IoR).


    // Accessor functions

    static const uint kMaterialTypeOffset = 0;
    static const uint kActiveLobesOffset = kMaterialTypeOffset + kMaterialTypeBits; // Note: kMaterialTypeBits is declared in MaterialData.slang.
    static const uint kDoubleSidedOffset = kActiveLobesOffset + kLobeTypeBits; // Note: kLobeTypeBits is declared in IBxDF.slang.
    static const uint kThinSurfaceOffset = kDoubleSidedOffset + 1;

    /** Set material type at the hit.
    */
    [mutating] void setMaterialType(MaterialType type) { flags = PACK_BITS(kMaterialTypeBits, kMaterialTypeOffset, flags, (uint)type); }

    /** Get material type at the hit.
    */
    MaterialType getMaterialType() { return MaterialType(EXTRACT_BITS(kMaterialTypeBits, kMaterialTypeOffset, flags)); }

    /** Set active BSDF lobes. See LobeType in IBxDF.slang.
    */
    [mutating] void setActiveLobes(uint activeLobes) { flags = PACK_BITS(kLobeTypeBits, kActiveLobesOffset, flags, activeLobes); }

    /** Get active BSDF lobes.
    */
    uint getActiveLobes() { return EXTRACT_BITS(kLobeTypeBits, kActiveLobesOffset, flags); }

    /** Set double-sided flag, if false only shade front face.
    */
    [mutating] void setDoubleSided(bool doubleSided) { flags = PACK_BITS(1, kDoubleSidedOffset, flags, doubleSided ? 1 : 0); }

    /** Get double-sided flag.
    */
    bool isDoubleSided() { return flags & (1u << kDoubleSidedOffset); }

    /** Set thin surface flag, if true material has a thin surface.
    */
    [mutating] void setThinSurface(bool thinSurface) { flags = PACK_BITS(1, kThinSurfaceOffset, flags, thinSurface ? 1 : 0); }

    /** Get thin surface flag.
    */
    bool isThinSurface() { return flags & (1u << kThinSurfaceOffset); }


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

    /** Transform vector from the local surface frame to world space.
        \param[in] v Vector in local space.
        \return Vector in world space.
    */
    float3 fromLocal(float3 v)
    {
        return T * v.x + B * v.y + N * v.z;
    }

    /** Transform vector from world space to the local surface frame.
        \param[in] v Vector in world space.
        \return Vector in local space.
    */
    float3 toLocal(float3 v)
    {
        return float3(dot(v, T), dot(v, B), dot(v, N));
    }
};
