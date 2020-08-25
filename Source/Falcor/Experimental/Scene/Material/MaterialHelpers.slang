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

/** Helper functions for the material/shading system.

    The data needed to shade a hit point consists of two parts (jointly represented by Falcor's 'ShadingData' struct):
    1) geometry parameters (position, normal etc.)
    2) material parameters (diffuse color, opacity etc.)

    This file defines a struct 'GeometryParams' to represent the hit point geometry.
    This is useful in passes that don't need the material parameters (for example, for shadows or AO).
    Another struct 'MaterialParams' provides a compact representation of the material parameters,
    suitable for storing in the G-buffer.
*/

__exported import Scene.ShadingData;
import Experimental.Scene.Material.BxDFTypes;

/** Struct representing the material parameters at a hit point.
*/
struct MaterialParams
{
    float4 diffuseOpacity;      ///< Diffuse color (rgb) and opacity (a).
    float4 specularRoughness;   ///< Specular color (rgb) and roughness (a).
    float4 emissive;            ///< Emissive color (rgb). TODO: Use the alpha channel either for a luminance scaler, or 1.0/IoR.
    float4 extraParams;         ///< Extra parameters (IoR, doubleSided, specularTransmission, metallic).

    /** Returns the roughness parameter.
    */
    float getRoughness() { return specularRoughness.w; }
};

/** Struct representing the geometry at a hit point, plus a few related fields.
*/
struct GeometryParams
{
    float3 posW;                ///< World-space position.
    float3 V;                   ///< Normalized view direction in world space.
    float2 uv;                  ///< Texture coordinate at hit point, or 0 if not available.
    float3 faceN;               ///< Face normal in world space.

    // Surface frame in world space. The axes (T,B,N) form an orthonormal basis.
    // The normal and tangent are normally provided by the framework, while bitangent is computed based on those.
    float3 N;                   ///< Shading normal (z-axis in local frame).
    float3 B;                   ///< Shading bitangent (y-axis in local frame).
    float3 T;                   ///< Shading tangent (x-axis in local frame).
};

/******************************************************************************
                                Material helpers
******************************************************************************/

/** Extract material params from ShadingData struct.
*/
MaterialParams getMaterialParams(ShadingData sd)
{
    MaterialParams m;
    m.diffuseOpacity = float4(sd.diffuse, sd.opacity);
    m.specularRoughness = float4(sd.specular, sd.linearRoughness);
    m.emissive = float4(sd.emissive, 0.f);
    m.extraParams = float4(sd.IoR, sd.doubleSided ? 1.f : 0.f, sd.specularTransmission, sd.metallic);
    return m;
}

/******************************************************************************
                                Geometry helpers
******************************************************************************/

/** Helper function for preparing a GeometryParams struct based on the local geometry.
    Note that the input vectors must not be (0,0,0) as the normalizations would cause NaNs.
    \param[in] worldPos World space position.
    \param[in] viewDir View direction (unnormalized).
    \param[in] normal Shading normal (unnormalized).
    \param[in] tangent Shading tangent (xyz) and sign (w) (unnormalized).
    \param[in] faceNormal Face normal (normalized).
    \param[in] texCrd Texture uv coordinate, or (0,0) if not available.
    \return GeometryParams struct.
*/
GeometryParams prepareGeometryParams(float3 worldPos, float3 viewDir, float3 normal, float4 tangent, float3 faceNormal, float2 texCrd = float2(0, 0))
{
    GeometryParams p;
    p.posW = worldPos;
    p.V = normalize(viewDir);
    p.N = normalize(normal);
    p.T = normalize(tangent.xyz - p.N * dot(tangent.xyz, p.N));
    p.B = cross(p.N, p.T) * tangent.w;
    p.uv = texCrd;
    p.faceN = faceNormal;
    return p;
}

/** Extract geometry params from ShadingData struct.
*/
GeometryParams getGeometryParams(ShadingData sd)
{
    GeometryParams p;
    p.posW = sd.posW;
    p.V = sd.V;
    p.N = sd.N;
    p.T = sd.T;
    p.B = sd.B;
    p.uv = sd.uv;
    p.faceN = sd.faceN;
    return p;
}

/** Helper function for populating the geometry and material parameters of a ShadingData struct.
*/
ShadingData prepareShadingData(GeometryParams geoParams, MaterialParams matParams)
{
    ShadingData sd = {};

    sd.posW = geoParams.posW;
    sd.V = geoParams.V;
    sd.N = geoParams.N;
    sd.B = geoParams.B;
    sd.T = geoParams.T;
    sd.uv = geoParams.uv;

    // Precompute N*V the same way as in Falcor (no clamp/epsilon).
    sd.NdotV = dot(sd.N, sd.V);
    sd.faceN = geoParams.faceN;
    sd.frontFacing = dot(sd.V, sd.faceN) >= 0.f;

    sd.diffuse = matParams.diffuseOpacity.rgb;
    sd.opacity = matParams.diffuseOpacity.a;
    sd.specular = matParams.specularRoughness.rgb;
    sd.linearRoughness = matParams.specularRoughness.a;
    // TODO: ggxAlpha should be removed and computed where it's needed.
    // The threshold of 0.0064 is because in previous versions we used to clamp linearRoughness to 0.08.
    sd.ggxAlpha = max(0.0064, sd.linearRoughness * sd.linearRoughness);
    sd.emissive = matParams.emissive.rgb;
    sd.IoR = matParams.extraParams.x;
    sd.eta = sd.frontFacing ? 1.f / sd.IoR : sd.IoR;
    sd.doubleSided = matParams.extraParams.y != 0.f;
    sd.specularTransmission = matParams.extraParams.z;
    sd.metallic = matParams.extraParams.w;

    sd.occlusion = 1.f;

    sd.activeLobes = (uint)LobeType::All;

    return sd;
}

/** Helper function to transform vector v from the local surface frame to world space.
*/
float3 fromLocal(float3 v, ShadingData sd)
{
    return sd.T * v.x + sd.B * v.y + sd.N * v.z;
}

/** Helper function to transform vector v from the local surface frame to world space.
*/
float3 fromLocal(float3 v, GeometryParams sd)
{
    return sd.T * v.x + sd.B * v.y + sd.N * v.z;
}

/** Helper function to transform world space vector v to the local surface frame.
*/
float3 toLocal(float3 v, ShadingData sd)
{
    return float3(dot(v, sd.T), dot(v, sd.B), dot(v, sd.N));
}

/** Helper function to transform world space vector v to the local surface frame.
*/
float3 toLocal(float3 v, GeometryParams sd)
{
    return float3(dot(v, sd.T), dot(v, sd.B), dot(v, sd.N));
}
