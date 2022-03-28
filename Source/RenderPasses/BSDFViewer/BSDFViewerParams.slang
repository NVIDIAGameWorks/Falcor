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
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class BSDFViewerMode : uint32_t
{
    Material,       ///< Rendered view with material sample.
    Slice,          ///< BSDF slice viewer.
};

enum class AlbedoSelection : uint32_t
{
    ShowAlbedo = 0x1,
    DiffuseReflection = 0x2,
    DiffuseTransmission = 0x4,
    SpecularReflection = 0x8,
    SpecularTransmission = 0x10,
};

/** BSDFViewer parameters shared between host and device.
    Make sure struct layout follows the HLSL packing rules as it is uploaded as a memory blob.
    Do not use bool's as they are 1 byte in Visual Studio, 4 bytes in HLSL.
    https://msdn.microsoft.com/en-us/library/windows/desktop/bb509632(v=vs.85).aspx
*/
struct BSDFViewerParams
{
    uint2   frameDim = { 0, 0 };                ///< Frame buffer dimension in pixels.
    uint    frameCount = 0;                     ///< Frames rendered.
    BSDFViewerMode viewerMode = BSDFViewerMode::Material; ///< Current viewer mode.

    float2  viewportOffset;                     ///< Top-left corner of viewport in pixels.
    float2  viewportScale;                      ///< 1/Size of viewport in pixels.

    // Material parameters
    uint    materialID = 0;                     ///< Scene material ID.
    int     useNormalMapping = 0;               ///< Use normal mapping.
    int     useFixedTexCoords = 0;              ///< Use fixed texture coordinates.
    int     _pad0;

    float2  texCoords = { 0.f, 0.f };           ///< Texture coordinates to use when 'useFixedTexCoords' is true.
    float2  _pad1;

    // BSDF settings
    int     useDisneyDiffuse = 0;               ///< Use the original Disney diffuse BRDF, otherwise Falcor's default (Frostbite's version).
    int     useSeparableMaskingShadowing = 0;   ///< Use the separable form of the masking-shadowing function, otherwise Falcor's default (the correlated form).
    int     useImportanceSampling = 1;          ///< Use importance sampling.
    int     usePdf = 0;                         ///< Use BRDF sampling pdf explicitly, otherwise the precomputed weight (for debugging).

    uint    outputAlbedo = 0;                   ///< Whether to output albedo instead of reflectance. See AlbedoSelection flags.
    int     enableDiffuse = 1;                  ///< Enable diffuse lobe in slice viewer.
    int     enableSpecular = 1;                 ///< Enable specular lobe in slice viewer.
    int     applyNdotL = 0;                     ///< Multiply BSDF by NdotL in slice viewer.

    // Lighting settings
    int     useGroundPlane = 0;                 ///< Draw a ground plane.
    int     useEnvMap = 0;                      ///< Use environment light (as opposed to omnidirectional).
    int2    _pad3;

    float   lightIntensity = 1.f;               ///< Light intensity, acts as a multiplier for the light color.
    float3  lightColor = { 1.f, 1.f, 1.f };     ///< Light color.

    int     useDirectionalLight = 0;            ///< Use directional light (as opposed to omnidirectional/envmap).
    float3  lightDir = { 0.f, 0.f, -1.f };      ///< Light direction to use when 'useDirectionalLight' is true (note: not normalized).

    // Camera settings
    int     orthographicCamera = 0;             ///< Use orthographic camera.
    float   cameraDistance = 1.5f;              ///< Camera distance from origin in projective mode. Valid range is (1,+inf).
    float   cameraFovY = 90.f;                  ///< Camera vertical field-of-view in degrees.
    float   cameraViewportScale;                ///< Camera viewport scale (= tan(fovY/2)*distance) computed at runtime in projective mode.

    // Misc settings
    int2    selectedPixel = { 0, 0 };           ///< Pixel coordinates selected for readback.
    int2    _pad5;
};

/** Struct for readback of per-pixel data.
*/
struct PixelData
{
    float2 texC;
    float3 T;
    float3 B;
    float3 N;
    float3 wi;
    float3 wo;
    float3 output;

    // BSDF properties
    float3 emission;
    float  roughness;
    float3 diffuseReflectionAlbedo;
    float3 diffuseTransmissionAlbedo;
    float3 specularReflectionAlbedo;
    float3 specularTransmissionAlbedo;
    float3 specularReflectance;
    int    isTransmissive;
};

END_NAMESPACE_FALCOR
