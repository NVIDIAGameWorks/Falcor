/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
    Material, ///< Rendered view with material sample.
    Slice,    ///< BSDF slice viewer.
};

FALCOR_ENUM_INFO(
    BSDFViewerMode,
    {
        { BSDFViewerMode::Material, "Material" },
        { BSDFViewerMode::Slice, "Slice" },
    }
);
FALCOR_ENUM_REGISTER(BSDFViewerMode);

enum class AlbedoSelection : uint32_t
{
    ShowAlbedo = 0x1,
    DiffuseReflection = 0x2,
    DiffuseTransmission = 0x4,
    SpecularReflection = 0x8,
    SpecularTransmission = 0x10,
};

/**
 * BSDFViewer parameters shared between host and device.
 * Make sure struct layout follows the HLSL packing rules as it is uploaded as a memory blob.
 * Do not use bool's as they are 1 byte in Visual Studio, 4 bytes in HLSL.
 * https://msdn.microsoft.com/en-us/library/windows/desktop/bb509632(v=vs.85).aspx
 */
struct BSDFViewerParams
{
    /// Frame buffer dimension in pixels.
    uint2 frameDim = { 0, 0 };
    /// Frames rendered.
    uint frameCount = 0;
    /// Current viewer mode.
    BSDFViewerMode viewerMode = BSDFViewerMode::Material;

    /// Top-left corner of viewport in pixels.
    float2 viewportOffset;
    /// 1/Size of viewport in pixels.
    float2 viewportScale;

    // Material parameters

    /// Scene material ID.
    uint materialID = 0;
    /// Use normal mapping.
    int useNormalMapping = 0;
    /// Use fixed texture coordinates.
    int useFixedTexCoords = 0;
    int _pad0;

    /// Texture coordinates to use when 'useFixedTexCoords' is true.
    float2 texCoords = { 0.f, 0.f };
    float2 _pad1;

    // BSDF settings

    /// Use the original Disney diffuse BRDF, otherwise Falcor's default (Frostbite's version).
    int useDisneyDiffuse = 0;
    /// Use the separable form of the masking-shadowing function, otherwise Falcor's default (the correlated form).
    int useSeparableMaskingShadowing = 0;
    /// Use importance sampling.
    int useImportanceSampling = 1;
    /// Use BRDF sampling pdf explicitly, otherwise the precomputed weight (for debugging).
    int usePdf = 0;

    /// Whether to output albedo instead of reflectance. See AlbedoSelection flags.
    uint outputAlbedo = 0;
    /// Enable diffuse lobe in slice viewer.
    int enableDiffuse = 1;
    /// Enable specular lobe in slice viewer.
    int enableSpecular = 1;
    /// Multiply BSDF by NdotL in slice viewer.
    int applyNdotL = 0;

    // Lighting settings
    /// Draw a ground plane.
    int useGroundPlane = 0;
    /// Use environment light (as opposed to omnidirectional).
    int useEnvMap = 0;
    int2 _pad3;

    /// Light intensity, acts as a multiplier for the light color.
    float lightIntensity = 1.f;
    /// Light color.
    float3 lightColor = { 1.f, 1.f, 1.f };

    /// Use directional light (as opposed to omnidirectional/envmap).
    int useDirectionalLight = 0;
    /// Light direction to use when 'useDirectionalLight' is true (note: not normalized).
    float3 lightDir = { 0.f, 0.f, -1.f };

    // Camera settings

    /// Use orthographic camera.
    int orthographicCamera = 0;
    /// Camera distance from origin in projective mode. Valid range is (1,+inf).
    float cameraDistance = 1.5f;
    /// Camera vertical field-of-view in degrees.
    float cameraFovY = 90.f;
    /// Camera viewport scale (= tan(fovY/2)*distance) computed at runtime in projective mode.
    float cameraViewportScale;

    // Misc settings

    /// Pixel coordinates selected for readback.
    int2 selectedPixel = { 0, 0 };
    int2 _pad5;
};

/**
 * Struct for readback of per-pixel data.
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
    float3 guideNormal;
    float3 emission;
    float roughness;
    float3 diffuseReflectionAlbedo;
    float3 diffuseTransmissionAlbedo;
    float3 specularReflectionAlbedo;
    float3 specularTransmissionAlbedo;
    float3 specularReflectance;
    int isTransmissive;
};

END_NAMESPACE_FALCOR
