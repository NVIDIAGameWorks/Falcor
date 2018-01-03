/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#ifndef _HOST_DEVICE_SHARED_CODE_H
#define _HOST_DEVICE_SHARED_CODE_H

#include "HostDeviceSharedMacros.h"

#ifdef HOST_CODE

#include "glm/gtx/compatibility.hpp"

using glm::float2;
using glm::float3;
using glm::float4;

using glm::float2x2;
using glm::float3x3;
using glm::float4x4;

namespace Falcor {
/*******************************************************************
                    CPU declarations
*******************************************************************/
    class Sampler;
    class Texture;

#else
/*******************************************************************
                    HLSL declarations
*******************************************************************/
typedef uint uint32_t;
typedef int int32_t;
#endif


/*******************************************************************
Camera
*******************************************************************/
/**
This is a general host/device structure that describe a camera.
*/
struct CameraData
{
    float4x4 viewMat;               ///< Camera view matrix.
    float4x4 projMat;               ///< Camera projection matrix.
    float4x4 viewProjMat;           ///< Camera view-projection matrix.
    float4x4 invViewProj;           ///< Camera inverse view-projection matrix.
    float4x4 prevViewProjMat;       ///< Camera view-projection matrix associated to previous frame.

    float3   position;                                           ///< Camera world-space position.
    float    focalLength            DEFAULTS(21.0f);            ///< Camera focal length in mm. Default is 59 degree vertical, 90 horizontal FOV at 16:9 aspect ratio.
    float3   up                     DEFAULTS(float3(0, 1, 0));  ///< Camera world-space up vector.
    float    aspectRatio            DEFAULTS(1.7777f);          ///< 16:9 aspect-ratio
    float3   target                 DEFAULTS(float3(0, 0, -1)); ///< Camera target point in world-space.
    float    nearZ                  DEFAULTS(0.1f);             ///< Camera near plane.
    float3   cameraU                DEFAULTS(float3(0, 0, 1));  ///< Camera base vector U. normalized it indicates the left image plane vector. The length is dependent on the FOV. 
    float    farZ                   DEFAULTS(1000.0f);          ///< Camera far plane.
    float3   cameraV                DEFAULTS(float3(0, 1, 0));  ///< Camera base vector V. normalized it indicates the up image plane vector. The length is dependent on the FOV. 
    float    jitterX                DEFAULTS(0.0f);             ///< Eventual camera jitter in the x coordinate
    float3   cameraW                DEFAULTS(float3(1, 0, 0));  ///< Camera base vector U. normalized it indicates the forward direction. The length is the camera focal distance.
    float    jitterY                DEFAULTS(0.0f);             ///< Eventual camera jitter in the y coordinate

    float4x4 rightEyeViewMat;
    float4x4 rightEyeProjMat;
    float4x4 rightEyeViewProjMat;
    float4x4 rightEyePrevViewProjMat;
};

/*******************************************************************
                    Material
*******************************************************************/
struct MaterialTextures
{
    Texture2D diffuse;          // RGB - diffuse color, A - transparency
    Texture2D specular;         // RGB - specular color, A - roughness
    Texture2D emissive;         // RGB - emissive color, A - unused
    Texture2D normalMap;        // 2 or 3 channel normal map, depending on the type
    Texture2D occlusionMap;     // Ambient occlusion map
    Texture2D reflectionMap;    // Reflection map
    Texture2D lightMap;         // Light map
    Texture2D heightMap;        // Height map. Not used by the default material system
};

struct MaterialData
{
    float4 diffuse  DEFAULTS(float4(1));
    float4 specular DEFAULTS(float4(1));
    float3 emissive DEFAULTS(float3(1));
    float padf      DEFAULTS(0);

    float alphaThreshold DEFAULTS(0.5f); // Used in case the alpha mode is mask
    float IoR DEFAULTS(1);               // Index of refraction
    uint32_t id;
    uint32_t flags DEFAULTS(0);

    float2 heightScaleOffset  DEFAULTS(float2(1, 0));
    float2 pad                DEFAULTS(float2(0));

    MaterialTextures textures;
    SamplerState samplerState;
};

/*******************************************************************
                    Lights
*******************************************************************/

/**
    This is a general host/device structure that describe a light source.
*/

struct LightTextures
{
    // Textures for light-probes. #OPTME If the textures was filtered using linear-filtering then the spec and diffuse term is the same texture and we can save a fetch. The default is pre-integration, so that probably doesn't matter much
    Texture2D diffuseProbe2D;
    Texture2D specularProbe2D;
    SamplerState samplerState;
};

struct LightData
{
    float3   posW               DEFAULTS(float3(0, 0, 0));  ///< World-space position of the center of a light source
    uint32_t type               DEFAULTS(LightPoint);       ///< Type of the light source (see above)
    float3   dirW               DEFAULTS(float3(0, -1, 0)); ///< World-space orientation of the light source
    float    openingAngle       DEFAULTS(3.14159265f);      ///< For point (spot) light: Opening angle of a spot light cut-off, pi by default - full-sphere point light
    float3   intensity          DEFAULTS(float3(1, 1, 1));  ///< Emitted radiance of th light source
    float    cosOpeningAngle    DEFAULTS(-1.f);             ///< For point (spot) light: cos(openingAngle), -1 by default because openingAngle is pi by default
    float3   aabbMin            DEFAULTS(float3(1e20f));    ///< For area light: minimum corner of the AABB
    float    penumbraAngle      DEFAULTS(0.f);              ///< For point (spot) light: Opening angle of penumbra region in radians, usually does not exceed openingAngle. 0.f by default, meaning a spot light with hard cut-off
    float3   aabbMax            DEFAULTS(float3(-1e20f));   ///< For area light: maximum corner of the AABB
    float    surfaceArea        DEFAULTS(0.f);              ///< Surface area of the geometry mesh
	float3   tangent            DEFAULTS(float3());         ///< Tangent vector of the geometry mesh
	uint32_t numIndices         DEFAULTS(0);                ///< Number of triangle indices in a polygonal area light
	float3   bitangent          DEFAULTS(float3());         ///< BiTangent vector of the geometry mesh
	float    pad;
    float4x4 transMat           DEFAULTS(float4x4());       ///< Transformation matrix of the model instance for area lights

#ifdef HOST_CODE
    LightTextures textures;
#endif
    // For area light
// 	BufPtr          indexPtr;                                     ///< Buffer id for indices
// 	BufPtr          vertexPtr;                                    ///< Buffer id for vertices
// 	BufPtr          texCoordPtr;                                  ///< Buffer id for texcoord
// 	BufPtr          meshCDFPtr;                                   ///< Pointer to probability distributions of triangle meshes

    /*TODO(tfoley) HACK: Slang can't hanlde this
    // Keep that last
    MaterialData    material;                                     ///< Emissive material of the geometry mesh
    */
};

#ifndef HOST_CODE
// Textures for light-probes. #OPTME If the texure was filtered using linear-filtering then the spec and diffuse term is the same texture and we can save a fetch. The default is pre-integration, so that probably doesn't matter much
Texture2D gDiffuseProbe;        // The diffuse term of the probe (LD). For pre-integrated probes this texture is different then the spec component
Texture2D gSpecularProbe;       // The specular term of the probe (DFG). For pre-integrated probes this texture is different then the spec component
SamplerState gLightSampler;
#endif

/*******************************************************************
                    Shared material routines
*******************************************************************/


/** Converts specular power to roughness. Note there is no "the conversion".
    Reference: http://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html
    \param shininess specular power of an obsolete Phong BSDF
*/
inline float convertShininessToRoughness(float shininess)
{
    return clamp(sqrt(2.0f / (shininess + 2.0f)), 0.f, 1.f);
}

inline float2 convertShininessToRoughness(float2 shininess)
{
    return clamp(sqrt(2.0f / (shininess + 2.0f)), 0.f, 1.f);
}

inline float convertRoughnessToShininess(float a)
{
    return 2.0f / clamp(a*a, 1e-8f, 1.f) - 2.0f;
}

inline float2 convertRoughnessToShininess(float2 a)
{
    return 2.0f / clamp(a*a, 1e-8f, 1.f) - 2.0f;
}

/*******************************************************************
Other helpful shared routines
*******************************************************************/


/** Returns a relative luminance of an input linear RGB color in the ITU-R BT.709 color space
\param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
*/
inline float luminance(float3 rgb)
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

/** Converts color from RGB to YCgCo space
\param RGBColor linear HDR RGB color
*/
inline float3 RGBToYCgCo(float3 rgb)
{
    float Y = dot(rgb, float3(0.25f, 0.50f, 0.25f));
    float Cg = dot(rgb, float3(-0.25f, 0.50f, -0.25f));
    float Co = dot(rgb, float3(0.50f, 0.00f, -0.50f));
    return float3(Y, Cg, Co);
}

/** Converts color from YCgCo to RGB space
\param YCgCoColor linear HDR YCgCo color
*/
inline float3 YCgCoToRGB(float3 YCgCo)
{
    float tmp = YCgCo.x - YCgCo.y;
    float r = tmp + YCgCo.z;
    float g = YCgCo.x + YCgCo.y;
    float b = tmp - YCgCo.z;
    return float3(r, g, b);
}

/** Returns a YUV version of an input linear RGB color in the ITU-R BT.709 color space
\param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
*/
inline float3 RGBToYUV(float3 rgb)
{
    float3 ret;
    ret.x = dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
    ret.y = dot(rgb, float3(-0.09991f, -0.33609f, 0.436f));
    ret.z = dot(rgb, float3(0.615f, -0.55861f, -0.05639f));
    return ret;
}

/** Returns a RGB version of an input linear YUV color in the ITU-R BT.709 color space
\param YUVColor linear HDR YUV color in the ITU-R BT.709 color space
*/
inline float3 YUVToRGB(float3 yuv)
{
    float3 ret;
    ret.x = dot(yuv, float3(1.0f, 0.0f, 1.28033f));
    ret.y = dot(yuv, float3(1.0f, -0.21482f, -0.38059f));
    ret.z = dot(yuv, float3(1.0f, 2.12798f, 0.0f));
    return ret;
}

/** Returns a linear-space RGB version of an input RGB channel value in the ITU-R BT.709 color space
\param sRGBColor sRGB input channel value
*/
inline float sRGBToLinear(float srgb)
{
    if (srgb <= 0.04045f)
    {
        return srgb * (1.0f / 12.92f);
    }
    else
    {
        return pow((srgb + 0.055f) * (1.0f / 1.055f), 2.4f);
    }
}

/** Returns a linear-space RGB version of an input RGB color in the ITU-R BT.709 color space
\param sRGBColor sRGB input color
*/
inline float3 sRGBToLinear(float3 srgb)
{
    return float3(
        sRGBToLinear(srgb.x),
        sRGBToLinear(srgb.y),
        sRGBToLinear(srgb.z));
}

/** Returns a sRGB version of an input linear RGB channel value in the ITU-R BT.709 color space
\param LinearColor linear input channel value
*/
inline float linearToSRGB(float lin)
{
    if (lin <= 0.0031308f)
    {
        return lin * 12.92f;
    }
    else
    {
        return pow(lin, (1.0f / 2.4f)) * (1.055f) - 0.055f;
    }
}

/** Returns a sRGB version of an input linear RGB color in the ITU-R BT.709 color space
\param LinearColor linear input color
*/
inline float3 linearToSRGB(float3 lin)
{
    return float3(
        linearToSRGB(lin.x),
        linearToSRGB(lin.y),
        linearToSRGB(lin.z));
}


/** Returns Michelson contrast given minimum and maximum intensities of an image region
\param iMin minimum intensity of an image region
\param iMax maximum intensity of an image region
*/
inline float computeMichelsonContrast(float iMin, float iMax)
{
    if (iMin == 0.0f && iMax == 0.0f) return 0.0f;
    else return (iMax - iMin) / (iMax + iMin);
}

struct DrawArguments
{
    uint vertexCountPerInstance;
    uint instanceCount;
    uint startVertexLocation;
    uint startInstanceLocation;
};

struct DrawIndexedArguments
{
    uint indexCountPerInstance;
    uint instanceCount;
    uint startIndexLocation;
    int baseVertexLocation;
    uint startInstanceLocation;
};

struct DispatchArguments
{
    uint threadGroupCountX;
    uint threadGroupCountY;
    uint threadGroupCountZ;
};

#ifdef HOST_CODE
#undef SamplerState
#undef Texture2D
} // namespace Falcor
#endif // HOST_CODE

#endif //_HOST_DEVICE_SHARED_CODE_H
