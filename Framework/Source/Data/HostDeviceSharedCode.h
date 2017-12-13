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
    float4x4            viewMat                DEFAULTS(float4x4());                ///< Camera view matrix.
    float4x4            projMat                DEFAULTS(float4x4());                ///< Camera projection matrix.
    float4x4            viewProjMat            DEFAULTS(float4x4());                ///< Camera view-projection matrix.
    float4x4            invViewProj            DEFAULTS(float4x4());                ///< Camera inverse view-projection matrix.
    float4x4            prevViewProjMat        DEFAULTS(float4x4());                ///< Camera view-projection matrix associated to previous frame.

    float3            position               DEFAULTS(float3(0, 0, 0));         ///< Camera world-space position.
    float           focalLength            DEFAULTS(21.0f);                 ///< Camera focal length in mm. Default is 59 degree vertical, 90 horizontal FOV at 16:9 aspect ratio.
    float3            up                     DEFAULTS(float3(0, 1, 0));         ///< Camera world-space up vector.
    float           aspectRatio            DEFAULTS(1.f);                   ///< Camera aspect ratio.
    float3            target                 DEFAULTS(float3(0, 0, -1));        ///< Camera target point in world-space.
    float           nearZ                  DEFAULTS(0.1f);                  ///< Camera near plane.
    float3            cameraU                DEFAULTS(float3(0, 0, 1));         ///< Camera base vector U. normalized it indicates the left image plane vector. The length is dependent on the FOV. 
    float           farZ                   DEFAULTS(10000.0f);              ///< Camera far plane.
    float3            cameraV                DEFAULTS(float3(0, 1, 0));         ///< Camera base vector V. normalized it indicates the up image plane vector. The length is dependent on the FOV. 
    float           jitterX                DEFAULTS(0.0f);                  ///< Eventual camera jitter in the x coordinate
    float3            cameraW                DEFAULTS(float3(1, 0, 0));         ///< Camera base vector U. normalized it indicates the forward direction. The length is the camera focal distance.
    float           jitterY                DEFAULTS(0.0f);                  ///< Eventual camera jitter in the y coordinate

    float4x4            rightEyeViewMat;
    float4x4            rightEyeProjMat;
    float4x4            rightEyeViewProjMat;
    float4x4            rightEyePrevViewProjMat;
};

/*******************************************************************
                    Material
*******************************************************************/

/**
    A description for a single material layer.
    Contains information about underlying BRDF, NDF, and rules for blending with other layers.
    Also contains material properties, such as albedo and roughness.
*/
struct MaterialLayerDesc
{
    uint32_t    type            DEFAULTS(MatNone);             ///< Specifies a material Type: diffuse/conductor/dielectric/etc. None means there is no material
    uint32_t    ndf             DEFAULTS(NDFGGX);              ///< Specifies a model for normal distribution function (NDF): Beckmann, GGX, etc.
    uint32_t    blending        DEFAULTS(BlendAdd);            ///< Specifies how this layer should be combined with previous layers. E.g., blended based on Fresnel (useful for dielectric coatings), or just added on top, etc.
    uint32_t    hasTexture      DEFAULTS(0);                   ///< Specifies whether or not the material has textures. For dielectric and conductor layers this has special meaning - if the 2nd bit is on, it means we have roughness channel
};

struct MaterialLayerValues
{
    float4     albedo;                                       ///< Material albedo/specular color/emitted color
    float4     roughness;                                    ///< Material roughness parameter [0;1] for NDF
    float4     extraParam;                                   ///< Additional user parameter, can be IoR for conductor and dielectric
    float3     pad             DEFAULTS(float3(0, 0, 0));
    float    pmf             DEFAULTS(0.f);                 ///< Specifies the current value of the PMF of all layers. E.g., first layer just contains a probability of being selected, others accumulate further
};

/**
    The auxiliary structure that provides the first occurrence of the layer by its type.
*/
struct LayerIdxByType
{
    float3 pad;               // This is here due to HLSL alignment rules
    int32_t id DEFAULTS(-1);
};

/**
    The main material description structure. Contains a dense list of layers. The layers are stored from inner to outer, ending with a MatNone layer.
    Besides, the material contains its scene-unique id, as well as various modifiers, like normal/displacement map and alpha test map.
*/
struct MaterialDesc
{
    MaterialLayerDesc   layers[MatMaxLayers];     // First one is a terminal layer, usually either opaque with coating, or dielectric; others are optional layers, usually a transparent dielectric coating layer or a mixture with conductor
    uint32_t            hasAlphaMap     DEFAULTS(0);
    uint32_t            hasNormalMap    DEFAULTS(0);
    uint32_t            hasHeightMap    DEFAULTS(0);
    uint32_t            hasAmbientMap   DEFAULTS(0);
    LayerIdxByType      layerIdByType[MatNumTypes];             ///< Provides a layer idx by its type, if there is no layer of this type, the idx is -1
};

struct MaterialValues
{
    MaterialLayerValues layers[MatMaxLayers];
    float2  height;                        // Height (displacement) map modifier (scale, offset). If texture is non-null, one can apply a displacement or parallax mapping
    float alphaThreshold DEFAULTS(1.0f); // Alpha test threshold, in cast alpha-test is enabled (alphaMap is not nullptr)
    int32_t id           DEFAULTS(-1);   // Scene-unique material id, -1 is a wrong material
};

struct MaterialTextures
{
    Texture2D layers[MatMaxLayers];        // A single texture per layer
    Texture2D alphaMap;         // Alpha test parameter, if texture is non-null, alpha test is enabled, alpha threshold is stored in the constant color
    Texture2D normalMap;        // Normal map modifier, if texture is non-null, shading normal is perturbed
    Texture2D heightMap;        // Height (displacement) map modifier, if texture is non-null, one can apply a displacement or parallax mapping
    Texture2D ambientMap;       // Ambient occlusion map
};

struct MaterialData
{
    MaterialDesc desc;
    MaterialValues values;
    MaterialTextures textures;
    SamplerState samplerState;  // The sampler state to use when sampling the object
};

struct PreparedMaterialData
{
    MaterialDesc    desc;
    MaterialValues  values;
};

/**
    The structure stores the complete information about the shading point,
    except for a light source information.
    It stores pre-evaluated material parameters with pre-fetched textures,
    shading point position, normal, viewing direction etc.
*/
struct ShadingAttribs
{
    float3    P;                                  ///< Shading hit position in world space
    float3    E;                                  ///< Direction to the eye at shading hit
    float3    N;                                  ///< Shading normal at shading hit
    float3    T;                                  ///< Shading tangent at shading hit
    float3    B;                                  ///< Shading bitangent at shading hit
    float2    UV;                                 ///< Texture mapping coordinates

#ifdef _MS_USER_DERIVATIVES
    float2    DPDX            DEFAULTS(float2(0, 0));                                  
    float2    DPDY            DEFAULTS(float2(0, 0)); ///< User-provided 2x2 full matrix of duv/dxy derivatives of a shading point footprint in texture space
#else
    float   lodBias         DEFAULTS(0);        ///< LOD bias to use when sampling textures
#endif

#ifdef _MS_USER_HALF_VECTOR_DERIVATIVES
    float2    DHDX            DEFAULTS(float2(0, 0));
    float2    DHDY            DEFAULTS(float2(0, 0));  ///< User-defined half-vector derivatives
#endif
    PreparedMaterialData preparedMat;               ///< Copy of the original material with evaluated parameters (i.e., textures are fetched etc.)
    float aoFactor;
};

/*******************************************************************
                    Lights
*******************************************************************/

/**
    This is a general host/device structure that describe a light source.
*/
struct LightData
{
    float3            worldPos           DEFAULTS(float3(0, 0, 0));     ///< World-space position of the center of a light source
    uint32_t        type               DEFAULTS(LightPoint);      ///< Type of the light source (see above)
    float3            worldDir           DEFAULTS(float3(0, -1, 0));    ///< World-space orientation of the light source
    float           openingAngle       DEFAULTS(3.14159265f);     ///< For point (spot) light: Opening angle of a spot light cut-off, pi by default - full-sphere point light
    float3            intensity          DEFAULTS(float3(1, 1, 1));     ///< Emitted radiance of th light source
    float           cosOpeningAngle    DEFAULTS(-1.f);            ///< For point (spot) light: cos(openingAngle), -1 by default because openingAngle is pi by default
    float3            aabbMin            DEFAULTS(float3(1e20f));       ///< For area light: minimum corner of the AABB
    float           penumbraAngle      DEFAULTS(0.f);             ///< For point (spot) light: Opening angle of penumbra region in radians, usually does not exceed openingAngle. 0.f by default, meaning a spot light with hard cut-off
    float3            aabbMax            DEFAULTS(float3(-1e20f));      ///< For area light: maximum corner of the AABB
    float           surfaceArea        DEFAULTS(0.f);             ///< Surface area of the geometry mesh
	float3            tangent            DEFAULTS(float3());          ///< Tangent vector of the geometry mesh
	uint32_t        numIndices         DEFAULTS(0);               ///< Number of triangle indices in a polygonal area light
	float3            bitangent          DEFAULTS(float3());          ///< BiTangent vector of the geometry mesh
	float           pad;
    float4x4            transMat           DEFAULTS(float4x4());          ///< Transformation matrix of the model instance for area lights

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

/*******************************************************************
                    Shared material routines
*******************************************************************/


/** Converts specular power to roughness. Note there is no "the conversion".
    Reference: http://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html
    \param shininess specular power of an obsolete Phong BSDF
*/
inline float _fn convertShininessToRoughness(const float shininess)
{
    return clamp(sqrt(2.0f / (shininess + 2.0f)), 0.f, 1.f);
}

inline float2 _fn convertShininessToRoughness(const float2 shininess)
{
    return clamp(sqrt(2.0f / (shininess + 2.0f)), 0.f, 1.f);
}

inline float _fn convertRoughnessToShininess(const float a)
{
    return 2.0f / clamp(a*a, 1e-8f, 1.f) - 2.0f;
}

inline float2 _fn convertRoughnessToShininess(const float2 a)
{
    return 2.0f / clamp(a*a, 1e-8f, 1.f) - 2.0f;
}

/*******************************************************************
Other helpful shared routines
*******************************************************************/


/** Returns a relative luminance of an input linear RGB color in the ITU-R BT.709 color space
\param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
*/
inline float _fn luminance(const float3 rgb)
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

/** Converts color from RGB to YCgCo space
\param RGBColor linear HDR RGB color
*/
inline float3 _fn RGBToYCgCo(const float3 rgb)
{
    const float Y = dot(rgb, float3(0.25f, 0.50f, 0.25f));
    const float Cg = dot(rgb, float3(-0.25f, 0.50f, -0.25f));
    const float Co = dot(rgb, float3(0.50f, 0.00f, -0.50f));

    return float3(Y, Cg, Co);
}

/** Converts color from YCgCo to RGB space
\param YCgCoColor linear HDR YCgCo color
*/
inline float3 _fn YCgCoToRGB(const float3 YCgCo)
{
    const float tmp = YCgCo.x - YCgCo.y;
    const float r = tmp + YCgCo.z;
    const float g = YCgCo.x + YCgCo.y;
    const float b = tmp - YCgCo.z;

    return float3(r, g, b);
}

/** Returns a YUV version of an input linear RGB color in the ITU-R BT.709 color space
\param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
*/
inline float3 _fn RGBToYUV(const float3 rgb)
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
inline float3 _fn YUVToRGB(const float3 yuv)
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
inline float _fn SRGBToLinear(const float srgb)
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
inline float3 _fn SRGBToLinear(const float3 srgb)
{
    return float3(
        SRGBToLinear(srgb.x),
        SRGBToLinear(srgb.y),
        SRGBToLinear(srgb.z));
}

/** Returns a sRGB version of an input linear RGB channel value in the ITU-R BT.709 color space
\param LinearColor linear input channel value
*/
inline float _fn LinearToSRGB(const float lin)
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
inline float3 _fn LinearToSRGB(const float3 lin)
{
    return float3(
        LinearToSRGB(lin.x),
        LinearToSRGB(lin.y),
        LinearToSRGB(lin.z));
}


/** Returns Michelson contrast given minimum and maximum intensities of an image region
\param Imin minimum intensity of an image region
\param Imax maximum intensity of an image region
*/
inline float _fn computeMichelsonContrast(const float iMin, const float iMax)
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
static_assert((sizeof(MaterialValues) % sizeof(float4)) == 0, "MaterialValue has a wrong size");
static_assert((sizeof(MaterialLayerDesc) % sizeof(float4)) == 0, "MaterialLayerDesc has a wrong size");
static_assert((sizeof(MaterialLayerValues) % sizeof(float4)) == 0, "MaterialLayerValues has a wrong size");
static_assert((sizeof(MaterialDesc) % sizeof(float4)) == 0, "MaterialDesc has a wrong size");
static_assert((sizeof(MaterialValues) % sizeof(float4)) == 0, "MaterialValues has a wrong size");
static_assert((sizeof(MaterialData) % sizeof(float4)) == 0, "MaterialData has a wrong size");
#undef SamplerState
#undef Texture2D
} // namespace Falcor
#endif // HOST_CODE

#endif //_HOST_DEVICE_SHARED_CODE_H
