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
#pragma once

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optix_world.h>
#include <cuda_fp16.h>

// OptiX-specific overrides
#define tex2D optix::rtTex2D
#define tex2DLod optix::rtTex2DLod
#define tex2DGrad optix::rtTex2DGrad
#define mat3 optix::Matrix3x3
#define mat4 optix::Matrix4x4
#define ivec2 optix::int2
#define ivec3 optix::int3
#define ivec4 optix::int4
#define uvec2 optix::uint2
#define uvec3 optix::uint3
#define uvec4 optix::uint4

#ifdef _MS_RAY_DIFFERENTIALS
    #define _MS_USER_DERIVATIVES
#endif

#include "HostDeviceData.h"
#include "ShadingUtils/Shading.h"

// Static shading, lights, and camera data
rtDeclareVariable(int,          gInstanceId, , );
rtBuffer<LightData>             gLights;
rtBuffer<CameraData>            gCams;
rtBuffer<int>                   gFrameBuffers;

// Static scene data
rtDeclareVariable(float,        scene_epsilon, , );
rtDeclareVariable(rtObject,     top_object, , );
rtDeclareVariable(float,        gIterationCount, , );

// Static geometry-dependend buffers
struct BindlessInstance
{
    rtBufferId<int3>    indices;
    rtBufferId<vec3>    positions;
    rtBufferId<vec3>    normals;
    rtBufferId<vec3>    tangents;
    rtBufferId<vec3>    bitangents;
    rtBufferId<vec3>    texcoord;
    int                 _pad[2];
    mat4                transform;
    mat4                invTrTransform;
    MaterialData        material;
};
rtDeclareVariable(BindlessInstance,     gInstance, , );
rtDeclareVariable(MaterialData,         gInstMaterial, , );
rtBuffer<int3>                          gInstIndices;
rtBuffer<float3>                        gInstPositions;
rtBuffer<float3>                        gInstNormals;
rtBuffer<float3>                        gInstTangents;
rtBuffer<float3>                        gInstBitangents;
rtBuffer<float3>                        gInstTexcoords;


// Bindless scene objects
rtBuffer<BindlessInstance>      gSceneInstances;

// Static per-launch attributes
rtDeclareVariable(uint2,        launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2,        launch_dim,   rtLaunchDim, );

// Dynamic ray attributes
rtDeclareVariable(optix::Ray,   ray,    rtCurrentRay, );
rtDeclareVariable(float,        t_hit,  rtIntersectionDistance, );

// Dynamic shading attributes
rtDeclareVariable(int,          gPrimOffset, attribute gPrimOffset, ); 
rtDeclareVariable(int3,         gPrimId, attribute gPrimId, ); 
rtDeclareVariable(float2,       gBarys, attribute gBarys, ); 
rtDeclareVariable(float3,       gGeoNormal, attribute gGeoNormal, ); 


// Default per-ray workload
struct PerRayData
{
  float3 result;
  int    depth;
  float  t;
};

////////////////////////////////////////////////////////////////////////////////////////////////
//  Output framebuffer routines
////////////////////////////////////////////////////////////////////////////////////////////////

// For 8 bits/channel sRGB output buffer
static _fn void pack_color(optix::uchar4& ret, const vec4& v)
{
    const float3 c = LinearToSRGB(v3(v));
    ret = optix::make_uchar4( (unsigned char)(__saturatef(c.x)*255.f + .5f), (unsigned char)(__saturatef(c.y)*255.f + .5f), (unsigned char)(__saturatef(c.z)*255.f + .5f), (unsigned char)(__saturatef(v.w)*255.f + .5f));
}

static _fn vec4 unpack_color(const optix::uchar4& c)
{
    const vec3 col = v3( (float)(c.x), (float)(c.y), (float)(c.z)) / 255.f;
    return v4(SRGBToLinear(col), (float)(c.w) / 255.f);
}

// For 16 bits/channel 4x half-precision floating point linear output buffer
static _fn void pack_color(optix::ushort4& ret, const vec4& v)
{
    const half r = __float2half(v.x), g = __float2half(v.y), b = __float2half(v.z), a = __float2half(v.w);
    ret = optix::make_ushort4(*(optix::ushort*)(&r), *(optix::ushort*)(&g), *(optix::ushort*)(&b), *(optix::ushort*)(&a));
}

static _fn vec4 unpack_color(const optix::ushort4 & c)
{
    return v4( __half2float(*(half*)&c.x), __half2float(*(half*)&c.y), __half2float(*(half*)&c.z), __half2float(*(half*)&c.w));
}

// For 16 bits 2x half linear output buffer
static _fn void pack_color(optix::ushort2& ret, const vec2& v)
{
    const half r = __float2half(v.x), g = __float2half(v.y);
    ret = optix::make_ushort2(*(optix::ushort*)(&r), *(optix::ushort*)(&g));
}

static _fn vec4 unpack_color(const optix::ushort2 & c)
{
    return v4( __half2float(*(half*)&c.x), __half2float(*(half*)&c.y), 0.f, 0.f);
}

// For 32 bits/channel single-precision floating point linear output buffer
static _fn void pack_color(vec4& ret, const vec4& v)
{
    ret = v;
}

static _fn vec4 unpack_color(const vec4& c)
{
    return c;
}

// For 32 bits/channel single-precision floating point linear output buffer
static _fn void pack_color(float& ret, const float v)
{
	ret = v;
}

static _fn float unpack_color(const float c)
{
	return c;
}

// For 32 bits/channel unsigned integer output buffer
static _fn void pack_color(int& ret, const int v)
{
	ret = v;
}

static _fn int unpack_color(const int c)
{
	return c;
}

template<int idx, class T, class T2>
static _fn T2 fbLoad()
{
    rtBufferId<T, 2> buf(gFrameBuffers[idx]);
    return unpack_color(buf[launch_index]);   
}

template<int idx, class T>
static _fn T fbLoad()
{
	rtBufferId<T, 2> buf(gFrameBuffers[idx]);
	return unpack_color(buf[launch_index]);
}

template<int idx, class T, class T2>
static _fn void fbStore(const uint2& index, const T2& v)
{
	rtBufferId<T, 2> buf(gFrameBuffers[idx]);
	pack_color(buf[index], v);
}

template<int idx, class T, class T2>
static _fn void fbStore(const T2& v)
{
    fbStore<idx, T, T2>(launch_index, v);
}

// Moving average into frame buffer

static _fn void accumulate(const vec4& v)
{
    fbStore<0, vec4>((fbLoad<0, vec4>() * gIterationCount + v) / (gIterationCount + 1.f));
}

static _fn void accumulate(const vec3& v)
{
    accumulate(v4(v, 1.f));
}

////////////////////////////////////////////////////////////////////////////////////////////////
//  Shading and sampling routines
////////////////////////////////////////////////////////////////////////////////////////////////

template <class Buffer_t>
_fn auto fetchAndInterpolate(Buffer_t& d, const int3& face, vec2 barys) -> std::remove_reference<decltype(d[0])>::type
{
    return d[face.x] * (1.0f - barys.x - barys.y) + d[face.y] * barys.x + d[face.z] * barys.y;
}

template <class Buffer_t>
_fn auto fetchRayDifferentialUV(const int3& idx, Buffer_t& texcoord, const vec3& p0, const vec3& p1, const vec3& p2, const vec3& origin, const vec3& dir) -> std::remove_reference<decltype(texcoord[0])>::type
{
    vec2 bary;
    vec3 n; float t;
    optix::Ray diff = optix::Ray(origin, dir, 0, 0.0f, RT_DEFAULT_MAX);
    intersect_triangle(diff, p0, p1, p2, n, t, bary.x, bary.y);
    return fetchAndInterpolate(texcoord, idx, bary);
}

template <class Buffer_t>
_fn void interpolateAndPrepareShadingAttribs(const int3 & idx, const vec2& barys, 
    Buffer_t& positions, Buffer_t& normals, Buffer_t& tangents, Buffer_t& bitangents, Buffer_t& texcoord, 
    const MaterialData& material, const vec3 prevPos, _ref(ShadingAttribs) shAttr
#ifdef _MS_RAY_DIFFERENTIALS
    , const vec3& diff_x_origin, const vec3& diff_x_dir, const vec3& diff_y_origin, const vec3& diff_y_dir
#endif
     )
{
    // First fetch and interpolate initial shading attributes
    const vec3 position    = fetchAndInterpolate(positions, idx, barys);
    const vec3 normal      = fetchAndInterpolate(normals, idx, barys);
    const vec3 tangent     = fetchAndInterpolate(tangents, idx, barys);
    const vec3 bitangent   = fetchAndInterpolate(bitangents, idx, barys);
    const vec2 uv          = v2(fetchAndInterpolate(texcoord, idx, barys));

#ifdef _MS_RAY_DIFFERENTIALS
    const float3 p0 = positions[idx.x];
    const float3 p1 = positions[idx.y];
    const float3 p2 = positions[idx.z];
    vec2 uv_x = v2(fetchRayDifferentialUV(idx, texcoord, p0, p1, p2, diff_x_origin, diff_x_dir));
    vec2 uv_y = v2(fetchRayDifferentialUV(idx, texcoord, p0, p1, p2, diff_y_origin, diff_y_dir));
    vec2 du = uv_x - uv;
    vec2 dv = uv_y - uv;
#elif defined(_MS_USER_DERIVATIVES)
    vec2 du = shAttr.DPDX;
    vec2 dv = shAttr.DPDY;
#endif

    // Then prepare colors and fetch textures
    prepareShadingAttribs(material, position, prevPos, normal, tangent, bitangent, uv, 
#ifdef _MS_USER_DERIVATIVES
        du, dv,    
#endif
        shAttr);
}

_fn void interpolateAndPrepareShadingAttribs(_ref(ShadingAttribs) shAttr, 
#ifdef _MS_RAY_DIFFERENTIALS
    const vec3 & diff_x_origin, const vec3 & diff_x_dir, const vec3 & diff_y_origin, const vec3 & diff_y_dir,
#endif
    const bool ffNormal = false )
{
    interpolateAndPrepareShadingAttribs(gPrimId, gBarys, 
        gInstPositions, gInstNormals, gInstTangents, gInstBitangents, gInstTexcoords,
        gInstMaterial, ray.origin, shAttr
#ifdef _MS_RAY_DIFFERENTIALS
     , diff_x_origin, diff_x_dir, diff_y_origin,  diff_y_dir
#endif
     );
    
    // Apply transformation
    shAttr.P = rtTransformPoint(RT_OBJECT_TO_WORLD,  shAttr.P);
    shAttr.N = rtTransformNormal(RT_OBJECT_TO_WORLD, shAttr.N);
    shAttr.T = rtTransformNormal(RT_OBJECT_TO_WORLD, shAttr.T);
    shAttr.B = rtTransformNormal(RT_OBJECT_TO_WORLD, shAttr.B);

    // Recompute the new camera direction
    shAttr.E = normalize(ray.origin - shAttr.P);

    // Flip shading normal frame
    if(ffNormal)
    {
        vec3 world_geometric_normal = rtTransformNormal(RT_OBJECT_TO_WORLD, gGeoNormal);
        shAttr.N = optix::faceforward(shAttr.N, -ray.direction, world_geometric_normal);
    }
}
