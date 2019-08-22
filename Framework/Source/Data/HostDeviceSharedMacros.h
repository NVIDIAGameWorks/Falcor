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
#ifndef _HOST_DEVICE_SHARED_MACROS_H
#define _HOST_DEVICE_SHARED_MACROS_H

/*******************************************************************
                    Common structures & routines
*******************************************************************/

#define MAX_INSTANCES 64    ///< Max supported instances per draw call
#define MAX_BONES 256       ///< Max supported bones per model

/*******************************************************************
                    Glue code for CPU/GPU compilation
*******************************************************************/

#if (defined(__STDC_HOSTED__) || defined(__cplusplus))   // we're in C-compliant compiler, probably host
#define HOST_CODE 1
#else
#define FALCOR_SHADER_CODE
#endif


#ifdef HOST_CODE

/*******************************************************************
                    CPU declarations
*******************************************************************/
#define DEFAULTS(x_) = x_
#define SamplerState std::shared_ptr<Sampler>
#define Texture2D std::shared_ptr<Texture>
#else
/*******************************************************************
                    HLSL declarations
*******************************************************************/
#define inline 
#define DEFAULTS(x_)
#endif

/*******************************************************************
    Materials
*******************************************************************/

// Shading model
#define ShadingModelMetalRough 0
//#define ShadingModelMetalAnisoRough 1 Reserved for future use
#define ShadingModelSpecGloss 2

// Channel type
#define ChannelTypeUnused    0
#define ChannelTypeConst     1
#define ChannelTypeTexture   2

// Normal map type
#define NormalMapUnused     0
#define NormalMapRGB        1
#define NormalMapRG         2

// Alpha mode
#define AlphaModeOpaque      0
#define AlphaModeMask        1

// Bit count
#define SHADING_MODEL_BITS   (3)
#define DIFFUSE_TYPE_BITS    (3)
#define SPECULAR_TYPE_BITS   (3)
#define EMISSIVE_TYPE_BITS   (3)
#define NORMAL_MAP_BITS      (2)
#define OCCLUSION_MAP_BITS   (1)
#define LIGHT_MAP_BITS       (1)
#define HEIGHT_MAP_BITS      (1)
#define ALPHA_MODE_BITS      (2)
#define DOUBLE_SIDED_BITS    (1)

// Offsets
#define SHADING_MODEL_OFFSET (0)
#define DIFFUSE_TYPE_OFFSET  (SHADING_MODEL_OFFSET + SHADING_MODEL_BITS)
#define SPECULAR_TYPE_OFFSET (DIFFUSE_TYPE_OFFSET  + DIFFUSE_TYPE_BITS)
#define EMISSIVE_TYPE_OFFSET (SPECULAR_TYPE_OFFSET + SPECULAR_TYPE_BITS)
#define NORMAL_MAP_OFFSET    (EMISSIVE_TYPE_OFFSET + EMISSIVE_TYPE_BITS)
#define OCCLUSION_MAP_OFFSET (NORMAL_MAP_OFFSET    + NORMAL_MAP_BITS)
#define LIGHT_MAP_OFFSET     (OCCLUSION_MAP_OFFSET + OCCLUSION_MAP_BITS)
#define HEIGHT_MAP_OFFSET    (LIGHT_MAP_OFFSET     + LIGHT_MAP_BITS)
#define ALPHA_MODE_OFFSET    (HEIGHT_MAP_OFFSET    + HEIGHT_MAP_BITS)
#define DOUBLE_SIDED_OFFSET  (ALPHA_MODE_OFFSET    + ALPHA_MODE_BITS)

// Extract bits
#define EXTRACT_BITS(bits, offset, value) ((value >> offset) & ((1 << bits) - 1))
#define EXTRACT_SHADING_MODEL(value)    EXTRACT_BITS(SHADING_MODEL_BITS,    SHADING_MODEL_OFFSET,   value)
#define EXTRACT_DIFFUSE_TYPE(value)     EXTRACT_BITS(DIFFUSE_TYPE_BITS,     DIFFUSE_TYPE_OFFSET,    value)
#define EXTRACT_SPECULAR_TYPE(value)    EXTRACT_BITS(SPECULAR_TYPE_BITS,    SPECULAR_TYPE_OFFSET,   value)
#define EXTRACT_EMISSIVE_TYPE(value)    EXTRACT_BITS(EMISSIVE_TYPE_BITS,    EMISSIVE_TYPE_OFFSET,   value)
#define EXTRACT_NORMAL_MAP_TYPE(value)  EXTRACT_BITS(NORMAL_MAP_BITS,       NORMAL_MAP_OFFSET,      value)
#define EXTRACT_OCCLUSION_MAP(value)    EXTRACT_BITS(OCCLUSION_MAP_BITS,    OCCLUSION_MAP_OFFSET,   value)
#define EXTRACT_LIGHT_MAP(value)        EXTRACT_BITS(LIGHT_MAP_BITS,        LIGHT_MAP_OFFSET,       value)  
#define EXTRACT_HEIGHT_MAP(value)       EXTRACT_BITS(HEIGHT_MAP_BITS,       HEIGHT_MAP_OFFSET,      value)
#define EXTRACT_ALPHA_MODE(value)       EXTRACT_BITS(ALPHA_MODE_BITS,       ALPHA_MODE_OFFSET,      value)
#define EXTRACT_DOUBLE_SIDED(value)     EXTRACT_BITS(DOUBLE_SIDED_BITS,     DOUBLE_SIDED_OFFSET,    value)

// Pack bits
#define PACK_BITS(bits, offset, flags, value) (((value & ((1 << bits) - 1)) << offset) | (flags & (~(((1 << bits) - 1) << offset))))
#define PACK_SHADING_MODEL(flags, value)    PACK_BITS(SHADING_MODEL_BITS,    SHADING_MODEL_OFFSET,   flags, value)
#define PACK_DIFFUSE_TYPE(flags, value)     PACK_BITS(DIFFUSE_TYPE_BITS,     DIFFUSE_TYPE_OFFSET,    flags, value)
#define PACK_SPECULAR_TYPE(flags, value)    PACK_BITS(SPECULAR_TYPE_BITS,    SPECULAR_TYPE_OFFSET,   flags, value)
#define PACK_EMISSIVE_TYPE(flags, value)    PACK_BITS(EMISSIVE_TYPE_BITS,    EMISSIVE_TYPE_OFFSET,   flags, value)
#define PACK_NORMAL_MAP_TYPE(flags, value)  PACK_BITS(NORMAL_MAP_BITS,       NORMAL_MAP_OFFSET,      flags, value)
#define PACK_OCCLUSION_MAP(flags, value)    PACK_BITS(OCCLUSION_MAP_BITS,    OCCLUSION_MAP_OFFSET,   flags, value)
#define PACK_LIGHT_MAP(flags, value)        PACK_BITS(LIGHT_MAP_BITS,        LIGHT_MAP_OFFSET,       flags, value)
#define PACK_HEIGHT_MAP(flags, value)       PACK_BITS(HEIGHT_MAP_BITS,       HEIGHT_MAP_OFFSET,      flags, value)
#define PACK_ALPHA_MODE(flags, value)       PACK_BITS(ALPHA_MODE_BITS,       ALPHA_MODE_OFFSET,      flags, value)
#define PACK_DOUBLE_SIDED(flags, value)     PACK_BITS(DOUBLE_SIDED_BITS,     DOUBLE_SIDED_OFFSET,    flags, value)

/*******************************************************************
                    Lights
*******************************************************************/

/**
    Types of light sources. Used in LightData structure.
*/
#define LightPoint                  0    ///< Point light source, can be a spot light if its opening angle is < 2pi
#define LightDirectional            1    ///< Directional light source
#define LightArea                   2    ///< Area light source, potentially with arbitrary geometry
#define LightAreaRect               3    ///< Quad shaped area light source
#define LightAreaSphere             4    ///< Spherical area light source
#define LightAreaDisc               5    ///< Disc shaped area light source

#define MAX_LIGHT_SOURCES 16

// To bind area lights, use this macro to declare the constant buffer in your shader
#define AREA_LIGHTS(n) shared cbuffer InternalAreaLightCB \
{ \
    AreaLightData gAreaLights[n]; \
};

/** Light probe types
*/
#define LightProbeLinear2D          0    ///< Light probe filtered with linear-filtering, 2D texture
#define LightProbePreIntegrated2D   1    ///< Pre-integrated light probe, 2D texture
#define LightProbeLinearCube        2    ///< Light probe filtered with linear-filtering, texture-cube
#define LightProbePreIntegratedCube 3    ///< Pre-integrated light probe, texture-cube

/*******************************************************************
                Math
*******************************************************************/
#define M_PI     3.14159265358979323846
#define M_PI2    6.28318530717958647692
#define M_INV_PI 0.3183098861837906715

#endif //_HOST_DEVICE_SHARED_MACROS_H

