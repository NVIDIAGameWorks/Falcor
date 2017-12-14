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
#define MAX_BONES 128       ///< Max supported bones per model

/*******************************************************************
                    Glue code for CPU/GPU compilation
*******************************************************************/

#if (defined(__STDC_HOSTED__) || defined(__cplusplus)) && !defined(__CUDACC__)    // we're in C-compliant compiler, probably host
#    define HOST_CODE 1
#else
#   define HLSL_CODE
#define FALCOR_SHADER_CODE
#endif

#ifdef HLSL_CODE
//#extension GL_NV_shader_buffer_load : enable
#endif

#ifdef HOST_CODE

/*******************************************************************
                    CPU declarations
*******************************************************************/
#define loop_unroll
#define v2 vec2
#define v3 vec3
#define v4 vec4
#define _fn
#define DEFAULTS(x_) = x_
#define SamplerState std::shared_ptr<Sampler>
#define Texture2D std::shared_ptr<Texture>
#else
/*******************************************************************
                    HLSL declarations
*******************************************************************/
#define loop_unroll [unroll]
#define _fn 
#define __device__ 
#define inline 
#define _ref(__x) inout __x
#define DEFAULTS(x_)
#endif

/*******************************************************************
                    Lights
*******************************************************************/

/**
    Types of light sources. Used in LightData structure.
*/
#define LightPoint           0    ///< Point light source, can be a spot light if its opening angle is < 2pi
#define LightDirectional     1    ///< Directional light source
#define LightArea            2    ///< Area light source, potentially with arbitrary geometry
//#define LightVolume        3    ///< Volumetric light source

#define MAX_LIGHT_SOURCES 16

/*******************************************************************
                    Material
*******************************************************************/

/** Type of the material layer:
    Diffuse (Lambert model, can be Oren-Nayar if roughness is not 1),
    Reflective material (conductor),
    Refractive material (dielectric)
*/
#define     MatNone            0            ///< A "null" material. Used to end the list of layers
#define     MatLambert         1            ///< A simple diffuse Lambertian BRDF layer
#define     MatConductor       2            ///< A conductor material, metallic reflection, no refraction nor subscattering
#define     MatDielectric      3            ///< A refractive dielectric material, if applied on top of others acts like a coating
#define     MatEmissive        4            ///< An emissive material. Can be assigned to a geometry to create geometric a light source (will be supported only with ray tracing)
#define     MatUser            5            ///< User-defined material, should be parsed and processed by user
#define     MatNumTypes        (MatUser+1)  ///< Number of material types

/** Type of used Normal Distribution Function (NDF). Options so far
    Beckmann distribution (original Blinn-Phong)
    GGX distribution (smoother highlight, better fit for some materials, default)
*/
#define     NDFBeckmann        0    ///< Beckmann distribution for NDF
#define     NDFGGX             1    ///< GGX distribution for NDF
#define     NDFUser            2    ///< User-defined distribution for NDF, should be processed by user

#define     BlendFresnel       0    ///< Material layer is blended according to Fresnel
#define     BlendConstant      1    ///< Material layer is blended according to a constant factor stored in w component of constant color
#define     BlendAdd           2    ///< Material layer is added to the previous layers

/**
    This number specifies a maximum possible number of layers in a material.
    There seems to be a good trade-off between performance and flexibility.
    With three layers, we can represent e.g. a base conductor material with diffuse component, coated with a dielectric.
    If this number is changed, the scene serializer should make sure the new number of layers is saved/loaded correctly.
*/
#define     MatMaxLayers    3

#define ROUGHNESS_CHANNEL_BIT 2

#endif //_HOST_DEVICE_SHARED_MACROS_H
