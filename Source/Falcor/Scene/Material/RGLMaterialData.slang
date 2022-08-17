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

#ifdef HOST_CODE
#include "Scene/Material/TextureHandle.slang"
#else
__exported import Scene.Material.TextureHandle;
#endif

BEGIN_NAMESPACE_FALCOR

/** This is a host/device structure that describes a measured MERL material.
*/
struct RGLMaterialData
{
    // MaterialHeader (8B) is stored just before this struct in memory.
    uint phiSize = 0;               ///< Number of angular sampling points.
    uint thetaSize = 0;
    uint2 sigmaSize = uint2(0);     ///< Size of sigma/NDF lookup tables. These are 2D.
    uint2 ndfSize = uint2(0);
    uint2 vndfSize = uint2(0);      ///< Size of VNDF/luminance lookup tables. These are 4D (phi/theta x vndf/lumi size).
    uint2 lumiSize = uint2(0);
    uint thetaBufID = 0;            ///< Inclinations that were sampled by the goniophotometer.
    uint   phiBufID = 0;            ///< Azimuths that were sampled by the goniophotometer. This has size 1 for isotropic materials.
    uint sigmaBufID = 0;            ///< Projected microfacet area.
    uint   ndfBufID = 0;            ///< Best matching normal distribution function (NDF) from retroreflective measurements.
    uint  vndfBufID = 0;            ///< Corresponding distribution of visible normals (VNDF).
    uint  lumiBufID = 0;            ///< Luminance of RGB buffer, normalized (so that it is a PDF).
    uint   rgbBufID = 0;            ///< RGB measurements in VNDF space.
    uint vndfMarginalBufID = 0;     ///< Marginal and conditional CDFs of VNDF/luminance to make them samplable.
    uint lumiMarginalBufID = 0;
    uint vndfConditionalBufID = 0;
    uint lumiConditionalBufID = 0;
    uint samplerID = 0;             ///< Texture sampler ID for LUT sampler.
    TextureHandle texAlbedoLUT = {};///< Texture handle for albedo LUT.

    // Maximum resolution allowed for the various lookup tables.
    static const uint kMaxResolution = 0xffff;

    static const uint kAlbedoLUTSize = 256;
};

END_NAMESPACE_FALCOR
