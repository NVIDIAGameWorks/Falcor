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
#include "Utils/Math/MathConstants.slangh"

import Scene.Shading;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Rendering.Materials.RGLCommon;
import Utils.Sampling.UniformSampleGenerator;

struct RGLAcquisition
{
    uint materialID;
    uint2 sigmaIntegrationGrid;
    uint phiSize;
    uint thetaSize;
    uint2 ndfSize;
    uint4 vndfSize;
    uint4 lumiSize;
    
    RWStructuredBuffer<float3> ndfDirections;  // Size ndfSize.x * ndfSize.y
    RWStructuredBuffer<float> retroReflection; // Size ndfSize.x * ndfSize.y
    RWStructuredBuffer<float> ndfKernel;       // Size (ndfSize.x * ndfSize.y) ^ 2
    RWStructuredBuffer<float> ndf;             // Size ndfSize.x * ndfSize.y
    RWStructuredBuffer<float> ndfTmp;          // Size ndfSize.x * ndfSize.y
    RWStructuredBuffer<float> sigma;           // Size ndfSize.x * ndfSize.y
    RWStructuredBuffer<float> thetas;          // Size thetaSize
    RWStructuredBuffer<float> phis;            // Size phiSize
    RWStructuredBuffer<float> vndf;            // Size phiSize * thetaSize * vndfSize.z * vndfSize.w (vndfSize.xy == phiSize, thetaSize)
    
    ByteAddressBuffer vndfBuf;                 // Size phiSize * thetaSize * vndfSize.z * vndfSize.w
    ByteAddressBuffer vndfMarginalBuf;         // Size phiSize * thetaSize * vndfSize.z * vndfSize.w
    ByteAddressBuffer vndfConditionalBuf;      // Size phiSize * thetaSize * vndfSize.z * vndfSize.w
    
    RWBuffer<float> lumi; // Size phiSize * thetaSize * lumiSize.z * lumiSize.w (lumiSize.xy == phiSize, thetaSize)
    RWBuffer<float> rgb;  // Size phiSize * thetaSize * lumiSize.z * lumiSize.w

    static const uint2 kSampleCount = {8, 8};
    
    ShadingData buildShadingData(float3 wi)
    {
        ShadingData sd = {};
        sd.T = float3(1.0f, 0.0f, 0.0f);
        sd.B = float3(0.0f, 1.0f, 0.0f);
        sd.N = float3(0.0f, 0.0f, 1.0f);
        sd.faceN = sd.N;
        sd.frontFacing = true;

        sd.mtl = gScene.materials.getMaterialHeader(materialID);
        sd.materialID = materialID;
        sd.opacity = 1.0f;
        sd.IoR = 1.0f;
        sd.V = wi;
        
        return sd;
    }

    // Clamp directions to lie above the cutoff imposed by the BRDF interface.
    // Otherwise we measure zero reflectance at glancing angles.
    float3 clampDirection(float3 x)
    {
        return normalize(float3(x.xy, max(x.z, kMinCosTheta * 2.0f)));
    }
    
    void measureRetroreflection(const uint2 threadID)
    {
        float2 unitI = float2(threadID) / float2(ndfSize - 1);
        float3 wi = clampDirection(fromSpherical(unitToSpherical(unitI)));
        
        ShadingData sd = buildShadingData(wi);
        
        let lod = ExplicitLodTextureSampler(0.f);
        let bsdf = gScene.materials.getBSDF(sd, lod);
        
        // TODO: We should probably take multiple samples if eval is actually probabilistic.
        // No way currently to find out if eval is deterministic or not.
        UniformSampleGenerator sg = UniformSampleGenerator(threadID, 0);
        // The paper measures at lambda=532nm. Just grab the green channel to approximate this.
        float retro = bsdf.eval(sd, wi, sg).g;
        
        uint idx = threadID.x + threadID.y * ndfSize.x;
        ndfDirections[idx] = wi;
        retroReflection[idx] = retro;
    }

    void buildPowerIterationKernel(const uint2 threadID)
    {
        float3 wj = ndfDirections[threadID.y];
        float3 wk = ndfDirections[threadID.x];
        
        uint N = ndfSize.x * ndfSize.y;
        ndfKernel[threadID.x + threadID.y * N] = (1.0f / N) * retroReflection[threadID.y] * saturate(dot(wk, wj));
    }

    void powerIteration(const uint2 threadID)
    {
        uint N = ndfSize.x * ndfSize.y;
        uint idx = threadID.x + threadID.y * ndfSize.x;
        
        float result = 0.0f;
        for (int i = 0; i < N; ++i)
        {
            result += ndfKernel[idx * N + i] * ndf[i];
        }
        ndfTmp[idx] = result;
    }
    
    float interpolate(RWStructuredBuffer<float> buf, float2 uv, uint2 size)
    {
        uv *= size - 1;
        int2 xy = clamp(int2(uv), int2(0), size - 2);
        uv -= xy;
        
        float v00 = buf[(xy.x + 0) + (xy.y + 0) * size.x];
        float v10 = buf[(xy.x + 1) + (xy.y + 0) * size.x];
        float v01 = buf[(xy.x + 0) + (xy.y + 1) * size.x];
        float v11 = buf[(xy.x + 1) + (xy.y + 1) * size.x];
        
        return lerp(lerp(v00, v10, uv.x), lerp(v01, v11, uv.x), uv.y);
    }

    float evalNDF  (float2 uv) { return interpolate(ndf,   uv, ndfSize); }

    float evalSigma(float2 uv) { return interpolate(sigma, uv, ndfSize); }
    
    void integrateSigma(const uint3 threadID)
    {
        uint N = ndfSize.x * ndfSize.y;
        float2 unitI = float2(threadID.z % ndfSize.x, threadID.z / ndfSize.x) / float2(ndfSize - 1);
        float3 wi = fromSpherical(unitToSpherical(unitI));

        float binIntegral = 0.0f;
        for (uint sy = 0; sy < kSampleCount.y; ++sy)
        {
            for (uint sx = 0; sx < kSampleCount.x; ++sx)
            {
                float2 offset = (float2(sx, sy) + 0.5f) * (1.0f / kSampleCount);
                float2 uv = (float2(threadID.xy) + offset) / sigmaIntegrationGrid;

                float phi = uv.y * 2.0f * float(M_PI);
                float z = uv.x;
                float r = sqrt(1.0f - uv.x * uv.x);
                float3 h = float3(cos(phi) * r, sin(phi) * r, z);
                float2 sphericalH = toSpherical(h);
                float2 unitH = sphericalToUnit(sphericalH);

                float pdf = 1.0f / float(2.0f * M_PI);
                binIntegral += saturate(dot(h, wi)) * evalNDF(unitH) / pdf;
            }
        }
        binIntegral *= 1.0f / (kSampleCount.x * kSampleCount.y);
        
        float partialSum = WaveActiveSum(binIntegral / WaveGetLaneCount());
        if (WaveIsFirstLane())
        {
            uint sliceIdx = (threadID.x + threadID.y * sigmaIntegrationGrid.y) / WaveGetLaneCount();
            sigma[threadID.z + N * sliceIdx] = partialSum;
        }
    }

    void sumSigma(const uint2 threadID)
    {
        uint N = ndfSize.x * ndfSize.y;
        uint M = sigmaIntegrationGrid.x * sigmaIntegrationGrid.y / WaveGetLaneCount();
        uint base = threadID.x + threadID.y * ndfSize.x;
        
        float sum = 0.0f;
        for (int i = 0; i < M; ++i)
        {
            sum += sigma[base + i * N];
        }
        sigma[base] = sum / M;
    }
    
    // TODO: Only works for the isotropic case. It's not entirely clear from the paper how
    // to compute the incident parametrization in the anisotropic case.
    // Only support isotropic measurements for now. Will have to contact authors if anisotropy
    // becomes important
    void computeTheta(const uint threadID)
    {
        float minSigma = sigma[ndfSize.x - 1];
        float x = lerp(1.0f, minSigma, threadID / (thetaSize - 1.0f));

        // These tables are small enough that a linear search is not a problem.
        int i;
        for (i = 1; i < ndfSize.x - 1; ++i)
        {
            if (sigma[i] <= x)
            {
                break;
            }
        }
        float a = sigma[i];
        float b = sigma[i - 1];
        float u = a == b ? 0.0f : (x - a) / (b - a);
        u = threadID == 0 ? 0.0f : (i - u) / (ndfSize.x - 1);
        
        thetas[threadID] = unitToSpherical(float2(u, 0.0f)).x;
    }

    void computeVNDF(const uint3 threadID)
    {
        float2 sphericalI = float2(thetas[threadID.y], phis[threadID.x]);
        float2 unitI = sphericalToUnit(sphericalI);
        
        uint2 idxH = uint2(threadID.z % vndfSize.z, threadID.z / vndfSize.z);
        float2 unitH = float2(idxH) / float2(vndfSize.zw - 1);
        
        float3 wi = fromSpherical(sphericalI);
        float3 h  = fromSpherical(unitToSpherical(unitH));
        
        uint vndfIdx = (threadID.x * vndfSize.y + threadID.y) * vndfSize.z * vndfSize.w + threadID.z;

        float jacobian = (2.0f * M_PI * M_PI * unitH.x * length(h.xy));
        vndf[vndfIdx] = jacobian * saturate(dot(wi, h)) * evalNDF(unitH) / evalSigma(unitI);
    }
    
    void acquireBRDF(const uint3 threadID)
    {
        uint2 idxI = threadID.xy;
        uint2 idxH = uint2(threadID.z % lumiSize.z, threadID.z / lumiSize.z);
        
        float2 slice = float2(idxI);
        float2 xi = float2(idxH) / float2(lumiSize.zw - 1);
        
        let vndf = InterpolatedDistribution2D(vndfSize, vndfMarginalBuf, vndfConditionalBuf, vndfBuf);
        
        float3 unitHPdf = vndf.sample(slice, xi);
        float2 unitH = unitHPdf.xy;
        
        float2 sphericalI = float2(thetas[idxI.y], phis[idxI.x]);
        float2 sphericalH = unitToSpherical(unitH);
        
        float2 unitI = sphericalToUnit(sphericalI);
        
        float3 wi = fromSpherical(sphericalI);
        float3 h  = fromSpherical(sphericalH);
        float3 wo = h * 2.0f * dot(h, wi) - wi;
        
        float3 fr;
        if (wo.z > 0.0f)
        {
            wi = clampDirection(wi);
            wo = clampDirection(wo);
            ShadingData sd = buildShadingData(wi);
            
            let lod = ExplicitLodTextureSampler(0.0f);
            let bsdf = gScene.materials.getBSDF(sd, lod);
            
            // TODO: We should probably take multiple samples if eval is actually probabilistic.
            // No way currently to find out if eval is deterministic or not.
            UniformSampleGenerator sg = UniformSampleGenerator(uint2(threadID.x * lumiSize.y + threadID.y, threadID.z), 0);
            
            fr = bsdf.eval(sd, wo, sg);
        }
        else
        {
            fr = float3(0.0f);
        }

        fr *= 4.0f * evalSigma(unitI) / evalNDF(unitH);
        
        uint sliceSize = lumiSize.z * lumiSize.w;
        uint lumiBase = (threadID.x * lumiSize.y + threadID.y) * sliceSize     + threadID.z;
        uint  rgbBase = (threadID.x * lumiSize.y + threadID.y) * sliceSize * 3 + threadID.z;
        
        lumi[lumiBase] = luminance(fr);
        rgb[rgbBase + sliceSize * 0] = fr.x;
        rgb[rgbBase + sliceSize * 1] = fr.y;
        rgb[rgbBase + sliceSize * 2] = fr.z;
    }
};

ParameterBlock<RGLAcquisition> gAcquisition;


[numthreads(16, 16, 1)]
void measureRetroreflection(uint3 threadID : SV_DispatchThreadID)
{
    if (any(threadID.xy >= gAcquisition.ndfSize)) return;
    
    gAcquisition.measureRetroreflection(threadID.xy);
}

[numthreads(16, 16, 1)]
void buildPowerIterationKernel(uint3 threadID : SV_DispatchThreadID)
{
    int N = gAcquisition.ndfSize.x * gAcquisition.ndfSize.y;
    if (any(threadID >= N)) return;
    
    gAcquisition.buildPowerIterationKernel(threadID.xy);
}

[numthreads(16, 16, 1)]
void powerIteration(uint3 threadID : SV_DispatchThreadID)
{
    if (any(threadID.xy >= gAcquisition.ndfSize)) return;
    
    gAcquisition.powerIteration(threadID.xy);
}

[numthreads(16, 16, 1)]
void integrateSigma(uint3 threadID : SV_DispatchThreadID)
{
    int N = gAcquisition.ndfSize.x * gAcquisition.ndfSize.y;
    if (any(threadID >= int3(gAcquisition.sigmaIntegrationGrid, N))) return;
    
    gAcquisition.integrateSigma(threadID);
}

[numthreads(16, 16, 1)]
void sumSigma(uint3 threadID : SV_DispatchThreadID)
{
    if (any(threadID.xy >= gAcquisition.ndfSize)) return;
    
    gAcquisition.sumSigma(threadID.xy);
}

[numthreads(16, 16, 1)]
void computeTheta(uint3 threadID : SV_DispatchThreadID)
{
    if (threadID.x >= gAcquisition.thetaSize) return;
    
    gAcquisition.computeTheta(threadID.x);
}

[numthreads(16, 16, 1)]
void computeVNDF(uint3 threadID : SV_DispatchThreadID)
{
    int N = gAcquisition.vndfSize.z * gAcquisition.vndfSize.w;
    if (any(threadID >= int3(gAcquisition.vndfSize.xy, N))) return;
    
    gAcquisition.computeVNDF(threadID);
}

[numthreads(16, 16, 1)]
void acquireBRDF(uint3 threadID : SV_DispatchThreadID)
{
    int N = gAcquisition.lumiSize.z * gAcquisition.lumiSize.w;
    if (any(threadID >= int3(gAcquisition.lumiSize.xy, N))) return;
    
    gAcquisition.acquireBRDF(threadID);
}
