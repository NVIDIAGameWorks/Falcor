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
__exported import Scene.Shading;
__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;

RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float>  gDepth;
RWTexture2D<float2> gMotionVector;
RWTexture2D<float4> gViewW;
RWTexture2D<uint> gTime;

#define is_valid(name) (is_valid_##name != 0)

#if !defined(COMPUTE_DEPTH_OF_FIELD) || !defined(USE_ALPHA_TEST) || !defined(RAY_FLAGS)
#error "Not all defines are set!"
#endif

struct VBufferRT
{
    static const bool kComputeDepthOfField = COMPUTE_DEPTH_OF_FIELD;
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const uint kRayFlags = RAY_FLAGS;

    uint2 frameDim;
    uint frameCount;

    Ray generateRay(uint2 pixel)
    {
        if (kComputeDepthOfField)
        {
            SampleGenerator sg = SampleGenerator(pixel, frameCount);
            return gScene.camera.computeRayThinlens(pixel, frameDim, sampleNext2D(sg));
        }
        else
        {
            return gScene.camera.computeRayPinhole(pixel, frameDim);
        }
    }

    void writeHit(uint2 pixel, float3 rayOrigin, float3 rayDir, const HitInfo hit)
    {
        gVBuffer[pixel] = hit.getData();

        VertexData v;
        float depth = 1.f;
        float3 prevPosW = float3(0.f);
        float2 motionVector = float2(0.f);

        if (hit.getType() == HitType::Triangle)
        {
            const TriangleHit triangleHit = hit.getTriangleHit();
            v = gScene.getVertexData(triangleHit);
            prevPosW = gScene.getPrevPosW(triangleHit);
        }
        else if (hit.getType() == HitType::DisplacedTriangle)
        {
            const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
            v = gScene.getVertexData(displacedTriangleHit, -rayDir);
            prevPosW = gScene.getPrevPosW(displacedTriangleHit);
        }
        else if (hit.getType() == HitType::Curve)
        {
            const CurveHit curveHit = hit.getCurveHit();
            v = gScene.getVertexDataFromCurve(curveHit);
            prevPosW = gScene.getPrevPosWFromCurve(curveHit);
        }
        else if (hit.getType() == HitType::SDFGrid)
        {
            const SDFGridHit sdfGridHit = hit.getSDFGridHit();
            v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);

            prevPosW = gScene.getPrevPosWFromSDFGrid(sdfGridHit, v.posW);
        }

        if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle || hit.getType() == HitType::Curve || hit.getType() == HitType::SDFGrid)
        {
            // Compute depth similar to raster (NDC).
            float4 curPosH = mul(gScene.camera.data.viewProjMatNoJitter, float4(v.posW, 1.f));
            depth = curPosH.z / curPosH.w;

            // Compute motion vector.
            float2 pixelPos = pixel + float2(0.5f, 0.5f);
            float4 prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));
            motionVector = calcMotionVector(pixelPos, prevPosH, frameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector
        }

        if (is_valid(gDepth))           gDepth[pixel]           = depth;
        if (is_valid(gMotionVector))    gMotionVector[pixel]    = motionVector;
    }

    void writeMiss(uint2 pixel, float3 rayOrigin, float3 rayDir)
    {
        gVBuffer[pixel] = {};

        if (is_valid(gDepth))           gDepth[pixel]           = 1.f;
        if (is_valid(gMotionVector))    gMotionVector[pixel]    = {};
    }

    void writeAux(uint2 pixel, const Ray ray)
    {
        // Write view direction.
        if (is_valid(gViewW)) gViewW[pixel] = float4(-ray.dir, 0.f);
    }

    void beginTime(inout GpuTimer timer)
    {
        if (is_valid(gTime)) timer.start();
    }

    void endTime(uint2 pixel, inout GpuTimer timer)
    {
        if (is_valid(gTime)) gTime[pixel] = timer.getElapsed();
    }
};
