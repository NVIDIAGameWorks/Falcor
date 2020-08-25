/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import Scene.Raytracing;
import Scene.HitInfo;
import Utils.Timing.GpuTimer;
import Utils.Sampling.SampleGenerator;

cbuffer PerFrameCB
{
    uint frameCount;
};

RWTexture2D<uint2> gVBuffer;
RWTexture2D<uint> gTime;

#define isValid(name) (is_valid_##name != 0)

/** ***************************** Ray index 0 ****************************** */

struct RayData
{
    int dummy; // TODO: Passing in an empty payload struct doesn't work. Declare a dummy variable so that the compiler doesn't remove the declaration.
};

[shader("miss")]
void miss(inout RayData rayData)
{
    // Write invalid hit to output buffer.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gVBuffer[launchIndex] = uint2(HitInfo::kInvalidIndex);
}

[shader("anyhit")]
void anyHit(
    uniform HitShaderParams hitParams,
    inout RayData rayData : SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if !DISABLE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
#endif
}

[shader("closesthit")]
void closestHit(
    uniform HitShaderParams hitParams,
    inout RayData rayData,
    BuiltInTriangleIntersectionAttributes attribs)
{
    // Store hit information. Note we don't access the materials here.
    HitInfo hit;
    hit.meshInstanceID = hitParams.getGlobalHitID();
    hit.primitiveIndex = PrimitiveIndex();
    hit.barycentrics = attribs.barycentrics;
    uint2 packedHitInfo = hit.encode();

    // Write hit info to output buffer.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gVBuffer[launchIndex] = packedHitInfo;
}


/** ******************************** RayGen ******************************** */

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    GpuTimer timer;
    if (isValid(gTime)) timer.start();

    // Sample ray
#if USE_DEPTH_OF_FIELD
    SampleGenerator sg = SampleGenerator.create(launchIndex, frameCount);
    RayDesc ray = gScene.camera.computeRayThinlens(launchIndex, launchDim, sampleNext2D(sg)).toRayDesc();
#else
    RayDesc ray = gScene.camera.computeRayPinhole(launchIndex, launchDim).toRayDesc();
#endif

    // Trace ray
    RayData rayData;
    TraceRay(gRtScene, RAY_FLAG_NONE, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, hitProgramCount, 0 /* missIdx */, ray, rayData);

    // Write time.
    if (isValid(gTime)) gTime[launchIndex] = timer.getElapsed();
}
