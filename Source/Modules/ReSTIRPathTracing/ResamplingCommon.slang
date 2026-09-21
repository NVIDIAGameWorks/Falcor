/***************************************************************************
 # Copyright (c) 2015-26, NVIDIA CORPORATION. All rights reserved.
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
import Scene.HitInfo;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
import LoadShadingDataDynamic;
import ReSTIRPT;
import PathReservoir;
import Scene.RaytracingInline;
import Scene.Camera.CameraData;
import Utils.Sampling.SampleGenerator;

static const float kDuplicationMaxValue = 288.f;

[ForceInline]
float2 reprojectPixelForReuse(
    uint2 pixel,
    float2 motionVector,
    uint2 frameDim,
    bool useAreaReSTIRBackprojection,
    bool useStochasticReprojection,
    inout SampleGenerator sg)
{
    float2 prevPixel = pixel + motionVector * frameDim
        + (useAreaReSTIRBackprojection
            ? 0.f
            : (useStochasticReprojection
                ? sampleNext2D(sg)
                : 0.5));
    if (!useAreaReSTIRBackprojection)
        prevPixel = int2(prevPixel);
    return prevPixel;
}

/// Reproject `pixel` for duplication-map index lookup. Always rounds to integer coordinates.
[ForceInline]
int2 reprojectPixelForDupMap(
    uint2 pixel,
    float2 motionVector,
    uint2 frameDim,
    bool useAreaReSTIRBackprojection,
    bool useStochasticReprojection,
    inout SampleGenerator sg)
{
    return int2(pixel + motionVector * frameDim
        + (useStochasticReprojection && !useAreaReSTIRBackprojection
            ? sampleNext2D(sg)
            : 0.5));
}

Ray computeRayPredefinedThinLens(uint2 pixel, uint2 frameDim, uint frameCount, bool applyJitter = true)
{
    SampleGenerator sg = SampleGenerator(pixel, frameCount);
    float2 u = sampleNext2D(sg);
    return gScene.camera.computeRayThinlens(pixel, frameDim, u, applyJitter);
}

Ray computeRayPredefinedThinLensPrevFrame(uint2 pixel, uint2 frameDim, uint frameCount, bool applyJitter = true)
{
    SampleGenerator sg = SampleGenerator(pixel, frameCount - 1);
    float2 u = sampleNext2D(sg);
    return gScene.camera.computeRayThinlensPrevFrame(pixel, frameDim, u, applyJitter);
}

ShadingData getPixelShadingData(ReSTIRPathTracingParams params, Texture2D<PackedHitInfo> vbuffer, int2 pixel, out HitInfo hit)
{
    ShadingData sd = {};
    Ray ray = computeRayPredefinedThinLens(pixel, params.frameDim, params.frameCount);
    hit = HitInfo(vbuffer[pixel]);
    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, false);
    }
    return sd;
}

ShadingData getPixelShadingData( ReSTIRPathTracingParams params, StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer, float2 pixel, uint shiftedPrimaryHitBufferOffset, float2 subPixelSampleUV, float2 lensSampleUV, out HitInfo hit )
{
    ShadingData sd = {};
    Ray ray = gScene.camera.computeRayThinlensSubPixelRandom(pixel, params.frameDim, subPixelSampleUV, lensSampleUV);
    hit = HitInfo(shiftedPrimaryHitBuffer[shiftedPrimaryHitBufferOffset]);
    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, false);
    }
    return sd;
}

ShadingData getPixelTemporalShadingData( ReSTIRPathTracingParams params, Texture2D<PackedHitInfo> temporalVbuffer, int2 pixel, out HitInfo hit )
{
    ShadingData sd = {};
    Ray ray = computeRayPredefinedThinLensPrevFrame(pixel, params.frameDim, params.frameCount);
    hit = HitInfo(temporalVbuffer[pixel]);
    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, true);
    }
    return sd;
}

ShadingData getPixelTemporalShadingData( ReSTIRPathTracingParams params, StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer, float2 pixel, uint shiftedPrimaryHitBufferOffset, float2 subPixelSampleUV, float2 lensSampleUV, out HitInfo hit )
{
    ShadingData sd = {};
    // Note: it seems there will be a strange precison error when using previous camera data to compute ray direction
    Ray ray = gScene.camera.computeRayThinlensSubPixelRandomPrevFrame(pixel, params.frameDim, subPixelSampleUV, lensSampleUV);
    hit = HitInfo(shiftedPrimaryHitBuffer[shiftedPrimaryHitBufferOffset]);
    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, true);
    }
    return sd;
}

HitInfo getPixelHitInfo(ReSTIRPathTracingParams params, Texture2D<PackedHitInfo> vbuffer, int2 pixel, out float3 V)
{
    Ray ray = computeRayPredefinedThinLens(pixel, params.frameDim, params.frameCount);
    V = -ray.dir;
    return HitInfo(vbuffer[pixel]);
}

HitInfo getPixelHitInfo(ReSTIRPathTracingParams params, RWTexture2D<PackedHitInfo> vbuffer, int2 pixel, out float3 V)
{
    Ray ray = computeRayPredefinedThinLens(pixel, params.frameDim, params.frameCount);
    V = -ray.dir;
    return HitInfo(vbuffer[pixel]);
}

HitInfo getPixelHitInfo( ReSTIRPathTracingParams params, StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer, float2 pixel, uint shiftedPrimaryHitBufferOffset, float2 subPixelSampleUV, float2 lensSampleUV, out float3 V )
{
    Ray ray = gScene.camera.computeRayThinlensSubPixelRandom(pixel, params.frameDim, subPixelSampleUV, lensSampleUV);
    V = -ray.dir;
    return HitInfo(shiftedPrimaryHitBuffer[shiftedPrimaryHitBufferOffset]);
}

HitInfo getPixelTemporalHitInfo( ReSTIRPathTracingParams params, Texture2D<PackedHitInfo> temporalVbuffer, int2 pixel, out float3 V)
{
    Ray ray = computeRayPredefinedThinLensPrevFrame(pixel, params.frameDim, params.frameCount);
    V = -ray.dir;
    return HitInfo(temporalVbuffer[pixel]);
}

HitInfo getPixelTemporalHitInfo( ReSTIRPathTracingParams params, StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer, float2 pixel, uint shiftedPrimaryHitBufferOffset, float2 subPixelSampleUV, float2 lensSampleUV, out float3 V )
{
    // Note: it seems there will be a strange precison error when using previous camera data to compute ray direction
    Ray ray = gScene.camera.computeRayThinlensSubPixelRandomPrevFrame(pixel, params.frameDim, subPixelSampleUV, lensSampleUV);
    V = -ray.dir;
    return HitInfo(shiftedPrimaryHitBuffer[shiftedPrimaryHitBufferOffset]);
}

ShadingData getPixelShadingDataGeneral( ReSTIRPathTracingParams params, Texture2D<PackedHitInfo> vbuffer, Texture2D<PackedHitInfo> temporalVbuffer, int2 pixel, out HitInfo hit, bool isTemporal)
{
    ShadingData sd = {};
    Ray ray;
    if (isTemporal)
        ray = computeRayPredefinedThinLensPrevFrame(pixel, params.frameDim, params.frameCount);
    else
        ray = computeRayPredefinedThinLens(pixel, params.frameDim, params.frameCount);

    if (isTemporal)
        hit = HitInfo(temporalVbuffer[pixel]);
    else
        hit = HitInfo(vbuffer[pixel]);

    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, isTemporal);
    }

    return sd;
}

ShadingData getPixelShadingDataGeneral( ReSTIRPathTracingParams params, StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer,
    float2 pixel, uint shiftedPrimaryHitBufferOffset, float2 subPixelSampleUV, float2 lensSampleUV, out HitInfo hit, bool isPrevFrame )
{
    ShadingData sd = {};
    // Note: it seems there will be a strange precison error when using previous camera data to compute ray direction
    // If we are retracing in previous frame, we first need to generate the ray using previous camera data
    Ray ray;
    if (isPrevFrame)
    {
        ray = gScene.camera.computeRayThinlensSubPixelRandomPrevFrame(pixel, params.frameDim, subPixelSampleUV, lensSampleUV);
    }
    else
    {
        ray = gScene.camera.computeRayThinlensSubPixelRandom(pixel, params.frameDim, subPixelSampleUV, lensSampleUV);
    }

    hit = HitInfo(shiftedPrimaryHitBuffer[shiftedPrimaryHitBufferOffset]);

    if (hit.isValid())
    {
        sd = loadShadingData(hit, ray.origin, ray.dir, isPrevFrame);
    }

    return sd;
}

bool isBorderPixel(ReSTIRPathTracingParams params, int2 pixel) { return any(pixel == 0 || pixel == params.frameDim - 1); }

bool isValidScreenRegion(ReSTIRPathTracingParams params, int2 pixel) { return all(pixel >= 0 && pixel < params.frameDim); }

bool isValidScreenRegionFloat(ReSTIRPathTracingParams params, float2 pixel) { return all(pixel >= -1.f && pixel < float2(params.frameDim) + 1.f); }

HitInfo getPixelTemporalHitInfo( ReSTIRPathTracingParams params, Texture2D<PackedHitInfo> temporalVbuffer, int2 pixel)
{
    if (!isValidScreenRegion(params, pixel)) return HitInfo();
    Ray ray = computeRayPredefinedThinLensPrevFrame(pixel, params.frameDim, params.frameCount);
    return HitInfo(temporalVbuffer[pixel]);
}

bool isReprojectedToOccluder(ShadingData centralSd, ShadingData neighborSd)
{
    float3 camPos = gScene.camera.data.posW;
    float centralDist = distance(camPos, centralSd.posW);
    float neighborDist = distance(camPos, neighborSd.posW);
    return centralDist - neighborDist > 0.1f * centralDist;
}

bool isValidGeometry(ShadingData centralSd, ShadingData neighborSd, bool useSymmetricCondition=false)
{
    float3 camPos = gScene.camera.data.posW;
    float centralDist = distance(camPos, centralSd.posW);
    float neighborDist = distance(camPos, neighborSd.posW);
    return dot(centralSd.frame.N, neighborSd.frame.N) >= 0.5f && abs(centralDist - neighborDist) < 0.1f * max(centralDist, (useSymmetricCondition ? neighborDist : 0.f));
}

VertexData getVertexData(HitInfo hit, float3 rayOrigin, float3 rayDir, bool isPrevFrame=false)
{
    uint materialID;
    return loadVertexData(hit, rayOrigin, rayDir, materialID, isPrevFrame);
}

bool isValidPackedHitInfo(PackedHitInfo packed)
{
    return packed.x != 0;
}

static const uint kNeighborOffsetCount = NEIGHBOR_OFFSET_COUNT;
static const uint kNeighborOffsetMask = kNeighborOffsetCount - 1;

int2 getNextNeighborPixel(const uint startIndex, Texture1D<float2> neighborOffsets, int2 pixel, int i, float gatherRadius)
{
    uint neighborIndex = (startIndex + i) & kNeighborOffsetMask;
    int2 neighborPixel = pixel + int2(neighborOffsets[neighborIndex] * gatherRadius);

    return neighborPixel;
}

// Retrace the primary ray for reused sample at different pixel (domain)
HitInfo retracePrimaryRay( float2 pixel, uint2 frameDim, float2 subPixelSampleUV, float2 lensSampleUV, const bool isPrevFrame)
{
    // Note: it seems there will be a strange precison error when using previous camera data to compute ray direction

    // If we are retracing in previous frame, we first need to generate the ray using previous camera data
    Ray ray;
    if (isPrevFrame)
    {
        ray = gScene.camera.computeRayThinlensSubPixelRandomPrevFrame(pixel, frameDim, subPixelSampleUV, lensSampleUV);
    }
    else
    {
        ray = gScene.camera.computeRayThinlensSubPixelRandom(pixel, frameDim, subPixelSampleUV, lensSampleUV);
    }

    SceneRayQuery<kUseAlphaTest> sceneRayQuery;
    float hitT;
    HitInfo hit = sceneRayQuery.traceRay(ray, hitT, RAY_FLAG_NONE, 0xff, isPrevFrame);

    return hit;
}

float computePrimaryReconnectionJacobian( float2 pixel, uint2 frameDim, HitInfo primaryHit, float3 primaryDir, float2 subPixelSampleUV, const bool isPrevFrame )
{
    bool isEnvMap = !primaryHit.isValid();

    // Note: it seems there will be a strange precison error when using previous camera data to compute ray direction

    // If we are retracing in previous frame, we first need to generate the ray using previous camera data
    float3 u_World;
    if (isPrevFrame)
    {
        u_World = computeRayThinlensGetPointOnFocalPlane(gScene.camera.prevData, pixel, frameDim, subPixelSampleUV);
    }
    else
    {
        u_World = computeRayThinlensGetPointOnFocalPlane(gScene.camera.data, pixel, frameDim, subPixelSampleUV);
    }

    uint dummy;
    VertexData v = loadVertexData(primaryHit, 0.f, 0.f, dummy, isPrevFrame);
    // fetch
    float3 x1 = v.posW;
    float3 n1 = v.faceNormalW;

    if (isEnvMap)
    {
        primaryDir = gScene.envMap.toWorld(gScene.envMap.toLocal(primaryDir, !isPrevFrame), isPrevFrame);
        x1 = u_World + primaryDir; // arbitrary point along primary dir
        n1 = 0.f;
    }

    float3 rayDir = normalize(x1 - u_World);

    float3 cameraW = isPrevFrame ? gScene.camera.prevData.cameraW : gScene.camera.data.cameraW;
    float3 cameraU = isPrevFrame ? gScene.camera.prevData.cameraU : gScene.camera.data.cameraU;
    float3 cameraV = isPrevFrame ? gScene.camera.prevData.cameraV : gScene.camera.data.cameraV;

    // center of lens
    float3 x_cam = isPrevFrame ? gScene.camera.data.prevPosW : gScene.camera.data.posW;
    float3 cam_dir = normalize(cameraW);

    // Z in camera space (assuming z axis is forward)
    float x1_camZ = dot(x1 - x_cam, cam_dir);

    float rayT = x1_camZ / dot(rayDir, cam_dir);

    float focalPlaneZ = length(cameraW);

    // compute Jacobian
    float d0Sqr = focalPlaneZ / abs(x1_camZ) * rayT;
    float d1Sqr = rayT - d0Sqr;
    d0Sqr *= d0Sqr;
    d1Sqr *= d1Sqr;

    float cos1 = abs(dot(rayDir, n1));
    float cos0 = abs(dot(rayDir, cam_dir));

    float JacobianTerm = d0Sqr / cos0;
    if (!isEnvMap) JacobianTerm *= cos1 / d1Sqr;

    // inverse disk sampling
    return JacobianTerm;
}

float3 computeRayThinlensGetPointOnFocalPlane(CameraData camData, float2 pixel, uint2 frameDim, float2 u0)
{
    // Sample position in screen space in [0,1] with origin at the top-left corner.
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    float2 p = float2(pixel + u0) / frameDim;
    float2 ndc = float2(2, -2) * p + float2(-1, 1);

    // Compute the normalized ray direction assuming a thin-lens camera.
    float3 ray_origin = camData.posW;
    float3 ray_dir = ndc.x * camData.cameraU + ndc.y * camData.cameraV + camData.cameraW;
    float3 rayTarget = ray_origin + ray_dir;

    return rayTarget;
}

float3 solveLensSampleForPrimaryHit( float2 pixel, uint2 frameDim, HitInfo primaryHit, float3 primaryDir, float2 subPixelSampleUV, const bool isPrevFrame )
{
    bool isEnvMap = !primaryHit.isValid();

    // Note: it seems there will be a strange precison error when using previous camera data to compute ray direction

    // If we are retracing in previous frame, we first need to generate the ray using previous camera data
    float3 u_World;
    if (isPrevFrame)
    {
        u_World = computeRayThinlensGetPointOnFocalPlane(gScene.camera.prevData, pixel, frameDim, subPixelSampleUV);
    }
    else
    {
        u_World = computeRayThinlensGetPointOnFocalPlane(gScene.camera.data, pixel, frameDim, subPixelSampleUV);
    }


    float3 x1;
    float3 n1;
    if (isEnvMap)
    {
        primaryDir = gScene.envMap.toWorld(gScene.envMap.toLocal(primaryDir, !isPrevFrame), isPrevFrame);
        x1 = u_World + primaryDir;// arbitrary point along primary dir
        n1 = 0.f;
    }
    else
    {
        uint dummy;
        VertexData v = loadVertexData(primaryHit, 0.f, 0.f, dummy, isPrevFrame);
        // fetch
        x1 = v.posW;
        n1 = v.faceNormalW;
    }

    float3 rayDir = isEnvMap ? primaryDir : normalize(x1 - u_World);

    float3 cameraW = isPrevFrame ? gScene.camera.prevData.cameraW : gScene.camera.data.cameraW;
    float3 cameraU = isPrevFrame ? gScene.camera.prevData.cameraU : gScene.camera.data.cameraU;
    float3 cameraV = isPrevFrame ? gScene.camera.prevData.cameraV : gScene.camera.data.cameraV;

    // center of lens
    float3 x_cam = isPrevFrame ? gScene.camera.data.prevPosW : gScene.camera.data.posW;
    float3 cam_dir = normalize(cameraW);

    // Z in camera space (assuming z axis is forward)
    float x1_camZ = dot(x1 - x_cam, cam_dir);

    float rayT = x1_camZ / dot(rayDir, cam_dir);

    // ray-lens plane intersection
    float3 x0 = x1 - rayT * rayDir;

    // use lens center as (0,0,0)
    float3 x0_local_temp = x0 - x_cam;
    float x0_local_x = dot(x0_local_temp, normalize(cameraU)) / gScene.camera.data.apertureRadius;
    float x0_local_y = dot(x0_local_temp, normalize(cameraV)) / gScene.camera.data.apertureRadius;

    // outside disk
    if (x0_local_x * x0_local_x + x0_local_y * x0_local_y >= 1.f) return float3(0.f, 0.f, 0.f);

    // test shadow ray

    float3 shadowRayDir = x1 - x0;
    float shadowRayLength = isEnvMap ? 1e10 : length(shadowRayDir);
    shadowRayDir = isEnvMap ? shadowRayDir : shadowRayDir / shadowRayLength;

    Ray ray = Ray(x0, shadowRayDir, 0.f, 0.999 * shadowRayLength);
    SceneRayQuery<kUseAlphaTest> sceneRayQuery;
    if (!sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame)) return float3(0.f, 0.f, 0.f);

    float JacobianTerm = 1.f;

    float focalPlaneZ = length(cameraW);
    // compute Jacobian
    float d0 = focalPlaneZ / abs(x1_camZ) * rayT;
    float cos0 = abs(dot(rayDir, cam_dir));
    JacobianTerm = d0 * d0 / cos0;

    if (!isEnvMap)
    {
        float d1Sqr = rayT - d0;
        d1Sqr *= d1Sqr;
        float cos1 = abs(dot(rayDir, n1));
        JacobianTerm *= cos1 / d1Sqr;
    }

    // inverse disk sampling
    return float3(saturate(inverse_sample_disk(float2(x0_local_x, x0_local_y))), JacobianTerm);
}

float computeOnePixelDistW(float2 frameDim)
{
    // Pick two neighbor pixels on screen and project to focal plane in world space
    float2 p1 = (float2(uint2(frameDim / 2)) + float2(0.5f, 0.5f)) / frameDim;
    float2 p2 = (float2(uint2(frameDim / 2 + 1)) + float2(0.5f, 0.5f)) / frameDim;
    float2 ndc1 = float2(2, -2) * p1 + float2(-1, 1);
    float2 ndc2 = float2(2, -2) * p2 + float2(-1, 1);
    float3 p1PosW = gScene.camera.data.posW + ndc1.x * gScene.camera.data.cameraU + ndc1.y * gScene.camera.data.cameraV + gScene.camera.data.cameraW;
    float3 p2PosW = gScene.camera.data.posW + ndc2.x * gScene.camera.data.cameraU + ndc2.y * gScene.camera.data.cameraV + gScene.camera.data.cameraW;
    float onePixeldistW = distance(p1PosW, p2PosW);
    return onePixeldistW;
}

float computeScreenSpaceCircleOfConfusion(float2 frameDim, float3 primaryHitPosW, bool isEnvMap)
{
    float onePixeldistW = computeOnePixelDistW(frameDim);
    // Compute lens diameter on focal plane
    float dLens = gScene.camera.data.apertureRadius * 2.0f;
    float3 primaryHitPosCamera = mul(gScene.camera.data.viewMat, float4(primaryHitPosW, 1.0f)).xyz;
    float dFocal = dLens * (isEnvMap ? 1.f : abs(abs(primaryHitPosCamera.z) - gScene.camera.data.focalDistance) / abs(primaryHitPosCamera.z));

    // Scale the diameter to screen space ([-1, 1] -> [0, 1])
    return dFocal / onePixeldistW;
}

inline float fitted_function(float x)
{
    static const float3 coefficents = float3(6.2,5.6,0.2);
    return coefficents.x / (x + coefficents.y) + coefficents.z;
}

float computeConfidenceWeightForTwoShifts(float2 frameDim, float3 primaryPos, bool isEnvMap)
{
    // Note: use sample's primay hit to compute CoC here will cause bias, so we need to use pixel center's hit
    float coc = computeScreenSpaceCircleOfConfusion(frameDim, primaryPos, isEnvMap);

    // Scale the random replay confidence weight
    float confidenceWeightScaler = coc <= 2.0f ? 1.0f : fitted_function(coc);
    confidenceWeightScaler = clamp(confidenceWeightScaler, 0.2f, 1.0f);
    return confidenceWeightScaler;
}

float computeImageSplatJacobian(float3 x0, float3 x1, float3 n1, const bool isPrevFrame)
{
    float3 disp = x1 - x0;
    float dist2 = dot(disp, disp);
    float3 dir = disp / sqrt(dist2);
    float3 cameraForward = normalize(isPrevFrame ? gScene.camera.prevData.cameraW : gScene.camera.data.cameraW);
    float cosTheta = dot(cameraForward, dir);
    float G = all(n1 == 0.f) ? 1.f : abs(dot(-dir, n1)) / dist2;
    return G / (cosTheta * cosTheta * cosTheta);
}

float3 splatHitToImagePlane(uint2 frameDim, HitInfo primaryHit, float3 primaryDir, float2 lensSampleUV, const bool isPrevFrame,
    bool usePackedPosNormal = false, bool isPackedPosNormalValid = true)
{
    bool isEnvMap = !primaryHit.isValid();

    float3 x_cam = isPrevFrame ? gScene.camera.data.prevPosW : gScene.camera.data.posW;

    // these are unnormalized
    float3 cameraU = isPrevFrame ? gScene.camera.prevData.cameraU : gScene.camera.data.cameraU;
    float3 cameraV = isPrevFrame ? gScene.camera.prevData.cameraV : gScene.camera.data.cameraV;
    float3 cameraW = isPrevFrame ? gScene.camera.prevData.cameraW : gScene.camera.data.cameraW;
    float apertureRadius = gScene.camera.data.apertureRadius; // assume aperture radius doesn't change
    ShadingFrame camSf;
    camSf.T = normalize(cameraU);
    camSf.B = normalize(cameraV);
    camSf.N = normalize(cameraW);
    float2 apertureSample = sample_disk(lensSampleUV);                      // Sample lies in the unit disk [-1,1]^2
    float2 a = apertureRadius * float2(apertureSample.x, apertureSample.y); // 2D location of lens vertex in camera space
    float3 x0 = x_cam + camSf.fromLocal(float3(a, 0));

    float3 x1, n1;
    float3 world_dir;

    if (isEnvMap)
    {
        world_dir = gScene.envMap.toWorld(gScene.envMap.toLocal(primaryDir, !isPrevFrame), isPrevFrame);
        x1 = x0 + world_dir;
        n1 = 0.f;
    }
    else
    {
        uint dummy;
        VertexData v = loadVertexData(primaryHit, 0.f, 0.f, dummy, isPrevFrame);
        // fetch
        x1 = v.posW;
        n1 = v.faceNormalW;
        world_dir = normalize(x1 - x0);
    }

    // for curve, we need to check intersection
    if (!isEnvMap && primaryHit.getType() == HitType::Curve && dot(world_dir, -n1) < 0.f) return float3(0.f);

    // transform to camera space direction
    float3 cam_dir = camSf.toLocal(world_dir);

    // reject if dir.z <= 0
    if (cam_dir.z <= 0) return float3(0.f);

    // intersect with the image plane
    float d = length(cameraW); // distance to image plane
    float2 t = a + d / cam_dir.z * cam_dir.xy;
    float2 ndc = float2(t.x / length(cameraU), t.y / length(cameraV));

    // reject if outside bound
    if (any(ndc < -1.f) || any(ndc >= 1.f)) return float3(0.f);

    float2 imageCoord = float2(0.5 * ndc.x + 0.5, -0.5 * ndc.y + 0.5) * frameDim;

    // test shadow ray
    float3 shadowRayDir = x1 - x0;
    float shadowRayLength = isEnvMap ? 1e10 : length(shadowRayDir);
    shadowRayDir = isEnvMap ? shadowRayDir : shadowRayDir / shadowRayLength;

    Ray ray = Ray( x0, shadowRayDir, 0.f, 0.999 * shadowRayLength ); // alternatively, compute shadowRayDir with surface-offseted x1 position 
    SceneRayQuery<kUseAlphaTest> sceneRayQuery;

    if (!sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff, isPrevFrame)) return float3(0.f);

    float dstJacobian = computeImageSplatJacobian(x0, x1, n1, isPrevFrame);
    float Jacobian;
    {
        float3 x_cam = !isPrevFrame ? gScene.camera.data.prevPosW : gScene.camera.data.posW;

        // these are unnormalized
        float3 cameraU = !isPrevFrame ? gScene.camera.prevData.cameraU : gScene.camera.data.cameraU;
        float3 cameraV = !isPrevFrame ? gScene.camera.prevData.cameraV : gScene.camera.data.cameraV;
        float3 cameraW = !isPrevFrame ? gScene.camera.prevData.cameraW : gScene.camera.data.cameraW;
        // assume aperture radius doesn't change
        ShadingFrame camSf;
        camSf.T = normalize(cameraU);
        camSf.B = normalize(cameraV);
        camSf.N = normalize(cameraW);
        float3 x0 = x_cam + camSf.fromLocal(float3(a, 0));

        float3 x1, n1;

        if (isEnvMap)
        {
            x1 = x0 + world_dir;
            n1 = 0.f;
        }
        else
        {
            uint dummy;
            VertexData v = loadVertexData(primaryHit, 0.f, 0.f, dummy, !isPrevFrame);
            // fetch
            x1 = v.posW;
            n1 = v.faceNormalW;
        }

        float sreconnectionJacobian = computeImageSplatJacobian(x0, x1, n1, !isPrevFrame);
        Jacobian = dstJacobian / sreconnectionJacobian;
    }

    return float3(imageCoord, Jacobian);
}

bool hasCameraChanged()
{
    return any(gScene.camera.data.prevPosW != gScene.camera.data.posW) || any(gScene.camera.prevData.cameraU != gScene.camera.data.cameraU) ||
           any(gScene.camera.prevData.cameraV != gScene.camera.data.cameraV) || any(gScene.camera.prevData.cameraW != gScene.camera.data.cameraW)
           || gScene.camera.prevData.apertureRadius != gScene.camera.data.apertureRadius;
}

bool shouldGetSamePath(HitInfo temporalShiftedPrimaryHit, bool isSameSample)
{
    if (!temporalShiftedPrimaryHit.isValid()) return false;
    bool cameraChanged = hasCameraChanged();
    const GeometryInstanceData instance = gScene.getGeometryInstance(temporalShiftedPrimaryHit.getInstanceID());
    const float4x4 prevWorldMat = gScene.loadPrevWorldMatrix(instance.globalMatrixID);
    const float4x4 curWorldMat = gScene.loadWorldMatrix(instance.globalMatrixID);
    bool hasGeometryMoved = any(curWorldMat[0] != prevWorldMat[0]) || any(curWorldMat[1] != prevWorldMat[1]) ||
                            any(curWorldMat[2] != prevWorldMat[2]) || any(curWorldMat[3] != prevWorldMat[3]) || instance.isDynamic();
    return !cameraChanged && !hasGeometryMoved && isSameSample;
}

ReconnectionData loadReconnectionData( StructuredBuffer<ReconnectionData> reconnectionDataBuffer, StructuredBuffer<uint> reconnectionDataOffsets, uint shiftedPrimaryHitAddress,
    PathReservoir srcReservoir, HitInfo dstPrimaryHit, float3 dstPrimaryWo )
{
    ReconnectionData rcData;
    if (srcReservoir.pathFlags.rcVertexLength() > 1 && dstPrimaryHit.isValid())
    {
        rcData = reconnectionDataBuffer[reconnectionDataOffsets[shiftedPrimaryHitAddress]];
    }
    else
    {
        rcData.pathThroughput = dstPrimaryHit.isValid() ? 1.f : 0.f;
        rcData.rcPrevHit = dstPrimaryHit;
        rcData.rcPrevWo = dstPrimaryWo;
        rcData.rcFootprintFactor = 0.f; // not used
    }
    return rcData;
}

HitInfo LoadPixelCenterPrimaryHit( ReSTIRPathTracingParams params, ReSTIRPT restirpt,Texture2D<PackedHitInfo> vbuffer, int2 dstPixel, inout float3 dstPrimaryWo, inout float dstPrimaryRayFootprint, inout float3 dstPrimaryHitPosition )
{
    HitInfo centralPrimaryHit = getPixelHitInfo(params, vbuffer, dstPixel, dstPrimaryWo);
    VertexData centralPrimaryVertexData = getVertexData(centralPrimaryHit, gScene.camera.data.posW, -dstPrimaryWo);
    dstPrimaryRayFootprint = restirpt.computePrimaryRayFootprint( centralPrimaryVertexData.faceNormalW, gScene.camera.data.posW,
                                                                            -dstPrimaryWo, centralPrimaryVertexData.posW );
    dstPrimaryHitPosition = centralPrimaryVertexData.posW;
    return centralPrimaryHit;
}

HitInfo LoadShiftedPrimaryHit( ReSTIRPathTracingParams params, ReSTIRPT restirpt, StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer, Texture2D<PackedHitInfo> vbuffer,
    StructuredBuffer<float4> doFReconnectionShiftBuffer, bool isPrimaryHitReconnection,
    float2 dstPixel, uint shiftedPrimaryHitAddress, PathReservoir srcReservoir, inout float3 dstPrimaryWo, inout float dstPrimaryRayFootprint)
{
    HitInfo dstPrimaryHit;
    if (restirpt.kUseAreaReservoirs)
    {
        float2 lensSample = srcReservoir.lensSample;
        if (isPrimaryHitReconnection)
        {
            lensSample = doFReconnectionShiftBuffer[shiftedPrimaryHitAddress].xy;
        }
        dstPrimaryHit = getPixelHitInfo( params, shiftedPrimaryHitBuffer, dstPixel, shiftedPrimaryHitAddress, srcReservoir.filmSample, lensSample, dstPrimaryWo );
    }
    else
        dstPrimaryHit = getPixelHitInfo(params, vbuffer, dstPixel, dstPrimaryWo);

    VertexData dstPrimaryVertexData = getVertexData(dstPrimaryHit, gScene.camera.data.posW, -dstPrimaryWo);
    dstPrimaryRayFootprint = restirpt.computePrimaryRayFootprint( dstPrimaryVertexData.faceNormalW, gScene.camera.data.posW,
                                                                                    -dstPrimaryWo, dstPrimaryVertexData.posW );
    return dstPrimaryHit;
}

// this function is identical as the previous one, except that vbuffer is RWTexture2D
HitInfo LoadShiftedPrimaryHit( ReSTIRPathTracingParams params, ReSTIRPT restirpt, StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer, RWTexture2D<PackedHitInfo> vbuffer,
    StructuredBuffer<float4> doFReconnectionShiftBuffer, bool isPrimaryHitReconnection,
    float2 dstPixel, uint shiftedPrimaryHitAddress, PathReservoir srcReservoir, inout float3 dstPrimaryWo, inout float dstPrimaryRayFootprint )
{
    HitInfo dstPrimaryHit;
    if (restirpt.kUseAreaReservoirs)
    {
        float2 lensSample = srcReservoir.lensSample;
        if (isPrimaryHitReconnection)
        {
            lensSample = doFReconnectionShiftBuffer[shiftedPrimaryHitAddress].xy;
        }
        dstPrimaryHit = getPixelHitInfo(params, shiftedPrimaryHitBuffer, dstPixel, shiftedPrimaryHitAddress, srcReservoir.filmSample, lensSample, dstPrimaryWo);
    }
    else
        dstPrimaryHit = getPixelHitInfo(params, vbuffer, dstPixel, dstPrimaryWo);

    VertexData dstPrimaryVertexData = getVertexData(dstPrimaryHit, gScene.camera.data.posW, -dstPrimaryWo);
    dstPrimaryRayFootprint = restirpt.computePrimaryRayFootprint( dstPrimaryVertexData.faceNormalW, gScene.camera.data.posW,
                                                                                    -dstPrimaryWo, dstPrimaryVertexData.posW );
    return dstPrimaryHit;
}

HitInfo LoadShiftedPrimaryHitPrevFrame( ReSTIRPathTracingParams params, ReSTIRPT restirpt, StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer, Texture2D<PackedHitInfo> vbuffer,
    StructuredBuffer<float4> doFReconnectionShiftBuffer, bool isPrimaryHitReconnection,
    float2 dstPixel, uint shiftedPrimaryHitAddress, PathReservoir srcReservoir, float2 lensSample, inout float3 dstPrimaryWo, inout float dstPrimaryRayFootprint )
{
    HitInfo dstPrimaryHit;
    if (restirpt.kUseAreaReservoirs)
    {
        if (isPrimaryHitReconnection)
        {
            lensSample = doFReconnectionShiftBuffer[shiftedPrimaryHitAddress].xy;
        }
        dstPrimaryHit = getPixelTemporalHitInfo(params, shiftedPrimaryHitBuffer, dstPixel, shiftedPrimaryHitAddress, srcReservoir.filmSample, lensSample, dstPrimaryWo);
    }
    else
        dstPrimaryHit = getPixelTemporalHitInfo(params, vbuffer, dstPixel, dstPrimaryWo);

    VertexData dstPrimaryVertexData = getVertexData(dstPrimaryHit, gScene.camera.data.prevPosW, -dstPrimaryWo, true);
    dstPrimaryRayFootprint = restirpt.computePrimaryRayFootprint( dstPrimaryVertexData.faceNormalW, gScene.camera.data.prevPosW,
                                                                                    -dstPrimaryWo, dstPrimaryVertexData.posW );
    return dstPrimaryHit;
}

HitInfo LoadShiftedPrimaryHitAssumeAreaReSTIRAndLoadedLensSample( ReSTIRPathTracingParams params, ReSTIRPT restirpt, StructuredBuffer<PackedHitInfo> shiftedPrimaryHitBuffer,
    int2 dstPixel, uint shiftedPrimaryHitAddress, PathReservoir srcReservoir, inout float3 dstPrimaryWo, inout float dstPrimaryRayFootprint)
{
    HitInfo dstPrimaryHit;
    float2 lensSample = srcReservoir.lensSample;
    dstPrimaryHit = getPixelHitInfo(params, shiftedPrimaryHitBuffer, dstPixel, shiftedPrimaryHitAddress, srcReservoir.filmSample, lensSample, dstPrimaryWo);

    VertexData dstPrimaryVertexData = getVertexData(dstPrimaryHit, gScene.camera.data.posW, -dstPrimaryWo);
    dstPrimaryRayFootprint = restirpt.computePrimaryRayFootprint( dstPrimaryVertexData.faceNormalW, gScene.camera.data.posW,
                                                                                    -dstPrimaryWo, dstPrimaryVertexData.posW );
    return dstPrimaryHit;
}

void DetermineMCap( inout float MCap,
    Texture2D<uint> duplicationMap,
    int2 prevPixelForDupMap,
    bool enableDupMapMCapDownweighting,
    float duplicationMCapPower
)
{
    float duplicationRate = 0.f;
    if (enableDupMapMCapDownweighting)
        duplicationRate = duplicationMap[prevPixelForDupMap] / kDuplicationMaxValue;

    MCap = enableDupMapMCapDownweighting ? lerp(MCap, 1.f, pow(duplicationRate, duplicationMCapPower)) : MCap;
}

