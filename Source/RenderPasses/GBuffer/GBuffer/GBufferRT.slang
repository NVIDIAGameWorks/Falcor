/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
import Scene.Scene;
import Scene.Shading;
__exported import Scene.HitInfo;
__exported import Utils.Timing.GpuTimer;
__exported import Utils.Math.Ray;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Experimental.Scene.Material.TexLODTypes;
import Experimental.Scene.Material.TexLODHelpers;
import GBufferHelpers;

// GBuffer channels
RWTexture2D<float4> gPosW;
RWTexture2D<float4> gNormW;
RWTexture2D<float4> gTangentW;
RWTexture2D<float4> gTexC;
RWTexture2D<float4> gDiffuseOpacity;
RWTexture2D<float4> gSpecRough;
RWTexture2D<float4> gEmissive;
RWTexture2D<float4> gMatlExtra;

// GBufferRT channels
RWTexture2D<PackedHitInfo> gVBuffer;
RWTexture2D<float2> gLinearZ;
RWTexture2D<float>  gDepth;
RWTexture2D<float2> gMotionVectors;
RWTexture2D<float4> gMotionVectorsW;
RWTexture2D<float4> gNormalWRoughness;
RWTexture2D<float4> gRoughness;
RWTexture2D<float4> gMetallic;
RWTexture2D<float4> gFaceNormalW;
RWTexture2D<float4> gViewW;
RWTexture2D<uint> gTime;
RWTexture2D<float> gDisocclusion;

#define is_valid(name) (is_valid_##name != 0)

static const float kEnvMapDepth = 100000000.0f; // Arbitrary big number

/** Ray differentials for primary hit. Code from RayTracingGems, Chapter 20.
*/
void computeRayDifferentials(const TriangleHit hit, float3 rayDir, float hitT, const Camera camera, float2 invFrameDim, out float2 ddx, out float2 ddy)
{
    // TODO: Is this code correct for instance transforms that flip the handedness of the coordinate system?

    // Ray differentials
    float3 P[3];
    gScene.getVertexPositionsW(hit.instanceID, hit.primitiveIndex, P);
    float3 e1 = P[1] - P[0];
    float3 e2 = P[2] - P[0];
    float3 d = rayDir;
    float k = dot(cross(e1, e2), d);
    k = abs(k) > 1e-20f ? rcp(k) : 0.0f;
    float3 cu = cross(e2, d);
    float3 cv = cross(d, e1);
    // Assumes a normalized ray direction
    float3 dx = camera.data.cameraU * 2.f * invFrameDim.x / camera.data.focalDistance; // dDdx in ray gen
    float3 dy = camera.data.cameraV * 2.f * invFrameDim.y / camera.data.focalDistance; // dDdy in ray gen
    float3 q = dx * hitT; // Transfer to primary hit
    float3 r = dy * hitT;
    float dudx = k * dot(cu, q);
    float dudy = k * dot(cu, r);
    float dvdx = k * dot(cv, q);
    float dvdy = k * dot(cv, r);
    float2 T[3];
    gScene.getVertexTexCoords(hit.instanceID, hit.primitiveIndex, T);
    float2 g1 = T[1] - T[0];
    float2 g2 = T[2] - T[0];
    float dsdx = (dudx * g1.x + dvdx * g2.x);
    float dsdy = (dudy * g1.x + dvdy * g2.x);
    float dtdx = (dudx * g1.y + dvdx * g2.y);
    float dtdy = (dudy * g1.y + dvdy * g2.y);
    ddx = float2(dsdx, dtdx);
    ddy = float2(dsdy, dtdy);
}

void computeAnisotropicAxesRayCones(const TriangleHit hit, VertexData v, float3 rayDir, float hitT, float pixelAngle, out float2 ddx, out float2 ddy)
{
    float3 positions[3];
    float2 texCoords[3];
    gScene.getVertexPositionsW(hit.instanceID, hit.primitiveIndex, positions);
    gScene.getVertexTexCoords(hit.instanceID, hit.primitiveIndex, texCoords);

    float coneRadiusAtHitPoint = hitT * tan(pixelAngle);
    // Using faceNormal, since it is needed for the barycentric computations inside computeAnisotropicEllipseAxes().
    computeAnisotropicEllipseAxes(v.posW, v.faceNormalW, rayDir, coneRadiusAtHitPoint, positions, texCoords, v.texC, ddx, ddy);
}

float3 computeDdxPosW(float3 posW, float3 normW, float2 invFrameDim)
{
    float3 projRight = normalize(cross(normW, cross(normW, gScene.camera.data.cameraV)));
    float distanceToHit = length(posW - gScene.camera.data.posW);
    float2 ddNdc = float2(2.f, -2.f) * invFrameDim;
    float distRight = distanceToHit * ddNdc.x / dot(normalize(gScene.camera.data.cameraV), projRight);
    return distRight * projRight;
}

float3 computeDdyPosW(float3 posW, float3 normW, float2 invFrameDim)
{
    float3 projUp = normalize(cross(normW, cross(normW, gScene.camera.data.cameraU)));
    float distanceToHit = length(posW - gScene.camera.data.posW);
    float2 ddNdc = float2(2.f, -2.f) * invFrameDim;
    float distUp = distanceToHit * ddNdc.y / dot(normalize(gScene.camera.data.cameraU), projUp);
    return distUp * projUp;
}


#if !defined(USE_DEPTH_OF_FIELD) || !defined(USE_ALPHA_TEST) || !defined(LOD_MODE) || !defined(ADJUST_SHADING_NORMALS) || !defined(RAY_FLAGS)
#error "Not all defines are set!"
#endif

struct GBufferRT
{
    static const bool kUseDepthOfField = USE_DEPTH_OF_FIELD;
    static const bool kUseAlphaTest = USE_ALPHA_TEST;
    static const TexLODMode kLODMode = TexLODMode(LOD_MODE);
    static const bool kAdjustShadingNormals = ADJUST_SHADING_NORMALS;
    static const uint kRayFlags = RAY_FLAGS;

    uint2 frameDim;
    float2 invFrameDim;
    uint frameCount;
    float screenSpacePixelSpreadAngle;

    Ray generateRay(uint2 pixel)
    {
        if (kUseDepthOfField)
        {
            SampleGenerator sg = SampleGenerator.create(pixel, frameCount);
            return gScene.camera.computeRayThinlens(pixel, frameDim, sampleNext2D(sg));
        }
        else
        {
            return gScene.camera.computeRayPinhole(pixel, frameDim);
        }
    }

    void writeHit(uint2 pixel, float3 rayOrigin, float3 rayDir, const HitInfo hit, float hitT)
    {
        VertexData v;
        ShadingData sd;
        float curveRadius = 0.f;
        float3 prevPosW = float3(0.f);
        float2 motionVector = float2(0.f);
        float4 motionVectorW = float4(0.f);

        if (hit.getType() == HitType::Triangle)
        {
            const TriangleHit triangleHit = hit.getTriangleHit();

            const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
            v = gScene.getVertexData(triangleHit);

            if (kLODMode == TexLODMode::Mip0)
            {
                sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, 0.f);
            }
            else if (kLODMode == TexLODMode::RayCones)
            {
                float2 ddx, ddy;
                computeAnisotropicAxesRayCones(triangleHit, v, rayDir, hitT, screenSpacePixelSpreadAngle, ddx, ddy);
                sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, ddx, ddy);
            }
            else if (kLODMode == TexLODMode::RayDiffs)
            {
                float2 ddx, ddy;
                computeRayDifferentials(triangleHit, rayDir, hitT, gScene.camera, invFrameDim, ddx, ddy);
                sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, ddx, ddy);
            }

            if (kAdjustShadingNormals) adjustShadingNormal(sd, v);

            prevPosW = gScene.getPrevPosW(triangleHit);
        }
        else if (hit.getType() == HitType::DisplacedTriangle)
        {
            const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();

            const uint materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
            v = gScene.getVertexData(displacedTriangleHit, -rayDir);
            sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, 0.f);

            if (kAdjustShadingNormals) adjustShadingNormal(sd, v);

            prevPosW = gScene.getPrevPosW(displacedTriangleHit);
        }
        else if (hit.getType() == HitType::Curve)
        {
            const CurveHit curveHit = hit.getCurveHit();

            const uint materialID = gScene.getCurveMaterialID(curveHit.instanceID);
            v = gScene.getVertexDataFromCurve(curveHit, curveRadius);
            sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, 0.f);

            prevPosW = gScene.getPrevPosWFromCurve(curveHit);
        }

        if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle || hit.getType() == HitType::Curve)
        {
            // Compute motion vector in screen and world space.
            float2 pixelPos = pixel + float2(0.5f, 0.5f);
            float4 prevPosH = mul(float4(prevPosW, 1.f), gScene.camera.data.prevViewProjMatNoJitter);
            motionVector = calcMotionVector(pixelPos, prevPosH, frameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector
            motionVectorW = float4(prevPosW - sd.posW, 0.0f);

            // Compute disocclusion.
            if (is_valid(gDisocclusion))
            {
                // Do the occlusion masking in linearZ space
                float4 curPosH = mul(float4(sd.posW, 1.f), gScene.camera.data.viewProjMatNoJitter);
                float depthDiff = prevPosH.w - curPosH.w;
                gDisocclusion[pixel] = depthDiff;
            }
        }

        // Write the outputs.
        GBuffer gbuf = storeGBufferOutput(sd, v);

        if (hit.getType() == HitType::Curve) gbuf.posW.w = curveRadius;

        gPosW[pixel]            = gbuf.posW;
        gNormW[pixel]           = gbuf.normW;
        gTangentW[pixel]        = gbuf.tangentW;
        gTexC[pixel]            = gbuf.texC;
        gDiffuseOpacity[pixel]  = gbuf.diffuseOpacity;
        gSpecRough[pixel]       = gbuf.specRough;
        gEmissive[pixel]        = gbuf.emissive;
        gMatlExtra[pixel]       = gbuf.matlExtra;

        if (is_valid(gFaceNormalW))
        {
            gFaceNormalW[pixel] = float4(v.faceNormalW, 0.f);
        }

        if (is_valid(gLinearZ))
        {
            float4 curPosH = mul(float4(sd.posW, 1.f), gScene.camera.data.viewProjMatNoJitter);
            float curLinearZ = curPosH.w;

            // TODO: Improve computation of derivatives:
            float3 ddxPosW = computeDdxPosW(sd.posW, sd.faceN, invFrameDim);
            float3 ddyPosW = computeDdyPosW(sd.posW, sd.faceN, invFrameDim);
            float4 curPosH_dx = mul(float4(sd.posW + ddxPosW, 1.f), gScene.camera.data.viewProjMatNoJitter);
            float4 curPosH_dy = mul(float4(sd.posW + ddxPosW, 1.f), gScene.camera.data.viewProjMatNoJitter);
            float ddxLinearZ = abs(curPosH_dx.w - curLinearZ);
            float ddyLinearZ = abs(curPosH_dy.w - curLinearZ);
            float dLinearZ = max(ddxLinearZ, ddyLinearZ);
            gLinearZ[pixel] = float2(curLinearZ, dLinearZ);
        }

        // Output a depth buffer similar to raster (NDC).
        if (is_valid(gDepth))
        {
            float4 curPosH = mul(float4(sd.posW, 1.f), gScene.camera.data.viewProjMatNoJitter);
            gDepth[pixel] = curPosH.z / curPosH.w;
        }

        if (is_valid(gMotionVectors)) gMotionVectors[pixel] = motionVector;
        if (is_valid(gMotionVectorsW)) gMotionVectorsW[pixel] = motionVectorW;

        // NRD-specific buffer that packs normals and clamps roughness.
        if (is_valid(gNormalWRoughness))
        {
            // TODO: roughness below 0.08 is used to indicate perfectly smooth surfaces
            // but that doesn't work well with NRD yet. Clamp roughness to 0.08 for now.

            // Clamp roughness so it's representable of what is actually used in the renderer.
            float clampedRoughness = max(0.08f, sd.linearRoughness);
            gNormalWRoughness[pixel] = float4(sd.N * 0.5 + 0.5, clampedRoughness);
        }

        if (is_valid(gRoughness))
        {
            gRoughness[pixel] = float4(float3(sd.linearRoughness), 1.0f);
        }

        if (is_valid(gMetallic))
        {
            gMetallic[pixel] = float4(float3(sd.metallic), 1.0f);
        }

        // Encode hit information.
        if (is_valid(gVBuffer))
        {
            gVBuffer[pixel] = hit.pack();
        }
    }

    void writeMiss(uint2 pixel, float3 rayOrigin, float3 rayDir)
    {
        gPosW[pixel]            = {};
        gNormW[pixel]           = {};
        gTangentW[pixel]        = {};
        gTexC[pixel]            = {};
        gDiffuseOpacity[pixel]  = {};
        gSpecRough[pixel]       = {};
        gEmissive[pixel]        = {};
        gMatlExtra[pixel]       = {};

        if (is_valid(gFaceNormalW))         gFaceNormalW[pixel]         = {};
        if (is_valid(gLinearZ))             gLinearZ[pixel]             = float2(kEnvMapDepth, 0.f);
        if (is_valid(gDepth))               gDepth[pixel]               = 1.f;
        if (is_valid(gMotionVectorsW))      gMotionVectorsW[pixel]      = {};
        if (is_valid(gDisocclusion))        gDisocclusion[pixel]        = 0.0f;
        if (is_valid(gNormalWRoughness))    gNormalWRoughness[pixel]    = float4(0.2215f,0.2215f,0.2215f, 0.f); // UE4 default value at miss
        if (is_valid(gVBuffer))             gVBuffer[pixel]             = {};

        // Hacky motion vectors for env map
        if (is_valid(gMotionVectors))
        {
            // Hacky motion vector computation for env map, taking the camera movement into account
            float3 worldPos = rayOrigin + normalize(rayDir) * kEnvMapDepth; // Hit on env map

            float2 pixelPos = pixel + float2(0.5f, 0.5f);
            float4 prevPosH = mul(float4(worldPos, 1.f), gScene.camera.data.prevViewProjMatNoJitter);
            float2 motionVec = calcMotionVector(pixelPos, prevPosH, frameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector
            gMotionVectors[pixel] = motionVec;
        }
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
