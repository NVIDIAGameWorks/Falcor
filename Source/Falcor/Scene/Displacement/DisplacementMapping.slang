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
import Scene.SceneTypes;
import Scene.Material.BasicMaterialData;
import Utils.Math.Ray;
import Utils.Geometry.IntersectionHelpers;
import Scene.Displacement.DisplacementData;

#define DISPLACEMENT_DISABLED                               0           // Debug: Used to disable displacement mapping and intersect base triangles in sw instead.
#define DISPLACEMENT_TWO_SIDED                              0           // Displaced surface is two sided. This can create some cracks.
#define DISPLACEMENT_DEBUG_DISPLAY_SHELL                    0           // Display the shell instead of displaced surface.

static const bool kShellForceMaxThickness = false;
static const bool kShellBoundsCalcUseSampleGrad = true;
static const bool kHitFaceNormalUseCentralGrad = true;
static const bool kDisplacementScalingUsePreciseLength = false;

static const float kSurfaceThickness = 0.008f;
static const int kRaymarchingMaxSampleCount = 32;
static const int kRaymarchingSampleCountFactor = 2;

/** Extension to add displacement mapping functionality to the
    DisplacementData struct.
*/
extension DisplacementData
{
    // Maps a raw value to a displacement value. Transforms raw displacement texture value into object-space distance.
    float mapValue(const float raw)
    {
        return scale * (raw + bias);
    }

    float2 mapValue(const float2 raw)
    {
        return scale * (raw + bias);
    }

    float4 mapValue(const float4 raw)
    {
        return scale * (raw + bias);
    }

    // Reads the displacement value at a given texel.
    float readValue(const float2 texelPos, const float lod = 0.f)
    {
#if DISPLACEMENT_DISABLED == 0
        return mapValue(texture.SampleLevel(samplerState, texelPos / size, lod).x);
#else
        return 0.f;
#endif
    }

    float4 readValueMinMax(const float2 texelPos, const float lod = 0.f)
    {
#if DISPLACEMENT_DISABLED == 0
        return mapValue(texture.SampleLevel(samplerState, texelPos / size, lod));
#else
        return float4(0.f);
#endif
    }

    float getConservativeGlobalExpansion()
    {
        return max(abs(getGlobalMinMax().x), abs(getGlobalMinMax().y));
    }

    float2 getGlobalMinMax()
    {
        // This is only valid if displacement map values are in [0, 1].
        // TODO: Sample largest MIP for real min/max values.
        return float2(mapValue(0.f), mapValue(1.f));
    }

    float2 getShellMinMax(const float2 texCrd0, const float2 texCrd1, const float2 texCrd2)
    {
#if DISPLACEMENT_DISABLED == 0
        if (!kShellForceMaxThickness)
        {
            float2 triangleCenterUV = (texCrd0 + texCrd1 + texCrd2) / 3.f;
            float2 localDisplacementTexVal;

            if (!kShellBoundsCalcUseSampleGrad)
            {
                float triangleLOD = getTriangleConservativeMipLevel(texCrd0, texCrd1, texCrd2);

                localDisplacementTexVal.r = texture.SampleLevel(samplerStateMin, triangleCenterUV, triangleLOD).g;
                localDisplacementTexVal.g = texture.SampleLevel(samplerStateMax, triangleCenterUV, triangleLOD).b;
            }
            else
            {
                float2 triangleUVGradX = (texCrd1 - texCrd0) * 1.1f; // x1.1 should be enough with 16x anisotropic filtering, otherwise -> x2
                float2 triangleUVGradY = (texCrd2 - texCrd0) * 1.1f;
                localDisplacementTexVal.r = texture.SampleGrad(samplerStateMin, triangleCenterUV, triangleUVGradX, triangleUVGradY).g;
                localDisplacementTexVal.g = texture.SampleGrad(samplerStateMax, triangleCenterUV, triangleUVGradX, triangleUVGradY).b;
            }

            float2 res = mapValue(localDisplacementTexVal) + kShellMinMaxMargin;
#if DISPLACEMENT_TWO_SIDED
            res += float2(-kSurfaceThickness, 0.f);
#endif
            return res;

        }
        else
        {
            return float2(mapValue(0.f), mapValue(1.f));
        }
#else
        return kShellMinMaxMargin;
#endif
    }

    float getTriangleConservativeMipLevel(const float2 texCrd0, const float2 texCrd1, const float2 texCrd2)
    {
        float2 avgUV = (texCrd0 + texCrd1 + texCrd2) / 3.f;
        float2 minUV = min(texCrd0, min(texCrd1, texCrd2));
        float2 maxUV = max(texCrd0, max(texCrd1, texCrd2));
        float2 bboxUV = max(abs(maxUV - avgUV), abs(avgUV - minUV)) * size * 2.f;
        float maxCompUV = max(bboxUV.x, bboxUV.y);
        float triangleLOD = ceil(log2(maxCompUV) + 4.f);

        return triangleLOD;
    }
};

struct DisplacementIntersection
{
    float2 barycentrics;
    float displacement;
    float t;

    float3 getBarycentricWeights()
    {
        return float3(1.f - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    }
};

bool calcDebugTriangleIntersection(const Ray ray, const float3 positions[3], out DisplacementIntersection result)
{
    result = {};

    float3 barycentrics;
    float t;
    if (intersectRayTriangle(ray.origin, ray.dir, positions, barycentrics, t))
    {
        result.barycentrics = float2(barycentrics.y, barycentrics.z);
        result.displacement = 0.f;
        result.t = t;
        return true;
    }

    return false;
}

struct IntersectAttribute
{
    bool intersected;       //< Intersection happened.

    float t;                //< Intersected t in, could be negative.
    // Triangle intersection: barycentric coordinate {u, v, 1 - u - v}.
    // Bilinear patch intersection: patch coordinate {u, v}, could use linear interpolation(u) to convert to projected.
    // Triangle barycentric, v could be used to interpolated projected height in texture space.
    float u;
    float v;

    float3 barycentric;
    float textureSpaceHeight;

    float3 hitNormal;       //< Only valid for triangle intersection.
    bool backFace;

    [mutating] bool max(const IntersectAttribute other)
    {
        if (other.intersected && (!intersected || other.t > t))
        {
            intersected = other.intersected;
            t = other.t;
            u = other.u;
            v = other.v;
            barycentric = other.barycentric;
            textureSpaceHeight = other.textureSpaceHeight;
            hitNormal = other.hitNormal;
            backFace = other.backFace;

            return true;
        }

        return false;
    }

    [mutating] bool min(const IntersectAttribute other)
    {
        if (other.intersected && (!intersected || other.t < t))
        {
            intersected = other.intersected;
            t = other.t;
            u = other.u;
            v = other.v;
            barycentric = other.barycentric;
            textureSpaceHeight = other.textureSpaceHeight;
            hitNormal = other.hitNormal;
            backFace = other.backFace;

            return true;
        }

        return false;
    }

    [mutating] void updateTriangleBarycentricAndHeight(const float texSpaceHeight)
    {
        barycentric = float3(u, v, 1.f - u - v);
        textureSpaceHeight = texSpaceHeight;
    }

    // for ray bilinear patch intersection.
    // u v means uv coord for bilinear patch.
    // we need to interpolate it as triangle barycentric.
    [mutating] void updateBilinearPatchBarycentricAndHeight(const float3 p00Barycentric, const float3 p10Barycentric, const float minHeight, const float maxHeight)
    {
        barycentric = lerp(p00Barycentric, p10Barycentric, u);
        textureSpaceHeight = lerp(minHeight, maxHeight, v);
    }
};

void rayTriangleIntersectionTest(float3 p0,
    float3 p1,
    float3 p2,
    float3 rayOrigin,
    float3 rayDirection,
    float triangleHeight,
    out IntersectAttribute intersectAttri)
{
    const float3 positions[3] = { p0, p1, p2 };
    float3 barycentrics;
    float t;
    if (intersectRayTriangle(rayOrigin, rayDirection, positions, barycentrics, t))
    {
        intersectAttri.intersected = true;
        intersectAttri.t = t;
        intersectAttri.u = barycentrics.x;
        intersectAttri.v = barycentrics.y;
        intersectAttri.hitNormal = normalize(cross(p1 - p0, p2 - p0));
        intersectAttri.backFace = dot(intersectAttri.hitNormal, rayDirection) >= 0.f;
        intersectAttri.updateTriangleBarycentricAndHeight(triangleHeight);
    }
    else
    {
        intersectAttri.intersected = false;
    }
}

// r: ray origin
// q: ray direction
// p: intersected point
float solveT(const float3 r, const float3 q, const float3 p)
{
    float t;
    if (abs(q.x) >= abs(q.y) && abs(q.x) >= abs(q.z))
    {
        t = (p.x - r.x) / q.x;
    }
    else if (abs(q.y) >= abs(q.z))
    {
        t = (p.y - r.y) / q.y;
    }
    else
    {
        t = (p.z - r.z) / q.z;
    }
    return t;
}

float copySign(const float x, const float y)
{
    return abs(x) * sign(y);
}

// Code from Alexander Reshetov : "Cool Patches: A Geometric Approach to Ray/Bilinear Patch Intersections" RTGems
// https://github.com/Apress/ray-tracing-gems/blob/master/Ch_08_Cool_Patches_A_Geometric_Approach_to_Ray-Bilinear_Patch_Intersections/rqx.cu
void rayBilinearPatchIntersectionTest(
    float3 q00,
    float3 q01,
    float3 q10,
    float3 q11,
    float3 p0Barycentric,
    float3 p1Barycentric,
    float3 rayOrigin,
    float3 rayDir,
    float rayTmax,
    float minHeight,
    float maxHeight,
    out IntersectAttribute intersectAttri0,
    out IntersectAttribute intersectAttri1)
{

    rayTmax = 1.0/0.0;     // We don't check against tmax, otherwise creates issues in some corner cases.
    intersectAttri0.intersected = false;
    intersectAttri1.intersected = false;

    float3 e10 = q10 - q00;                     // q01---------------q11
    float3 e11 = q11 - q10;                     // |                   |
    float3 e00 = q01 - q00;                     // | e00           e11 |  we precompute
    float3 qn = cross(q10 - q00, q01 - q11);    // |        e10        |  qn = cross(q10-q00,q01-q11)
    q00 -= rayOrigin;                           // q00---------------q10
    q10 -= rayOrigin;
    float a = dot(cross(q00, rayDir), e00);     // the equation is
    float c = dot(qn, rayDir);                  // a + b u + c u^2
    float b = dot(cross(q10, rayDir), e11);     // first compute a+b+c

    b -= a + c;                                 // and then b
    float det = b * b - 4.f * a * c;
    if (det < 0.f)
    {
        return;                                 // see the right part of Figure 8.5
    }

    det = sqrt(det);                            // we -use_fast_math in CUDA_NVRTC_OPTIONS

    float u1, u2;                               // two roots (u parameter)
    float t = rayTmax, u, v;                    // need solution for the smallest t > 0

    if (c == 0.f)
    {                                           // if c == 0, it is a trapezoid
        u1 = -a / b; u2 = -1.f;                 // and there is only one root
    }
    else
    {                                           // (c != 0 in Stanford models)
        u1 = (-b - copySign(det, b)) / 2.f;     // numerically "stable" root
        u2 = a / u1;                            // Viete's formula for u1*u2
        u1 /= c;
    }
    if (0.f <= u1 && u1 <= 1.f)
    {                                           // is it inside the patch?
        float3 pa = lerp(q00, q10, u1);         // point on edge e10 (Figure 8.4)
        float3 pb = lerp(e00, e11, u1);         // it is, actually, pb - pa
        float3 n = cross(rayDir, pb);
        det = dot(n, n);
        n = cross(n, pa);
        float t1 = dot(n, pb);
        float v1 = dot(n, rayDir);              // no need to check t1 < t
        if (0.f <= v1 && v1 <= det)
        {                                       // if t1 > ray.tmax it will be rejected in rtPotentialIntersection
            t = t1 / det;
            intersectAttri0.t = t1 / det;
            intersectAttri0.u = u1;
            intersectAttri0.v = v1 / det;
            intersectAttri0.backFace = false;
            intersectAttri0.updateBilinearPatchBarycentricAndHeight(p0Barycentric, p1Barycentric, minHeight, maxHeight);
            intersectAttri0.hitNormal = float3(1, 0, 0);
            intersectAttri0.intersected = true;
        }
    }
    if (0.f <= u2 && u2 <= 1.f)
    {                                           // it is slightly different,
        float3 pa = lerp(q00, q10, u2);         // since u1 might be good
        float3 pb = lerp(e00, e11, u2);         // and we need 0 < t2 < t1
        float3 n = cross(rayDir, pb);
        det = dot(n, n);
        n = cross(n, pa);
        float t2 = dot(n, pb) / det;
        float v2 = dot(n, rayDir);
        if (0.f <= v2 && v2 <= det)
        {
            if (t > t2)
            {
                if (intersectAttri0.intersected) intersectAttri1 = intersectAttri0;

                intersectAttri0.t = t2;
                intersectAttri0.u = u2;
                intersectAttri0.v = v2 / det;
                intersectAttri0.backFace = false;
                intersectAttri0.updateBilinearPatchBarycentricAndHeight(p0Barycentric, p1Barycentric, minHeight, maxHeight);
                intersectAttri0.hitNormal = float3(1, 0, 0);
                intersectAttri0.intersected = true;
            }
            else
            {
                intersectAttri1.t = t2;
                intersectAttri1.u = u2;
                intersectAttri1.v = v2 / det;
                intersectAttri1.backFace = false;
                intersectAttri1.updateBilinearPatchBarycentricAndHeight(p0Barycentric, p1Barycentric, minHeight, maxHeight);
                intersectAttri1.hitNormal = float3(1, 0, 0);
                intersectAttri1.intersected = true;
            }
        }
    }
}

bool traceHeightMapEstimated(const DisplacementData displacementData, float3 startPoint, float3 endPoint, out float intersectedT, out float intersectedHeight)
{
    const float entryHeightData = displacementData.readValue(startPoint.xy);
#if DISPLACEMENT_TWO_SIDED == 0
    if (startPoint.z < entryHeightData)
    {
        intersectedT = 0.f;             // This seems to make the intersection ignored. TODO: experiment with eg. 0.000001
        intersectedHeight = startPoint.z;
        return true;
    }
#endif

    float3 viewDirection = endPoint - startPoint;
    int maxCount = ceil(abs(viewDirection.x)) + ceil(abs(viewDirection.y)) + 2;
    maxCount = min(maxCount * kRaymarchingSampleCountFactor, kRaymarchingMaxSampleCount); // Prevent freeze due to corner cases. Todo: Investigate why this happens.

    float3 incrementUnit = viewDirection / maxCount;
    viewDirection = normalize(viewDirection);
    const float tMax = solveT(startPoint, viewDirection, endPoint);

    float3 lastTestPoint;
    float3 currentTestPoint = startPoint;
    float lastHeightDelta = currentTestPoint.z - entryHeightData;

    for (int i = 0; i < maxCount; i++)
    {
        lastTestPoint = currentTestPoint;
        currentTestPoint += incrementUnit;

        const float heightData = displacementData.readValue(currentTestPoint.xy);
        const float currentHeightDelta = currentTestPoint.z - heightData;

#if DISPLACEMENT_TWO_SIDED == 0
        if (currentHeightDelta < 0.f)
#else
        if ( ((currentHeightDelta <= 0.f) && (currentHeightDelta > -kSurfaceThickness)) || (sign(lastHeightDelta) != sign(currentHeightDelta)) )
#endif
        {
            float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
            float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);

            intersectedT = tMax * (estimatedT / maxCount);
            intersectedHeight = lerp(currentTestPoint.z, lastTestPoint.z, heightDeltaRatio);

            return true;
        }
        lastHeightDelta = currentHeightDelta;
    }

    return false;
}

bool calcDisplacementIntersection(const Ray ray, const StaticVertexData vertices[3], const DisplacementData displacementData, out DisplacementIntersection result)
{
    result = {};

    const float2 shellMinMax = displacementData.getShellMinMax(vertices[0].texCrd, vertices[1].texCrd, vertices[2].texCrd);
    const float minHeight = shellMinMax.x;
    const float maxHeight = shellMinMax.y;

    float3 extrudedP0 = vertices[0].position + vertices[0].normal * maxHeight;
    float3 extrudedP1 = vertices[1].position + vertices[1].normal * maxHeight;
    float3 extrudedP2 = vertices[2].position + vertices[2].normal * maxHeight;
    float3 intrudedP0 = vertices[0].position + vertices[0].normal * minHeight;
    float3 intrudedP1 = vertices[1].position + vertices[1].normal * minHeight;
    float3 intrudedP2 = vertices[2].position + vertices[2].normal * minHeight;

    // 1. ray "prism" intersection to grab valid start/end t for the valid segment along ray direction
    float tStart = ray.tMin;

    // extruded triangle
    IntersectAttribute extrudedTriangleIntersectAttri;
    rayTriangleIntersectionTest(extrudedP0, extrudedP1, extrudedP2, ray.origin, ray.dir, maxHeight,
        extrudedTriangleIntersectAttri);

    // min: entry point, max: exit point
    IntersectAttribute minIntersection = extrudedTriangleIntersectAttri;
    IntersectAttribute maxIntersection = extrudedTriangleIntersectAttri;

    // first slab/fin
    IntersectAttribute slabIntersectAttri0;
    IntersectAttribute slabIntersectAttri1;
    rayBilinearPatchIntersectionTest(intrudedP0, extrudedP0, intrudedP1, extrudedP1, float3(1, 0, 0), float3(0, 1, 0),
        ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri0, slabIntersectAttri1);
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // second slab/fin
    rayBilinearPatchIntersectionTest(intrudedP1, extrudedP1, intrudedP2, extrudedP2, float3(0, 1, 0), float3(0, 0, 1),
        ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri0, slabIntersectAttri1);
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // third slab/fin
    rayBilinearPatchIntersectionTest(intrudedP2, extrudedP2, intrudedP0, extrudedP0, float3(0, 0, 1), float3(1, 0, 0),
        ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri0, slabIntersectAttri1);
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // test base triangle lastly
    IntersectAttribute baseTriangleIntersectAttri;
    rayTriangleIntersectionTest(intrudedP0, intrudedP1, intrudedP2, ray.origin, ray.dir, minHeight,
        baseTriangleIntersectAttri);
    minIntersection.min(baseTriangleIntersectAttri);
    maxIntersection.max(baseTriangleIntersectAttri);

    // Early out when no hit.
    if (!(minIntersection.intersected && maxIntersection.intersected) || (minIntersection.t == maxIntersection.t) || (minIntersection.t > ray.tMax) || (maxIntersection.t < ray.tMin) )
    {
        return false;
    }

    // 2. convert intersection data from object space to trace space
    float2 minIntersectionUV = minIntersection.barycentric[0] * vertices[0].texCrd.xy + minIntersection.barycentric[1] * vertices[1].texCrd.xy + minIntersection.barycentric[2] * vertices[2].texCrd.xy;
    float2 maxIntersectionUV = maxIntersection.barycentric[0] * vertices[0].texCrd.xy + maxIntersection.barycentric[1] * vertices[1].texCrd.xy + maxIntersection.barycentric[2] * vertices[2].texCrd.xy;

    // start/end point texture space point
    float3 minIntersectionTexSpaceCoord = float3(minIntersectionUV * displacementData.size, minIntersection.textureSpaceHeight);
    float3 maxIntersectionTexSpaceCoord = float3(maxIntersectionUV * displacementData.size, maxIntersection.textureSpaceHeight);
    float validStartT = max(tStart, minIntersection.t);
    float validEndT = maxIntersection.t;
    minIntersectionTexSpaceCoord = lerp(minIntersectionTexSpaceCoord, maxIntersectionTexSpaceCoord,     (validStartT - minIntersection.t) / (maxIntersection.t - minIntersection.t));
    float3 startBarycentric = lerp(minIntersection.barycentric, maxIntersection.barycentric,            (validStartT - minIntersection.t) / (maxIntersection.t - minIntersection.t));
    float3 endBarycentric = maxIntersection.barycentric;

    // 3. trace height map in texture space
    float intersectedT = 0.f;
    float intersectedHeight = 0.f;
    bool ret = true;
#if DISPLACEMENT_DEBUG_DISPLAY_SHELL == 0
    ret = traceHeightMapEstimated(displacementData, minIntersectionTexSpaceCoord, maxIntersectionTexSpaceCoord, intersectedT, intersectedHeight);

    if (ret)
    {
        const float3 traceSpaceViewDirection = normalize(maxIntersectionTexSpaceCoord - minIntersectionTexSpaceCoord);
        // get t max in trace space.
        const float tMax = solveT(minIntersectionTexSpaceCoord, traceSpaceViewDirection, maxIntersectionTexSpaceCoord);
        // intersectedT is in trace space, interpolate barycentric in there.
        float3 intersectionProjectedBarycentric = lerp(startBarycentric, endBarycentric, intersectedT / tMax);
        result.barycentrics = float2(intersectionProjectedBarycentric.y, 1.f - intersectionProjectedBarycentric.x - intersectionProjectedBarycentric.y);  //intersectionProjectedBarycentric;
        result.displacement = intersectedHeight;
        result.t = lerp(validStartT, validEndT, intersectedT / tMax);

        return result.t > ray.tMin && result.t < ray.tMax;
    }
#else
    IntersectAttribute debugIntersection = maxIntersection;
    if (debugIntersection.intersected)
    {
        result.t = debugIntersection.t;
        result.barycentrics = float2(debugIntersection.barycentric.y, 1.f - debugIntersection.barycentric.x - debugIntersection.barycentric.y);
        result.displacement = 0.f;
        return result.t > ray.tMin && result.t < ray.tMax;
    }
#endif

    return false;
}

/** Intersect a displaced triangle.
    \param[in] ray Ray in world space.
    \param[in] vertices Triangle vertices in object space.
    \param[in] worldMat Triangle object to world transform.
    \param[in] displacementData Displacement data.
    \param[out] result Intersection result (only valid if triangle is intersected).
    \return Returns true if displaced triangle is intersected.
*/
bool intersectDisplacedTriangle(const Ray ray, StaticVertexData vertices[3], const float4x4 worldMat, const DisplacementData displacementData, out DisplacementIntersection result)
{
    result = {};

    float3 vertPosTr[3];
    vertPosTr[0] = mul(worldMat, float4(vertices[0].position, 1.f)).xyz;
    vertPosTr[1] = mul(worldMat, float4(vertices[1].position, 1.f)).xyz;
    vertPosTr[2] = mul(worldMat, float4(vertices[2].position, 1.f)).xyz;
    // Compute real-length normals.
    vertices[0].normal = mul(worldMat, float4(vertices[0].position + vertices[0].normal, 1.f)).xyz - vertPosTr[0];
    vertices[1].normal = mul(worldMat, float4(vertices[1].position + vertices[1].normal, 1.f)).xyz - vertPosTr[1];
    vertices[2].normal = mul(worldMat, float4(vertices[2].position + vertices[2].normal, 1.f)).xyz - vertPosTr[2];
    vertices[0].position = vertPosTr[0];
    vertices[1].position = vertPosTr[1];
    vertices[2].position = vertPosTr[2];
#if DISPLACEMENT_DISABLED == 0
    bool valid = calcDisplacementIntersection(ray, vertices, displacementData, result);
    // Transform displacement into WS distance.
    if (!kDisplacementScalingUsePreciseLength)
    {
        result.displacement *= (length(vertices[0].normal) + length(vertices[1].normal) + length(vertices[2].normal)) / 3.f;
    }
    else
    {
        // Proper normal interpolation.
        const float3 barycentrics = result.getBarycentricWeights();
        float3 interpNormal = (vertices[0].normal * barycentrics[0] + vertices[1].normal * barycentrics[1] + vertices[2].normal * barycentrics[2]);
        result.displacement *= length(interpNormal);
    }
#else
    float3 positions[3] = { vertices[0].position, vertices[1].position, vertices[2].position };
    bool valid = calcDebugTriangleIntersection(ray, positions, result);
#endif

    return valid;
}

/** Helper struct for interpolating triangle vertices.
    TODO: This should be moved inside 'computeDisplacedTriangleNormal' when supported in slang.
*/
struct TriangleVertex
{
    float3 position;
    float3 normal;
    float2 texCrd;

    static TriangleVertex interpolate(const TriangleVertex vertices[3], const float3 barycentrics, const bool normalizeNormal = true)
    {
        TriangleVertex res = {
            barycentrics[0] * vertices[0].position + barycentrics[1] * vertices[1].position + barycentrics[2] * vertices[2].position,
            barycentrics[0] * vertices[0].normal + barycentrics[1] * vertices[1].normal + barycentrics[2] * vertices[2].normal,
            barycentrics[0] * vertices[0].texCrd + barycentrics[1] * vertices[1].texCrd + barycentrics[2] * vertices[2].texCrd
        };

        if (normalizeNormal)
        {
            res.normal = normalize(res.normal);
        }
        return res;
    }
};

/** Compute displaced triangle normal.
    \param[in] barycentrics Barycentric weights.
    \param[in] vertices Triangle vertices in object space.
    \param[in] worldMat Triangle object to world transform.
    \param[in] worldInvTransposeMat Triangle object to world inverse transpose transform.
    \param[in] displacementData Displacement data.
    \return Returns the displaced normal in world space.
*/
float3 computeDisplacedTriangleNormal(const float3 barycentrics, const StaticVertexData vertices[3], const float4x4 worldMat, const float3x3 worldInvTransposeMat, const DisplacementData displacementData)
{
    // Transform vertices to world space.
    const TriangleVertex verticesW[3] = {
        mul(worldMat, float4(vertices[0].position, 1.f)).xyz, mul(worldMat, float4(vertices[0].position + vertices[0].normal, 1.f)).xyz - mul(worldMat, float4(vertices[0].position, 1.f)).xyz, vertices[0].texCrd,
        mul(worldMat, float4(vertices[1].position, 1.f)).xyz, mul(worldMat, float4(vertices[1].position + vertices[1].normal, 1.f)).xyz - mul(worldMat, float4(vertices[1].position, 1.f)).xyz, vertices[1].texCrd,
        mul(worldMat, float4(vertices[2].position, 1.f)).xyz, mul(worldMat, float4(vertices[2].position + vertices[2].normal, 1.f)).xyz - mul(worldMat, float4(vertices[2].position, 1.f)).xyz, vertices[2].texCrd
    };

    float2 baryDelta = float2(1.0) / float2(length((vertices[1].texCrd - (vertices[0].texCrd + vertices[2].texCrd) * 0.5) * displacementData.size), length((vertices[2].texCrd - (vertices[0].texCrd + vertices[1].texCrd) * 0.5) * displacementData.size));

    const float sampleLOD = 0.5;

    const float3 ptDuBary = barycentrics + float3(baryDelta.x, 0.0, -baryDelta.x);
    TriangleVertex vPtDu = TriangleVertex::interpolate(verticesW, ptDuBary, false);
    vPtDu.position = vPtDu.position + vPtDu.normal * displacementData.readValue(vPtDu.texCrd * displacementData.size, sampleLOD);

    const float3 ptDvBary = barycentrics + float3(0.0, baryDelta.y, -baryDelta.y);
    TriangleVertex vPtDv = TriangleVertex::interpolate(verticesW, ptDvBary, false);
    vPtDv.position = vPtDv.position + vPtDv.normal * displacementData.readValue(vPtDv.texCrd * displacementData.size, sampleLOD);

    if (kHitFaceNormalUseCentralGrad)
    {
        const float3 ptDuBaryN = barycentrics + float3(-baryDelta.x, 0.0, baryDelta.x);
        TriangleVertex vPtDuN = TriangleVertex::interpolate(verticesW, ptDuBaryN, false);
        vPtDuN.position = vPtDuN.position + vPtDuN.normal * displacementData.readValue(vPtDuN.texCrd * displacementData.size, sampleLOD);

        const float3 ptDvBaryN = barycentrics + float3(0.0, -baryDelta.y, baryDelta.y);
        TriangleVertex vPtDvN = TriangleVertex::interpolate(verticesW, ptDvBaryN, false);
        vPtDvN.position = vPtDvN.position + vPtDvN.normal * displacementData.readValue(vPtDvN.texCrd * displacementData.size, sampleLOD);

        const float3 posVecDu = (vPtDu.position - vPtDuN.position);
        const float3 posVecDv = (vPtDv.position - vPtDvN.position);

        return normalize(cross(posVecDu, posVecDv));
    }
    else
    {
        TriangleVertex vPt = TriangleVertex::interpolate(verticesW, barycentrics, false);
        vPt.position = vPt.position + vPt.normal * displacementData.readValue(vPt.texCrd * displacementData.size, sampleLOD);

        return normalize(cross(vPtDu.position - vPt.position, vPtDv.position - vPt.position));
    }
}

