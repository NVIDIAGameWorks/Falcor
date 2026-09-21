/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "Scene/SceneDefines.slangh"
import Scene.Raytracing;
import Scene.RaytracingInline; // For visibility queries.
import Scene.Intersection;
import Rendering.Utils.PixelStats;
import RenderPasses.Shared.Denoising.NRDConstants;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import PathTracer;
import PathTracerNRD;

import Utils.NVAPI; // TODO(@skallweit) this is currently needed to make g_NvidiaExt visible

ParameterBlock<PathTracer> gPathTracer;
#if USE_SER
ReorderingScheduler gScheduler;
#else
Scheduler gScheduler;
#endif

// Define ray indices.
static const uint kRayTypeScatter = 0;
static const uint kMissScatter = 0;

/** Minimal ray payload carrying only hit information.
    All path state (PathState, including ReSTIRPTData) lives in raygen registers.
*/
struct PathPayload
{
    PackedHitInfo hit = {};
    float hitT = 0.f;
};

struct VisibilityQuery : IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray);
    }
};

/** Helper function to create a HitInfo from a HitObject.
 */
HitInfo makeHitInfo(const HitObject hitObject)
{
    HitInfo hit = {};

    if (hitObject.IsHit())
    {
        const GeometryInstanceID instanceID = GeometryInstanceID(hitObject.GetInstanceID(), hitObject.GetGeometryIndex());
        const uint primitiveIndex = hitObject.GetPrimitiveIndex();

        // Lookup geometry type as we cannot reconstruct that from the HitObject directly.
        const GeometryType type = gScene.getGeometryInstanceType(instanceID);

        switch (type)
        {
        case GeometryType::None:
            break;
        case GeometryType::TriangleMesh:
        {
            TriangleHit triangleHit;
            triangleHit.instanceID = instanceID;
            triangleHit.primitiveIndex = primitiveIndex;
            BuiltInTriangleIntersectionAttributes attribs = hitObject.GetAttributes<BuiltInTriangleIntersectionAttributes>();
            triangleHit.barycentrics = attribs.barycentrics;
            hit = HitInfo(triangleHit);
            break;
        }
        case GeometryType::DisplacedTriangleMesh:
        {
            DisplacedTriangleHit displacedTriangleHit;
            displacedTriangleHit.instanceID = instanceID;
            displacedTriangleHit.primitiveIndex = primitiveIndex;
            DisplacedTriangleMeshIntersector.Attribs attribs = hitObject.GetAttributes<DisplacedTriangleMeshIntersector.Attribs>();
            displacedTriangleHit.barycentrics = attribs.barycentrics;
            displacedTriangleHit.displacement = attribs.displacement;
            hit = HitInfo(displacedTriangleHit);
            break;
        }
        case GeometryType::Curve:
        {
            CurveHit curveHit;
            curveHit.instanceID = instanceID;
            curveHit.primitiveIndex = primitiveIndex;
            CurveIntersector.Attribs attribs = hitObject.GetAttributes<CurveIntersector.Attribs>();
            curveHit.barycentrics = attribs.barycentrics;
            hit = HitInfo(curveHit);
            break;
        }
        case GeometryType::SDFGrid:
        {
            SDFGridHit sdfGridHit;
            sdfGridHit.instanceID = instanceID;
            SDFGridHitData attribs = hitObject.GetAttributes<SDFGridHitData>();
            sdfGridHit.hitData = attribs;
            hit = HitInfo(sdfGridHit);
            break;
        }
        case GeometryType::Custom:
            // Unsupported.
            break;
        }
    }

    return hit;
}

/** Reordering scheduler using SER/HitObject API.

    Uses HitObject::TraceRay() for traversal and ReorderThread() for coherence.
    All path processing happens directly in raygen after reordering, avoiding
    the need to pack/unpack path state through CHS/MS payloads.
*/
struct ReorderingScheduler
{
    /** Process a path vertex (hit or miss) after thread reordering.
        \param[in,out] path Path state.
    */
    void processVertex(inout PathState path)
    {
        gPathTracer.setupPathLogging(path);
        if (path.isHit())
        {
#if defined(DELTA_REFLECTION_PASS)
            gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
            gPathTracer.handleDeltaTransmissionHit(path);
#else
            VisibilityQuery vq;
            gPathTracer.handleHit(path, vq);
#endif
        }
        else
        {
            gPathTracer.handleMiss(path);
        }
    }

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path = {};

        gPathTracer.generatePath(pathID, path);
        gPathTracer.setupPathLogging(path);

#if USE_RESTIRPT && !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
        const bool processPrimaryMiss = gPathTracer.restirpt.kUseAreaReservoirs;
#else
        const bool processPrimaryMiss = false;
#endif

        if (path.isHit())
        {
            // Handle primary hit directly in raygen (no HitObject::Invoke needed).
#if defined(DELTA_REFLECTION_PASS)
            gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
            gPathTracer.handleDeltaTransmissionHit(path);
#else
            VisibilityQuery vq;
            gPathTracer.handleHit(path, vq);
#endif
        }
        else if (processPrimaryMiss)
        {
            // For ReSTIR PT area reservoirs, handle primary miss here so that the
            // environment emission contributes to the area reservoir and the result
            // is written via writeOutput below.
            path.setVertexIndex(1);
            gPathTracer.handleMiss(path);
        }
        else
        {
            // Note the primary miss has already been handled by the separate path generation pass
            // the path tracer runs first. Abort execution here to avoid double work.
            return;
        }

        // Trace path.
        while (path.isActive())
        {
            HitObject hitObject;

            {
                // Advance to next path vertex.
                path.incrementVertexIndex();

                // Trace ray with minimal payload (only carries hit info back).
                logTraceRay(PixelStatsRayType::ClosestHit);
                const Ray ray = path.getScatterRay();
                PathPayload payload = {};
                const uint rayFlags = kUseAlphaTest ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPAQUE;
                hitObject = HitObject::TraceRay(
                    gScene.rtAccel,
                    rayFlags,
                    0xff /* instanceInclusionMask */,
                    kRayTypeScatter /* hitIdx */,
                    getRayTypeCount(),
                    kMissScatter,
                    ray.toRayDesc(),
                    payload
                );

                if (hitObject.IsHit())
                {
                    path.setHit(makeHitInfo(hitObject));
                    path.sceneLength += float16_t(hitObject.GetRayDesc().TMax);
                }
                else
                {
                    path.clearHit();
                    path.sceneLength = float16_t(kNRDInvalidPathLength);
                }
            }

            // Add path coherence hints.
            const uint materialID = gScene.getMaterialID(path.hit.getInstanceID());
            const MaterialHeader materialHeader = gScene.materials.getMaterialHeader(materialID);
            uint coherenceHints = gPathTracer.getCoherenceHints(path, materialHeader.isDeltaSpecular(), materialHeader.isEmissive());

            // Reorder threads for coherence, then process hit/miss directly.
            ReorderThread(hitObject, coherenceHints, 4);
            processVertex(path);
        }

#if !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
        gPathTracer.writeOutput(path);
#endif
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        // Determine number of samples to take.
        uint samplesRemaining = kSamplesPerPixel;
        if (kSamplesPerPixel == 0)
        {
            samplesRemaining = gPathTracer.sampleCount[pixel];
        }

        // Loop over samples.
        while (samplesRemaining > 0)
        {
            samplesRemaining -= 1;
            uint pathID = pixel.x | (pixel.y << 12) | (samplesRemaining << 24);
            tracePath(pathID);

            // Use SER to compact active threads.
            if (kSamplesPerPixel == 0)
            {
                bool needReorder = WaveActiveAnyTrue(samplesRemaining == 0);
                if (samplesRemaining == 0)
                    break;
                if (needReorder)
                {
                    HitObject hitObject = HitObject::MakeNop();
                    ReorderThread(hitObject);
                }
            }
        }
    }
}

/** Simple scheduler using a fullscreen raytracing dispatch.
 */
struct Scheduler
{
    /** Trace a ray to find the next hit and process it.
        Path state lives in raygen registers; only hit info travels through the payload.
        \param[in,out] path Path state.
    */
    void nextHit(inout PathState path)
    {
        // Advance to next path vertex.
        path.incrementVertexIndex();

        // Trace ray with minimal payload.
        logTraceRay(PixelStatsRayType::ClosestHit);
        const Ray ray = path.getScatterRay();
        PathPayload payload = {};
        uint rayFlags = RAY_FLAG_NONE;
        if (!kUseAlphaTest)
            rayFlags |= RAY_FLAG_FORCE_OPAQUE;
        TraceRay(
            gScene.rtAccel,
            rayFlags,
            0xff /* instanceInclusionMask */,
            kRayTypeScatter /* hitIdx */,
            getRayTypeCount(),
            kMissScatter /* missIdx */,
            ray.toRayDesc(),
            payload
        );

        // Recover hit info from the minimal payload.
        HitInfo hit = unpackHitInfo(payload.hit);
        if (hit.isValid())
        {
            path.setHit(hit);
            path.sceneLength += float16_t(payload.hitT);
        }
        else
        {
            path.clearHit();
            path.sceneLength = float16_t(kNRDInvalidPathLength);
        }

        // Process the vertex directly in raygen.
        gPathTracer.setupPathLogging(path);
        if (hit.isValid())
        {
#if defined(DELTA_REFLECTION_PASS)
            gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
            gPathTracer.handleDeltaTransmissionHit(path);
#else
            VisibilityQuery vq;
            gPathTracer.handleHit(path, vq);
#endif
        }
        else
        {
            gPathTracer.handleMiss(path);
        }
    }

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path = {};

        gPathTracer.generatePath(pathID, path);
        gPathTracer.setupPathLogging(path);

#if USE_RESTIRPT && !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
        const bool processPrimaryMiss = gPathTracer.restirpt.kUseAreaReservoirs;
#else
        const bool processPrimaryMiss = false;
#endif

        // Handle primary hit/miss.
        if (path.isHit())
        {
#if defined(DELTA_REFLECTION_PASS)
            gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
            gPathTracer.handleDeltaTransmissionHit(path);
#else
            VisibilityQuery vq;
            gPathTracer.handleHit(path, vq);
#endif
        }
        else if (processPrimaryMiss)
        {
            // For ReSTIR PT area reservoirs, handle primary miss here so that the
            // environment emission contributes to the area reservoir and the result
            // is written via writeOutput below.
            path.setVertexIndex(1);
            gPathTracer.handleMiss(path);
        }
        else
        {
            // Note the primary miss has already been handled by the separate path generation pass
            // the path tracer runs first. Abort execution here to avoid double work.
            return;
        }

        // Trace path.
        while (path.isActive())
        {
            nextHit(path);
        }

#if !defined(DELTA_REFLECTION_PASS) && !defined(DELTA_TRANSMISSION_PASS)
        gPathTracer.writeOutput(path);
#endif
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        // Determine number of samples to take.
        uint samplesRemaining = kSamplesPerPixel;
        if (kSamplesPerPixel == 0)
        {
            samplesRemaining = gPathTracer.sampleCount[pixel];
        }

        // Loop over samples.
        while (samplesRemaining > 0)
        {
            samplesRemaining -= 1;
            uint pathID = pixel.x | (pixel.y << 12) | (samplesRemaining << 24);
            tracePath(pathID);
        }
    }
}

//
// Shader entry point for miss shaders.
//

[shader("miss")]
void scatterMiss(inout PathPayload payload: SV_RayPayload)
{
    // No-op: default-initialized payload (PackedHitInfo == 0) already indicates miss.
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void scatterTriangleAnyHit(
    inout PathPayload payload: SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes
)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
#endif
}

[shader("closesthit")]
void scatterTriangleClosestHit(
    inout PathPayload payload: SV_RayPayload,
    BuiltInTriangleIntersectionAttributes attribs: SV_IntersectionAttributes
)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    payload.hit = HitInfo(triangleHit).pack();
    payload.hitT = RayTCurrent();
}

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout PathPayload payload, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    payload.hit = HitInfo(displacedTriangleHit).pack();
    payload.hitT = RayTCurrent();
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterCurveClosestHit(inout PathPayload payload, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    payload.hit = HitInfo(curveHit).pack();
    payload.hitT = RayTCurrent();
}


//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void scatterSdfGridClosestHit(inout PathPayload payload, SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    payload.hit = HitInfo(sdfGridHit).pack();
    payload.hitT = RayTCurrent();
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    if (all(pixel >= frameDim))
        return;

    gScheduler.run(pixel);
}
