/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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

struct DummyAttribs { uint dummy; };

struct PathPayload
{
    uint4 packed[5];

    PackedHitInfo hit;
    GuideData guideData;
    InteriorList interiorList;  ///< Interior list. Keeping track of a stack of materials with medium properties. The size depends on the config (4B/item).
    SampleGenerator sg;         ///< Sample generator state. Typically 4-16B.

    static PathPayload pack(const PathState path)
    {
        PathPayload p = {};

        p.packed[0].xyz = asuint(path.origin);
        p.packed[0].w = path.id;

        p.packed[1].xyz = asuint(path.dir);
        p.packed[1].w = path.flagsAndVertexIndex;

        p.packed[2].xyz = asuint(path.normal);
        p.packed[2].w = uint(path.rejectedHits) | ((f32tof16(path.sceneLength) & 0xffff) << 16);

        p.packed[3].xyz = asuint(path.thp);
        p.packed[3].w = path.bounceCounters;

        p.packed[4].xyz = asuint(path.L);
        p.packed[4].w = asuint(path.pdf);

        p.hit = path.hit.getData();
        p.guideData = path.guideData;
        p.interiorList = path.interiorList;
        p.sg = path.sg;

        return p;
    }

    static PathState unpack(const PathPayload p)
    {
        PathState path = {};

        path.origin = asfloat(p.packed[0].xyz);
        path.id = p.packed[0].w;

        path.dir = asfloat(p.packed[1].xyz);
        path.flagsAndVertexIndex = p.packed[1].w;

        path.normal = asfloat(p.packed[2].xyz);
        path.rejectedHits = uint16_t(p.packed[2].w & 0xffff);
        path.sceneLength = float16_t(f16tof32(p.packed[2].w >> 16));

        path.thp = asfloat(p.packed[3].xyz);
        path.bounceCounters = p.packed[3].w;

        path.L = asfloat(p.packed[4].xyz);
        path.pdf = asfloat(p.packed[4].w);

        path.hit = HitInfo(p.hit);
        path.guideData = p.guideData;
        path.interiorList = p.interiorList;
        path.sg = p.sg;

        return path;
    }
};

struct VisibilityQuery : PathTracer::IVisibilityQuery
{
    bool traceVisibilityRay(const Ray ray)
    {
        SceneRayQuery<kUseAlphaTest> sceneRayQuery;
        return sceneRayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff);
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

    The API allows to implement the PathTracer::IClosestHitQuery interface, because
    after executing ray traversal using HitObject::TraceRay(), control is given back to the
    caller before the CHS/MS is explicitly invoked using InvokeHitObject().
    This allows for executing volume distance sampling as part of PathTracer::nextHit.
*/
struct ReorderingScheduler
{
    /** Handle a miss.
        Hit information is encoded in the payload.
        Note: This is also called for volume hits.
        \param[in,out] payload Payload data.
    */
    void handleMiss(inout PathPayload payload)
    {
        PathState path = PathPayload::unpack(payload);
        path.clearHit();
        path.sceneLength = float16_t(kNRDInvalidPathLength);

        gPathTracer.setupPathLogging(path);
        gPathTracer.handleMiss(path);

        payload = PathPayload::pack(path);
    }

    /** Handle a hit.
        \param[in,out] payload Payload data.
    */
    void handleHit(inout PathPayload payload)
    {
        PathState path = PathPayload::unpack(payload);

        gPathTracer.setupPathLogging(path);
        VisibilityQuery vq;
#if defined(DELTA_REFLECTION_PASS)
        gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
        gPathTracer.handleDeltaTransmissionHit(path);
#else
        gPathTracer.handleHit(path, vq);
#endif

        payload = PathPayload::pack(path);
    }

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path = {};

        gPathTracer.generatePath(pathID, path);
        gPathTracer.setupPathLogging(path);

        if (path.isHit())
        {
            // Handle primary hits.

            // Create a HitObject from a HitInfo.
            // The created HitObject does only contain the minimal required data,
            // most data is already passed in the path.hit object.
            // For volume hits, a miss object is returned(and handled in the handleMiss() function).
            HitObject hitObject;
            RayDesc dummyRay = {};
            DummyAttribs dummyAttribs = {};

            switch (path.hit.getType())
            {
            case HitType::None:
                hitObject = HitObject::MakeMiss(kMissScatter, dummyRay);
                break;
            case HitType::Volume:
                hitObject = HitObject::MakeMiss(kMissScatter, dummyRay);
                break;
            default:
            {
                const GeometryInstanceID instanceID = path.hit.getInstanceID();
                const GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
                const uint primitiveIndex = path.hit.getPrimitiveIndex();
                hitObject = HitObject::MakeHit(
                    gScene.rtAccel,         // AccelerationStructure
                    instance.instanceIndex, // InstanceIndex
                    instance.geometryIndex, // GeometryIndex
                    primitiveIndex,         // PrimitiveIndex
                    0,                      // HitKind
                    kRayTypeScatter,        // RayContributionToHitGroupIndex
                    rayTypeCount,           // MultiplierForGeometryContributionToHitGroupIndex
                    dummyRay,               // Ray
                    dummyAttribs            // Attributes
                );
                break;
            }
            }

            PathPayload payload = PathPayload::pack(path);
            HitObject::Invoke(gScene.rtAccel, hitObject, payload);
            path = PathPayload::unpack(payload);
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
            PathPayload payload = {};
            HitObject hitObject;

            {
                // Advance to next path vertex.
                path.incrementVertexIndex();

                // Trace ray.
                logTraceRay(PixelStatsRayType::ClosestHit);
                const Ray ray = path.getScatterRay();
                payload = PathPayload::pack(path);
                const uint rayFlags = kUseAlphaTest ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPAQUE;
                hitObject = HitObject::TraceRay( gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, rayTypeCount, kMissScatter, ray.toRayDesc(), payload );
                path = PathPayload::unpack(payload);

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

            // Reorder and invoke closest hit or miss shader.
            ReorderThread(hitObject, coherenceHints, 4);
            payload = PathPayload::pack(path);
            HitObject::Invoke(gScene.rtAccel, hitObject, payload);
            path = PathPayload::unpack(payload);
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
                if (samplesRemaining == 0) break;
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
    /** Handle a miss.
        \param[in,out] payload Payload data.
    */
    void handleMiss(inout PathPayload payload)
    {
        PathState path = PathPayload::unpack(payload);
        path.clearHit();
        path.sceneLength = float16_t(kNRDInvalidPathLength);

        gPathTracer.setupPathLogging(path);
        gPathTracer.handleMiss(path);

        payload = PathPayload::pack(path);
    }

    /** Handle a hit.
        \param[in,out] payload Payload data.
        \param[in] hit Hit info.
        \param[in] hitT Hit distance.
    */
    void handleHit(inout PathPayload payload, const HitInfo hit, const float hitT)
    {
        PathState path = PathPayload::unpack(payload);
        path.setHit(hit);
        path.sceneLength += float16_t(hitT);

        gPathTracer.setupPathLogging(path);

#if defined(DELTA_REFLECTION_PASS)
        gPathTracer.handleDeltaReflectionHit(path);
#elif defined(DELTA_TRANSMISSION_PASS)
        gPathTracer.handleDeltaTransmissionHit(path);
#else
        VisibilityQuery vq;
        gPathTracer.handleHit(path, vq);
#endif

        payload = PathPayload::pack(path);
    }

    /** Trace a ray to find the next hit.
        This replaces `PathTracer::nextHit` but without support for volume sampling.
        \param[in,out] path Path state.
    */
    void nextHit(inout PathState path)
    {
        // Advance to next path vertex.
        path.incrementVertexIndex();

        // Trace ray.
        logTraceRay(PixelStatsRayType::ClosestHit);
        const Ray ray = path.getScatterRay();
        PathPayload payload = PathPayload::pack(path);
        uint rayFlags = RAY_FLAG_NONE;
        if (!kUseAlphaTest) rayFlags |= RAY_FLAG_FORCE_OPAQUE;
        TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, kRayTypeScatter /* hitIdx */, rayTypeCount, kMissScatter /* missIdx */, ray.toRayDesc(), payload);

        path = PathPayload::unpack(payload);
    }

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path = {};

        gPathTracer.generatePath(pathID, path);
        gPathTracer.setupPathLogging(path);

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
void scatterMiss(inout PathPayload payload : SV_RayPayload)
{
    gScheduler.handleMiss(payload);
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void scatterTriangleAnyHit(inout PathPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
#endif
}

[shader("closesthit")]
void scatterTriangleClosestHit(inout PathPayload payload : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_SER
    gScheduler.handleHit(payload);
#else
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    gScheduler.handleHit(payload, HitInfo(triangleHit), RayTCurrent());
#endif
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
#if USE_SER
    gScheduler.handleHit(payload);
#else
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    gScheduler.handleHit(payload, HitInfo(displacedTriangleHit), RayTCurrent());
#endif
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
#if USE_SER
    gScheduler.handleHit(payload);
#else
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    gScheduler.handleHit(payload, HitInfo(curveHit), RayTCurrent());
#endif
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
#if USE_SER
    gScheduler.handleHit(payload);
#else
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    gScheduler.handleHit(payload, HitInfo(sdfGridHit), RayTCurrent());
#endif
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    if (all(pixel >= frameDim)) return;

    gScheduler.run(pixel);
}
