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
#include "SceneDefines.slangh"

__exported import Scene.HitInfo;
__exported import Scene.SceneTypes;
__exported import Scene.Camera.Camera;
__exported import Scene.Lights.LightData;
__exported import Scene.Lights.LightCollection;
__exported import Scene.Lights.LightProfile;
__exported import Scene.Lights.EnvMap;
__exported import Scene.Material.MaterialSystem;
__exported import Scene.Volume.GridVolume;
__exported import Scene.SDFs.SDFGrid;

import Utils.Attributes;
import Utils.Math.MathHelpers;
import Utils.Math.MatrixUtils;
import Utils.Geometry.GeometryHelpers;
import Scene.SDFs.SDFVoxelCommon;
import Scene.Displacement.DisplacementMapping;
import Rendering.Materials.TexLODHelpers;

#ifndef SCENE_GEOMETRY_TYPES
// This error occurs when a shader imports Scene.slang without setting the defines
// returned by Scene::getSceneDefines().
#error "SCENE_GEOMETRY_TYPES not defined!"
#endif

/** Scene data required for rendering.
*/
struct Scene
{
#ifdef SCENE_USE_LIGHT_PROFILE // DEMO21
    static bool kUseLightProfile = SCENE_USE_LIGHT_PROFILE;
#else
    static bool kUseLightProfile = false;
#endif


    // Raytracing
    [root] RaytracingAccelerationStructure rtAccel;

    // Instance transforms
    [root] StructuredBuffer<float4x4> worldMatrices;
    [root] StructuredBuffer<float4x4> inverseTransposeWorldMatrices; // TODO: Make this 3x3 matrices (stored as 4x3). See #795.
    StructuredBuffer<float4x4> prevWorldMatrices;
    StructuredBuffer<float4x4> prevInverseTransposeWorldMatrices;

    // Instances
    [root] StructuredBuffer<GeometryInstanceData> geometryInstances;

    // Triangle meshes
    StructuredBuffer<MeshDesc> meshes;

    [root] StructuredBuffer<PackedStaticVertexData> vertices;       ///< Vertex data for this frame.
    StructuredBuffer<PrevVertexData> prevVertices;                  ///< Vertex data for the previous frame, for dynamic meshes only.
#if SCENE_HAS_INDEXED_VERTICES
    [root] ByteAddressBuffer indexData;                             ///< Vertex indices, three indices per triangle packed tightly. The format is specified per mesh.
#endif

    // Curves
    StructuredBuffer<CurveDesc> curves;

    StructuredBuffer<StaticCurveVertexData> curveVertices;
    StructuredBuffer<DynamicCurveVertexData> prevCurveVertices;
    ByteAddressBuffer curveIndices;

    // SDF Grids
#if SCENE_SDF_GRID_COUNT > 0
    SDFGrid sdfGrids[SCENE_SDF_GRID_COUNT];
#else
    SDFGrid sdfGrids[1]; // Zero-length arrays are not supported.
#endif

    // Procedural primitives
    StructuredBuffer<AABB> proceduralPrimitiveAABBs;                ///< Global list of AABBs for procedural primitives (curves, custom primitives, etc.).

    // Custom primitives
    uint customPrimitiveInstanceOffset;
    uint customPrimitiveInstanceCount;
    uint customPrimitiveAABBOffset;
    StructuredBuffer<CustomPrimitiveDesc> customPrimitives;

    // Materials
    ParameterBlock<MaterialSystem> materials;

    // Lights and camera
    uint lightCount;
    StructuredBuffer<LightData> lights;
    LightCollection lightCollection;
    LightProfile lightProfile;
    EnvMap envMap;
    Camera camera;

    // Grid volumes
    uint gridVolumeCount;
    StructuredBuffer<GridVolume> gridVolumes;
#if SCENE_GRID_COUNT > 0
    Grid grids[SCENE_GRID_COUNT];
#else
    Grid grids[1]; // Zero-length arrays are not supported.
#endif

    // Geometry access

    /** Return the geometry type for a given global geometry instance ID.
        \param[in] instanceID Global geometry instance ID.
        \return The type of geometry this instance represents.
    */
    GeometryType getGeometryInstanceType(const GeometryInstanceID instanceID)
    {
        if (instanceID.index >= customPrimitiveInstanceOffset) return GeometryType::Custom;
        return geometryInstances[instanceID.index].getType();
    }

    // Mesh and instance data access

    float4x4 loadWorldMatrix(const uint matrixID)
    {
        return worldMatrices[matrixID];
    }

    float4x4 loadPrevWorldMatrix(const uint matrixID)
    {
        return prevWorldMatrices[matrixID];
    }

    float3x3 loadInverseTransposeWorldMatrix(const uint matrixID)
    {
        return float3x3(inverseTransposeWorldMatrices[matrixID]);
    }

    float3x3 loadPrevInverseTransposeWorldMatrix(const uint matrixID)
    {
        return float3x3(prevInverseTransposeWorldMatrices[matrixID]);
    }

    float4x4 getWorldMatrix(const GeometryInstanceID instanceID)
    {
        uint matrixID = geometryInstances[instanceID.index].globalMatrixID;
        return loadWorldMatrix(matrixID);
    }

    float4x4 getPrevWorldMatrix(const GeometryInstanceID instanceID)
    {
        uint matrixID = geometryInstances[instanceID.index].globalMatrixID;
        return loadPrevWorldMatrix(matrixID);
    }

    float3x3 getInverseTransposeWorldMatrix(const GeometryInstanceID instanceID)
    {
        uint matrixID = geometryInstances[instanceID.index].globalMatrixID;
        return loadInverseTransposeWorldMatrix(matrixID);
    }

    float3x3 getPrevInverseTransposeWorldMatrix(const GeometryInstanceID instanceID)
    {
        uint matrixID = geometryInstances[instanceID.index].globalMatrixID;
        return loadPrevInverseTransposeWorldMatrix(matrixID);
    }

    bool isWorldMatrixFlippedWinding(const GeometryInstanceID instanceID)
    {
        return (geometryInstances[instanceID.index].flags & uint(GeometryInstanceFlags::TransformFlipped)) != 0;
    }

    bool isObjectFrontFaceCW(const GeometryInstanceID instanceID)
    {
        return (geometryInstances[instanceID.index].flags & uint(GeometryInstanceFlags::IsObjectFrontFaceCW)) != 0;
    }

    bool isWorldFrontFaceCW(const GeometryInstanceID instanceID)
    {
        return (geometryInstances[instanceID.index].flags & uint(GeometryInstanceFlags::IsWorldFrontFaceCW)) != 0;
    }

    GeometryInstanceData getGeometryInstance(const GeometryInstanceID instanceID)
    {
        return geometryInstances[instanceID.index];
    }

    MeshDesc getMeshDesc(const GeometryInstanceID instanceID)
    {
        return meshes[geometryInstances[instanceID.index].geometryID];
    }

    // Materials access

    /** Return the material ID for a geometry instance.
        \param[in] instanceID Global geometry instance ID.
        \return Material ID.
    */
    uint getMaterialID(const GeometryInstanceID instanceID)
    {
        return geometryInstances[instanceID.index].materialID;
    };

    // Lights access

    uint getLightCount()
    {
        return lightCount;
    }

    LightData getLight(uint lightIndex)
    {
        return lights[lightIndex];
    }

    // Volume access

    uint getGridCount()
    {
        return SCENE_GRID_COUNT;
    }

    void getGrid(const uint gridIndex, out Grid grid)
    {
        grid = grids[gridIndex];
    }

    uint getGridVolumeCount()
    {
        return gridVolumeCount;
    }

    GridVolume getGridVolume(const uint volumeIndex)
    {
        return gridVolumes[volumeIndex];
    }

    // Geometry access

    /** Returns the local vertex indices for a given triangle.
        \param[in] ibOffset Offset into the index buffer.
        \param[in] triangleIndex Index of the triangle.
        \param[in] use16Bit Flag indicating that the indices are stored in 16-bit format.
        \return Vertex indices local to the current mesh.
    */
    uint3 getLocalIndices(const uint ibOffset, const uint triangleIndex, const bool use16Bit)
    {
#if SCENE_HAS_INDEXED_VERTICES
        // Determine what format of the indices.
        // It's only if the scene has mixed formats that we incur the cost of checking the flag bit.
#if SCENE_HAS_16BIT_INDICES && SCENE_HAS_32BIT_INDICES
        bool read16Bit = use16Bit;
#elif SCENE_HAS_16BIT_INDICES
        bool read16Bit = true;
#elif SCENE_HAS_32BIT_INDICES
        bool read16Bit = false;
#else
        #error Scene vertex formats are not specified
#endif
        // Load the vertex indices.
        uint baseIndex = ibOffset * 4;
        uint3 vtxIndices;
        if (read16Bit)
        {
            baseIndex += triangleIndex * 6;
            vtxIndices = (uint3)indexData.Load<uint16_t3>(baseIndex);
        }
        else
        {
            baseIndex += triangleIndex * 12;
            vtxIndices = indexData.Load3(baseIndex);
        }
#else // !SCENE_HAS_INDEXED_VERTICES
        uint baseIndex = triangleIndex * 3;
        uint3 vtxIndices = { baseIndex, baseIndex + 1, baseIndex + 2 };
#endif
        return vtxIndices;
    }

    /** Returns the global vertex indices for a given triangle.
        \param[in] meshID Mesh ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \return Vertex indices into the global vertex buffer.
    */
    uint3 getIndices(const uint meshID, const uint triangleIndex)
    {
        const MeshDesc mesh = meshes[meshID];
        uint3 vtxIndices = getLocalIndices(mesh.ibOffset, triangleIndex, mesh.flags & uint(MeshFlags::Use16BitIndices));
        vtxIndices += mesh.vbOffset;
        return vtxIndices;
    }

    /** Returns the global vertex indices for a given triangle.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \return Vertex indices into the global vertex buffer.
    */
    uint3 getIndices(const GeometryInstanceID instanceID, const uint triangleIndex)
    {
        const GeometryInstanceData instance = getGeometryInstance(instanceID);
        uint3 vtxIndices = getLocalIndices(instance.ibOffset, triangleIndex, instance.flags & uint(GeometryInstanceFlags::Use16BitIndices));
        vtxIndices += instance.vbOffset;
        return vtxIndices;
    }

    /** Returns vertex data for a vertex.
        \param[in] index Global vertex index.
        \return Vertex data.
    */
    StaticVertexData getVertex(const uint index)
    {
        return vertices[index].unpack();
    }

    /** Returns a triangle's face normal in object space.
        \param[in] vertices Unpacked fetched vertices which can be used for further computations involving individual vertices.
        \param[in] isFrontFaceCW True if front-facing side has clockwise winding in object space.
        \param[out] Face normal in object space (normalized).
    */
    float3 getFaceNormalInObjectSpace(const StaticVertexData vertices[3], const bool isFrontFaceCW)
    {
        float3 p0 = vertices[0].position;
        float3 p1 = vertices[1].position;
        float3 p2 = vertices[2].position;
        float3 N = normalize(cross(p1 - p0, p2 - p0));
        return isFrontFaceCW ? -N : N;
    }

    /** Returns a triangle's face normal in world space.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] Face normal in world space (normalized).
    */
    float3 getFaceNormalW(const GeometryInstanceID instanceID, const uint triangleIndex)
    {
        uint3 vtxIndices = getIndices(instanceID, triangleIndex);
        float3 p0 = vertices[vtxIndices[0]].position;
        float3 p1 = vertices[vtxIndices[1]].position;
        float3 p2 = vertices[vtxIndices[2]].position;
        float3 N = cross(p1 - p0, p2 - p0);
        if (isObjectFrontFaceCW(instanceID)) N = -N;
        float3x3 worldInvTransposeMat = getInverseTransposeWorldMatrix(instanceID);
        return normalize(mul(worldInvTransposeMat, N));
    }

    /** Returns a triangle's area in world space.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] Triangle area.
    */
    float getFaceAreaW(const GeometryInstanceID instanceID, const uint triangleIndex)
    {
        float3 p[3];
        getVertexPositionsW(instanceID, triangleIndex, p);
        return 0.5f * length(cross(p[1] - p[0], p[2] - p[0]));
    }

    /** Computes the face normal and area of a triangle given its vertices.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] p Position of vertex 0,1,2 in world space.
        \param[out] triangleArea Triangle area in world space units.
        \return Face normal in world space (normalized).
    */
    float3 computeFaceNormalAndAreaW(const GeometryInstanceID instanceID, const float3 p[3], out float triangleArea)
    {
        // Compute face normal in world space.
        // The length of the vector is twice the triangle area since we're in world space.
        // Note that this is not true if the normal is transformed using the inverse-transpose.
        float3 e[2];
        e[0] = p[1] - p[0];
        e[1] = p[2] - p[0];
        float3 N = cross(e[0], e[1]);
        triangleArea = 0.5f * length(N);

        // Flip the normal depending on final winding order in world space.
        if (isWorldFrontFaceCW(instanceID)) N = -N;

        return normalize(N);
    }

    /** Returns a triangle's face normal and area in world space.
        This function should only be used if the triangle area is needed, as it is less efficient than computing just its normal.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] triangleArea Triangle area in world space units.
        \return Face normal in world space (normalized).
    */
    float3 getFaceNormalAndAreaW(const GeometryInstanceID instanceID, const uint triangleIndex, out float triangleArea)
    {
        uint3 vtxIndices = getIndices(instanceID, triangleIndex);

        // Load vertices and transform to world space.
        float3 p[3];
        [unroll]
        for (int i = 0; i < 3; i++)
        {
            p[i] = vertices[vtxIndices[i]].position;
            p[i] = mul(getWorldMatrix(instanceID), float4(p[i], 1.f)).xyz;
        }

        return computeFaceNormalAndAreaW(instanceID, p, triangleArea);
    }

    /** Interpolated vertex attributes for a given triangle and transform to world space.
        \param[in] vertices Unpacked fetched vertices which can be used for further computations involving individual vertices.
        \param[in] worldMat Triangle object to world transform.
        \param[in] worldInvTransposeMat Triangle object to world inverse transpose transform.
        \param[in] isFrontFaceCW True if front-facing side has clockwise winding in object space.
        \param[in] barycentrics Barycentric coordinates in the triangle.
        \return Interpolated vertex attributes.
    */
    VertexData computeVertexData(const StaticVertexData vertices[3], const float4x4 worldMat, const float3x3 worldInvTransposeMat, const bool isFrontFaceCW, const float3 barycentrics)
    {
        VertexData v = {};

        v.posW = vertices[0].position * barycentrics[0];
        v.posW += vertices[1].position * barycentrics[1];
        v.posW += vertices[2].position * barycentrics[2];

        v.normalW = vertices[0].normal * barycentrics[0];
        v.normalW += vertices[1].normal * barycentrics[1];
        v.normalW += vertices[2].normal * barycentrics[2];

        v.tangentW = vertices[0].tangent * barycentrics[0];
        v.tangentW += vertices[1].tangent * barycentrics[1];
        v.tangentW += vertices[2].tangent * barycentrics[2];
        v.tangentW.w = sign(v.tangentW.w); // Preserve zero to indicate invalid tangent.

        v.texC = vertices[0].texCrd * barycentrics[0];
        v.texC += vertices[1].texCrd * barycentrics[1];
        v.texC += vertices[2].texCrd * barycentrics[2];

        v.curveRadius = vertices[0].curveRadius * barycentrics[0];
        v.curveRadius += vertices[1].curveRadius * barycentrics[1];
        v.curveRadius += vertices[2].curveRadius * barycentrics[2];

        v.faceNormalW = getFaceNormalInObjectSpace(vertices, isFrontFaceCW);

        v.posW = mul(worldMat, float4(v.posW, 1.f)).xyz;
        v.normalW = mul(worldInvTransposeMat, v.normalW);
        v.faceNormalW = mul(worldInvTransposeMat, v.faceNormalW);
        v.tangentW.xyz = mul((float3x3)worldMat, v.tangentW.xyz);
        v.curveRadius = length(mul((float3x3)worldMat, float3(v.curveRadius, 0.f, 0.f)));

        v.normalW = normalize(v.normalW);
        v.faceNormalW = normalize(v.faceNormalW);
        v.tangentW.xyz = normalize(v.tangentW.xyz);
        return v;
    }

    /** Returns the interpolated vertex attributes for a given triangle.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[in] barycentrics Barycentric coordinates in the triangle.
        \param[out] Unpacked fetched vertices which can be used for further computations involving individual vertices.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexData(const GeometryInstanceID instanceID, const uint triangleIndex, const float3 barycentrics, out StaticVertexData vertices[3])
    {
        const uint3 vtxIndices = getIndices(instanceID, triangleIndex);
        vertices = { gScene.getVertex(vtxIndices[0]), gScene.getVertex(vtxIndices[1]), gScene.getVertex(vtxIndices[2]) };

        const float4x4 worldMat = gScene.getWorldMatrix(instanceID);
        const float3x3 worldInvTransposeMat = getInverseTransposeWorldMatrix(instanceID);
        const bool isFrontFaceCW = isObjectFrontFaceCW(instanceID);
        return computeVertexData(vertices, worldMat, worldInvTransposeMat, isFrontFaceCW, barycentrics);
    }

    /** Returns the interpolated vertex attributes for a given triangle.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[in] barycentrics Barycentric coordinates in the triangle.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexData(const GeometryInstanceID instanceID, const uint triangleIndex, const float3 barycentrics)
    {
        StaticVertexData vertices[3];
        return getVertexData(instanceID, triangleIndex, barycentrics, vertices);
    }

    /** Returns the interpolated vertex attributes for a given hitpoint.
        \param[in] hit Triangle hit info.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexData(const TriangleHit hit)
    {
        return getVertexData(hit.instanceID, hit.primitiveIndex, hit.getBarycentricWeights());
    }

    /** Returns the interpolated vertex attributes for a given hitpoint.
        \param[in] hit Triangle hit info.
        \param[out] Unpacked fetched vertices which can be used for further computations involving individual vertices.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexData(const TriangleHit hit, out StaticVertexData vertices[3])
    {
        return getVertexData(hit.instanceID, hit.primitiveIndex, hit.getBarycentricWeights(), vertices);
    }

    /** Returns the interpolated vertex attributes for a given hitpoint.
        \param[in] hit Displaced triangle hit info.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexData(const DisplacedTriangleHit hit, const float3 viewDir)
    {
        const uint3 vtxIndices = getIndices(hit.instanceID, hit.primitiveIndex);
        const StaticVertexData vertices[3] = { gScene.getVertex(vtxIndices[0]), gScene.getVertex(vtxIndices[1]), gScene.getVertex(vtxIndices[2]) };
        const float3 barycentrics = hit.getBarycentricWeights();
        const float4x4 worldMat = gScene.getWorldMatrix(hit.instanceID);
        const float3x3 worldInvTransposeMat = getInverseTransposeWorldMatrix(hit.instanceID);
        const bool isFrontFaceCW = isObjectFrontFaceCW(hit.instanceID);
        VertexData v = computeVertexData(vertices, worldMat, worldInvTransposeMat, isFrontFaceCW, barycentrics);

        // Offset surface along the displaced direction to avoid self-intersections because of precision.
        v.posW += v.normalW * (hit.displacement * DisplacementData::kSurfaceSafetyScaleBias.x + DisplacementData::kSurfaceSafetyScaleBias.y);

        // Load displacement data.
        const uint materialID = getMaterialID(hit.instanceID);
        DisplacementData displacementData;
        materials.loadDisplacementData(materialID, displacementData);

        // Update face normal according to local displacement map values.
        // This is only an approximation in order to get plausible displaced face and smooth normals.
        float3 adjustedFaceNormalW = computeDisplacedTriangleNormal(barycentrics, vertices, worldMat, worldInvTransposeMat, displacementData);

        float dpVN = dot(viewDir, adjustedFaceNormalW);
        if (dpVN < 0.f)
        {
            // Keep the normal normal with viewDir
            adjustedFaceNormalW = normalize(adjustedFaceNormalW + viewDir * -(dpVN * 1.0001f));
        }

        float3 faceNormalDelta = adjustedFaceNormalW - v.faceNormalW;
        v.normalW = normalize(v.normalW + faceNormalDelta);
        v.faceNormalW = adjustedFaceNormalW;

        return v;
    }


    /** Returns interpolated vertex attributes in a ray tracing hit program when ray cones are used for texture LOD.
       \param[in] instanceID Geometry instance ID of the mesh.
       \param[in] triangleIndex Index of the triangle in the given mesh.
       \param[in] barycentrics Barycentric coordinates in the triangle.
       \return Interpolated vertex attributes.
   */
    VertexData getVertexDataRayCones(const GeometryInstanceID instanceID, const uint triangleIndex, const float3 barycentrics)
    {
        StaticVertexData triangleVertices[3];
        VertexData v = getVertexData(instanceID, triangleIndex, barycentrics, triangleVertices);
        v.coneTexLODValue = computeRayConeTriangleLODValue(triangleVertices, float3x3(getWorldMatrix(instanceID)));
        return v;
    }

    /** Returns interpolated position in world space for the previous frame.
        \param[in] hit Hit info.
        \return Interpolated position in world space for the previous frame.
    */
    float3 getPrevPosW(const HitInfo hit)
    {
        switch (hit.getType())
        {
        case HitType::Triangle:
            return gScene.getPrevPosW(hit.getTriangleHit());
        case HitType::DisplacedTriangle:
            return gScene.getPrevPosW(hit.getDisplacedTriangleHit());
        case HitType::Curve:
            return gScene.getPrevPosWFromCurve(hit.getCurveHit());
        default:
            return 0.f;
        }
    }

    /** Returns interpolated position on a triangle in world space for the previous frame.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[in] barycentrics Barycentric coordinates in the triangle.
        \return Interpolated position in world space for the previous frame.
    */
    float3 getPrevPosW(const GeometryInstanceID instanceID, const uint triangleIndex, const float3 barycentrics)
    {
        float3 prevPos = {};

        const GeometryInstanceData instance = getGeometryInstance(instanceID);
        uint3 vtxIndices = getLocalIndices(instance.ibOffset, triangleIndex, instance.flags & uint(GeometryInstanceFlags::Use16BitIndices));

        if (instance.isDynamic())
        {
            // For dynamic meshes, the previous position is stored in a separate buffer.
            vtxIndices += meshes[instance.geometryID].prevVbOffset;

            prevPos += prevVertices[vtxIndices[0]].position * barycentrics[0];
            prevPos += prevVertices[vtxIndices[1]].position * barycentrics[1];
            prevPos += prevVertices[vtxIndices[2]].position * barycentrics[2];
        }
        else
        {
            // For non-dynamic meshes, the previous positions are the same as the current.
            vtxIndices += instance.vbOffset;

            prevPos += vertices[vtxIndices[0]].position * barycentrics[0];
            prevPos += vertices[vtxIndices[1]].position * barycentrics[1];
            prevPos += vertices[vtxIndices[2]].position * barycentrics[2];
        }

        const float4x4 prevWorldMat = loadPrevWorldMatrix(instance.globalMatrixID);
        return mul(prevWorldMat, float4(prevPos, 1.f)).xyz;
    }

    /** Returns interpolated position on a triangle in world space for the previous frame.
        \param[in] hit Triangle hit info.
        \return Interpolated position in world space for the previous frame.
    */
    float3 getPrevPosW(const TriangleHit hit)
    {
        return getPrevPosW(hit.instanceID, hit.primitiveIndex, hit.getBarycentricWeights());
    }

    /** Returns interpolated position on a displaced triangle in world space for the previous frame.
        \param[in] hit Displaced triangle hit info.
        \return Interpolated position in world space for the previous frame.
    */
    float3 getPrevPosW(const DisplacedTriangleHit hit)
    {
        float3 prevPos = {};
        float3 prevNormal = {};

        const float3 barycentrics = hit.getBarycentricWeights();
        const GeometryInstanceData instance = getGeometryInstance(hit.instanceID);
        uint3 vtxIndices = getLocalIndices(instance.ibOffset, hit.primitiveIndex, instance.flags & uint(GeometryInstanceFlags::Use16BitIndices));

        // For non-dynamic meshes, the previous position/normal is the same as the current.
        vtxIndices += instance.vbOffset;

        prevPos += vertices[vtxIndices[0]].position * barycentrics[0];
        prevPos += vertices[vtxIndices[1]].position * barycentrics[1];
        prevPos += vertices[vtxIndices[2]].position * barycentrics[2];

        prevNormal += vertices[vtxIndices[0]].unpack().normal * barycentrics[0];
        prevNormal += vertices[vtxIndices[1]].unpack().normal * barycentrics[1];
        prevNormal += vertices[vtxIndices[2]].unpack().normal * barycentrics[2];

        // Offset surface along the displaced direction to avoid self-intersections because of precision.
        prevPos += prevNormal * (hit.displacement * DisplacementData::kSurfaceSafetyScaleBias.x + DisplacementData::kSurfaceSafetyScaleBias.y);

        const float4x4 prevWorldMat = loadPrevWorldMatrix(instance.globalMatrixID);
        return mul(prevWorldMat, float4(prevPos, 1.f)).xyz;
    }


    /** Returns a triangle's vertex positions in world space.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] p Position of vertex 0,1,2 in world space.
    */
    void getVertexPositionsW(const GeometryInstanceID instanceID, const uint triangleIndex, out float3 p[3])
    {
        uint3 vtxIndices = getIndices(instanceID, triangleIndex);
        float4x4 worldMat = getWorldMatrix(instanceID);

        [unroll]
        for (int i = 0; i < 3; i++)
        {
            p[i] = vertices[vtxIndices[i]].position;
            p[i] = mul(worldMat, float4(p[i], 1.f)).xyz;
        }
    }

    /** Returns a triangle's texture coordinates.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] texC Texture coordinate of vertex 0,1,2.
    */
    void getVertexTexCoords(const GeometryInstanceID instanceID, const uint triangleIndex, out float2 texC[3])
    {
        uint3 vtxIndices = getIndices(instanceID, triangleIndex);

        [unroll]
        for (int i = 0; i < 3; i++)
        {
            texC[i] = vertices[vtxIndices[i]].texCrd;
        }
    }

    // Curve access

    CurveDesc getCurveDesc(const GeometryInstanceID instanceID)
    {
        return curves[geometryInstances[instanceID.index].geometryID];
    }

    /** Returns the global curve vertex indices for the first control point of a curve segment.
        \param[in] instanceID Geometry instance ID of the curve.
        \param[in] curgeSegIndex Index of the curve segment in the given curve.
        \return Index of the first control point into the global curve vertex buffer.
    */
    uint getFirstCurveVertexIndex(const GeometryInstanceID instanceID, const uint curveSegIndex)
    {
        uint baseIndex = geometryInstances[instanceID.index].ibOffset + curveSegIndex;
        uint vertexIndex = curveIndices.Load(baseIndex * 4);
        vertexIndex += geometryInstances[instanceID.index].vbOffset;
        return vertexIndex;
    }

    /** Returns curve vertex data.
        \param[in] index Global curve vertex index.
        \return Curve vertex data.
    */
    StaticCurveVertexData getCurveVertex(const uint index)
    {
        return curveVertices[index];
    }

    /** Returns the interpolated vertex attributes for a curve segment.
        \param[in] instanceID Geometry instance ID of the curve.
        \param[in] curveSegIndex Index of the curve segment in the given curve.
        \param[in] uv UV parameters (between 0 and 1) of the ray-curve hit point.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexDataFromCurve(const GeometryInstanceID instanceID, const uint curveSegIndex, const float2 uv)
    {
        const uint v0Index = getFirstCurveVertexIndex(instanceID, curveSegIndex);
        VertexData v = {};

        StaticCurveVertexData vertices[2] = { getCurveVertex(v0Index), getCurveVertex(v0Index + 1) };

        // Note that worldMat should have an isotropic scale component.
        // Otherwise the curve endcaps might not be spheres.
        const float4x4 worldMat = getWorldMatrix(instanceID);
        const float4 sphereA = transformSphere(float4(vertices[0].position, vertices[0].radius), worldMat);
        const float4 sphereB = transformSphere(float4(vertices[1].position, vertices[1].radius), worldMat);

        float3 hitPos, sphereCenter;
        uvToCurveHitPos(uv, sphereA, sphereB, hitPos, sphereCenter);

        v.posW = hitPos;
        v.normalW = hitPos - sphereCenter;
        float curveRadius = length(v.normalW);
        v.curveRadius = curveRadius;
        v.normalW /= curveRadius;
        v.faceNormalW = v.normalW;

        // Note that the tangent direction is not perpendicular to the normal direction.
        // It will be fixed by computeTangentSpace() in Scene.Shading later.
        v.tangentW = float4(normalize(sphereB.xyz - sphereA.xyz), 1.f);

        // All curve segments in a strand share the same texture coordinates (i.e., determined at the root of the strand).
        v.texC = vertices[0].texCrd * (1.f - uv.x) + vertices[1].texCrd * uv.x;

        return v;
    }

    /** Returns the interpolated vertex attributes for a given hitpoint on curves.
        \param[in] hit Curve hit info.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexDataFromCurve(const CurveHit hit)
    {
        return getVertexDataFromCurve(hit.instanceID, hit.primitiveIndex, hit.barycentrics);
    }

    /** Returns interpolated position on a curve in world space for the previous frame.
        \param[in] instanceID Geometry instance ID of the curve.
        \param[in] curveSegIndex Index of the curve segment in the given curve.
        \param[in] uv UV parameters (between 0 and 1) of the ray-curve hit point.
        \return Interpolated position in world space for the previous frame.
    */
    float3 getPrevPosWFromCurve(const GeometryInstanceID instanceID, const uint curveSegIndex, const float2 uv)
    {
        const uint v0Index = getFirstCurveVertexIndex(instanceID, curveSegIndex);
        VertexData v = {};

        float3 positions[2] = { prevCurveVertices[v0Index].position, prevCurveVertices[v0Index + 1].position };
        float radii[2] = { getCurveVertex(v0Index).radius, getCurveVertex(v0Index + 1).radius };

        // Note that prevWorldMat should have an isotropic scale component.
        // Otherwise the curve endcaps might not be spheres.
        const float4x4 prevWorldMat = getPrevWorldMatrix(instanceID);
        const float4 sphereA = transformSphere(float4(positions[0], radii[0]), prevWorldMat);
        const float4 sphereB = transformSphere(float4(positions[1], radii[1]), prevWorldMat);

        float3 hitPos, sphereCenter;
        uvToCurveHitPos(uv, sphereA, sphereB, hitPos, sphereCenter);
        return hitPos;
    }

    /** Returns interpolated position on a curve in world space for the previous frame.
        \param[in] hit Curve hit info.
        \return Interpolated position in world space for the previous frame.
    */
    float3 getPrevPosWFromCurve(const CurveHit hit)
    {
        return getPrevPosWFromCurve(hit.instanceID, hit.primitiveIndex, hit.barycentrics);
    }

    // SDF grid access

    void getSDFGrid(const GeometryInstanceID instanceID, out SDFGrid sdfGrid)
    {
        sdfGrid = sdfGrids[geometryInstances[instanceID.index].geometryID];
    }

    /** Returns the interpolated vertex attributes for an SDF grid.
    \param[in] hit SDF grid hit info.
    \param[in] posW The world position of the hit point.
    \return Interpolated vertex attributes.
    */
    VertexData getVertexDataFromSDFGrid(const SDFGridHit hit, float3 posW)
    {
        VertexData v;

        const GeometryInstanceData instance = geometryInstances[hit.instanceID.index];
        const SDFGrid sdfGrid = sdfGrids[instance.geometryID];

        // Load Matrices.
        const float4x4 worldMat = loadWorldMatrix(instance.globalMatrixID);
        const float3x3 worldInvTransposeMat = loadInverseTransposeWorldMatrix(instance.globalMatrixID);

        // Multiplying v.M to perform the intended (M^T).v
        const float3 hitPosition = mul(posW - extractTranslate(worldMat), worldInvTransposeMat);

        float3 gradient;
        sdfGrid.decodeHit(hit.hitData, hitPosition, gradient);

        v.posW = posW;
        v.normalW = normalize(mul(worldInvTransposeMat, normalize(gradient)));
        v.tangentW = float4(perp_stark(v.normalW), 1.0f);
        v.faceNormalW = v.normalW;

        v.texC = float2(0.0f);
        v.coneTexLODValue = 0.0f;

        return v;
    }

    /** Returns the interpolated vertex attributes for an SDF grid.
    \param[in] hit SDF grid hit info.
    \param[in] rayOrigin The origin of the ray that led to the hit.
    \param[in] rayDir The normalized direction of the ray that led to the hit.
    \return Interpolated vertex attributes.
    */
    VertexData getVertexDataFromSDFGrid(const SDFGridHit hit, float3 rayOrigin, float3 rayDir)
    {
        return getVertexDataFromSDFGrid(hit, rayOrigin + rayDir * hit.hitData.hitT);
    }

    /** Returns world position of a point on an SDF grid for the previous frame.
        \param[in] hit SDF grid hit info.
        \param[in] posW SDF The world position of the point for the current frame.
        \return Position in world space for the previous frame.
    */
    float3 getPrevPosWFromSDFGrid(const SDFGridHit hit, float3 posW)
    {
        const GeometryInstanceData instance = geometryInstances[hit.instanceID.index];

        // Load Matrices.
        const float4x4 worldMat = loadWorldMatrix(instance.globalMatrixID);
        const float3x3 worldInvTransposeMat = loadInverseTransposeWorldMatrix(instance.globalMatrixID);
        const float4x4 prevWorldMat = loadPrevWorldMatrix(instance.globalMatrixID);

        /// Multiplying v.M to perform the intended (M^T).v
        const float4 hitPosition = float4(mul(posW - extractTranslate(worldMat), worldInvTransposeMat), 1.0f);

        return mul(prevWorldMat, hitPosition).xyz;
    }

    // Custom primitive access

    uint getCustomPrimitiveIndex(const GeometryInstanceID instanceID)
    {
        return instanceID.index - customPrimitiveInstanceOffset;
    }

    AABB getCustomPrimitiveAABB(const GeometryInstanceID instanceID)
    {
        // There is one AABB per custom primitive so we can directly index into the global AABB list.
        uint customPrimitiveIndex = getCustomPrimitiveIndex(instanceID);
        uint index = customPrimitiveAABBOffset + customPrimitiveIndex;
        return proceduralPrimitiveAABBs[index];
    }

    // Curvature helpers

    /** Returns the estimated curvature from a triangle vertex attributes for ray tracing using the provided estimator.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[in] curvatureEstimator Generic triangle curvature estimator to use.
        \return Estimated curvature.
    */
    float computeCurvatureGeneric<TCE : ITriangleCurvatureEstimator>(const GeometryInstanceID instanceID, const uint triangleIndex, const TCE curvatureEstimator)
    {
        const uint3 vtxIndices = getIndices(instanceID, triangleIndex);
        StaticVertexData vertices[3] = { getVertex(vtxIndices[0]), getVertex(vtxIndices[1]), getVertex(vtxIndices[2]) };
        float3 normals[3];
        float3 pos[3];
        normals[0] = vertices[0].normal;
        normals[1] = vertices[1].normal;
        normals[2] = vertices[2].normal;
        pos[0] = vertices[0].position;
        pos[1] = vertices[1].position;
        pos[2] = vertices[2].position;
        float3 edge01 = pos[1] - pos[0];
        float3 edge02 = pos[2] - pos[0];
        float3 edge12 = pos[2] - pos[1];
        float curvature01 = dot(normals[1] - normals[0], edge01) / dot(edge01, edge01);
        float curvature02 = dot(normals[2] - normals[0], edge02) / dot(edge02, edge02);
        float curvature12 = dot(normals[2] - normals[1], edge12) / dot(edge12, edge12);

        return curvatureEstimator.eval(edge01, edge02, edge12, curvature01, curvature02, curvature12);
    }

    /** Returns the estimated isoctropic curvature from vertex attributes for ray tracing.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \return Estimated curvature.
    */
    float computeCurvatureIsotropic(const GeometryInstanceID instanceID, const uint triangleIndex)
    {
        TriangleCurvature_Average tce = { };
        //TriangleCurvature_Max tce = { };          // Over-estimates curvature using max of the 3 edges.
        return computeCurvatureGeneric(instanceID, triangleIndex, tce);
    }

    /** Returns the estimated visible curvature from vertex attributes for ray tracing using the ellipse intersection of a cone with the surface.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \return Estimated curvature.
    */
    float computeCurvatureVisibleEllipse(const GeometryInstanceID instanceID, const uint triangleIndex, const float3 rayDir, const float rayConeWidth, const float rayConeAngle)
    {
        TriangleCurvature_EllipseVis tce = { rayDir, rayConeWidth, rayConeAngle };
        return computeCurvatureGeneric(instanceID, triangleIndex, tce);
    }

    /** Returns the estimated isoctropic curvature from vertex attributes for ray tracing at the first hit point.
        \param[in] instanceID Geometry instance ID of the mesh.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[in] rayDir Ray direction.
        \return Estimated curvature.
    */
    float computeCurvatureIsotropicFirstHit(const GeometryInstanceID instanceID, const uint triangleIndex, const float3 rayDir)
    {
        TriangleCurvature_DirClosestDP tce = { rayDir };
        return computeCurvatureGeneric(instanceID, triangleIndex, tce);
    }
};

ParameterBlock<Scene> gScene;
