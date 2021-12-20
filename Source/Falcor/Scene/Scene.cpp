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
#include "stdafx.h"
#include "Scene.h"
#include "SceneDefines.slangh"

#include <sstream>
#include <numeric>

namespace Falcor
{
    static_assert(sizeof(MeshDesc) % 16 == 0, "MeshDesc size should be a multiple of 16");
    static_assert(sizeof(GeometryInstanceData) == 32, "GeometryInstanceData size should be 32");
    static_assert(sizeof(PackedStaticVertexData) % 16 == 0, "PackedStaticVertexData size should be a multiple of 16");

    namespace
    {
        // Large scenes are split into multiple BLAS groups in order to reduce build memory usage.
        // The target is max 0.5GB intermediate memory per BLAS group. Note that this is not a strict limit.
        const size_t kMaxBLASBuildMemory = 1ull << 29;

        const std::string kParameterBlockName = "gScene";
        const std::string kGeometryInstanceBufferName = "geometryInstances";
        const std::string kMeshBufferName = "meshes";
        const std::string kIndexBufferName = "indexData";
        const std::string kVertexBufferName = "vertices";
        const std::string kPrevVertexBufferName = "prevVertices";
        const std::string kProceduralPrimAABBBufferName = "proceduralPrimitiveAABBs";
        const std::string kCurveBufferName = "curves";
        const std::string kCurveIndexBufferName = "curveIndices";
        const std::string kCurveVertexBufferName = "curveVertices";
        const std::string kPrevCurveVertexBufferName = "prevCurveVertices";
        const std::string kSDFGridsArrayName = "sdfGrids";
        const std::string kCustomPrimitiveBufferName = "customPrimitives";
        const std::string kMaterialsBlockName = "materials";
        const std::string kLightsBufferName = "lights";
        const std::string kGridVolumesBufferName = "gridVolumes";

        const std::string kStats = "stats";
        const std::string kBounds = "bounds";
        const std::string kAnimations = "animations";
        const std::string kLoopAnimations = "loopAnimations";
        const std::string kCamera = "camera";
        const std::string kCameras = "cameras";
        const std::string kCameraSpeed = "cameraSpeed";
        const std::string kLights = "lights";
        const std::string kAnimated = "animated";
        const std::string kRenderSettings = "renderSettings";
        const std::string kUpdateCallback = "updateCallback";
        const std::string kEnvMap = "envMap";
        const std::string kMaterials = "materials";
        const std::string kGridVolumes = "gridVolumes";
        const std::string kGetLight = "getLight";
        const std::string kGetMaterial = "getMaterial";
        const std::string kGetGridVolume = "getGridVolume";
        const std::string kSetEnvMap = "setEnvMap";
        const std::string kAddViewpoint = "addViewpoint";
        const std::string kRemoveViewpoint = "kRemoveViewpoint";
        const std::string kSelectViewpoint = "selectViewpoint";

        // Checks if the transform flips the coordinate system handedness (its determinant is negative).
        bool doesTransformFlip(const glm::mat4& m)
        {
            return glm::determinant((glm::mat3)m) < 0.f;
        }
    }

    const FileDialogFilterVec& Scene::getFileExtensionFilters()
    {
        return Importer::getFileExtensionFilters();
    }

    Scene::Scene(SceneData&& sceneData)
    {
        // Copy/move scene data to member variables.
        mFilename = sceneData.filename;
        mRenderSettings = sceneData.renderSettings;
        mCameras = std::move(sceneData.cameras);
        mSelectedCamera = sceneData.selectedCamera;
        mCameraSpeed = sceneData.cameraSpeed;
        mLights = std::move(sceneData.lights);

        mpMaterials = std::move(sceneData.pMaterials);
        mGridVolumes = std::move(sceneData.gridVolumes);
        mGrids = std::move(sceneData.grids);
        mpEnvMap = sceneData.pEnvMap;
        mSceneGraph = std::move(sceneData.sceneGraph);
        mMetadata = std::move(sceneData.metadata);

        // Merge all geometry instance lists into one.
        mGeometryInstanceData.reserve(sceneData.meshInstanceData.size() + sceneData.curveInstanceData.size() + sceneData.sdfGridInstances.size());
        mGeometryInstanceData.insert(std::end(mGeometryInstanceData), std::begin(sceneData.meshInstanceData), std::end(sceneData.meshInstanceData));
        mGeometryInstanceData.insert(std::end(mGeometryInstanceData), std::begin(sceneData.curveInstanceData), std::end(sceneData.curveInstanceData));
        mGeometryInstanceData.insert(std::end(mGeometryInstanceData), std::begin(sceneData.sdfGridInstances), std::end(sceneData.sdfGridInstances));

        mMeshDesc = std::move(sceneData.meshDesc);
        mMeshNames = std::move(sceneData.meshNames);
        mMeshBBs = std::move(sceneData.meshBBs);
        mMeshIdToInstanceIds = std::move(sceneData.meshIdToInstanceIds);
        mMeshGroups = std::move(sceneData.meshGroups);

        mUseCompressedHitInfo = sceneData.useCompressedHitInfo;
        mHas16BitIndices = sceneData.has16BitIndices;
        mHas32BitIndices = sceneData.has32BitIndices;

        mCurveDesc = std::move(sceneData.curveDesc);
        mCurveBBs = std::move(sceneData.curveBBs);
        mCurveIndexData = std::move(sceneData.curveIndexData);
        mCurveStaticData = std::move(sceneData.curveStaticData);

        mSDFGrids = std::move(sceneData.sdfGrids);
        mSDFGridDesc = std::move(sceneData.sdfGridDesc);
        mSDFGridMaxLODCount = std::move(sceneData.sdfGridMaxLODCount);

        mCustomPrimitiveDesc = std::move(sceneData.customPrimitiveDesc);
        mCustomPrimitiveAABBs = std::move(sceneData.customPrimitiveAABBs);

        // Setup additional resources.
        mFrontClockwiseRS[RasterizerState::CullMode::None] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(false).setCullMode(RasterizerState::CullMode::None));
        mFrontClockwiseRS[RasterizerState::CullMode::Back] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(false).setCullMode(RasterizerState::CullMode::Back));
        mFrontClockwiseRS[RasterizerState::CullMode::Front] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(false).setCullMode(RasterizerState::CullMode::Front));
        mFrontCounterClockwiseRS[RasterizerState::CullMode::None] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(true).setCullMode(RasterizerState::CullMode::None));
        mFrontCounterClockwiseRS[RasterizerState::CullMode::Back] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(true).setCullMode(RasterizerState::CullMode::Back));
        mFrontCounterClockwiseRS[RasterizerState::CullMode::Front] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(true).setCullMode(RasterizerState::CullMode::Front));

        // Setup volume grid -> id map.
        for (size_t i = 0; i < mGrids.size(); ++i) mGridIDs.emplace(mGrids[i], (uint32_t)i);

        // Set default SDF grid config.
        setSDFGridConfig();

        // Create vertex array objects for meshes and curves.
        createMeshVao(sceneData.meshDrawCount, sceneData.meshIndexData, sceneData.meshStaticData, sceneData.meshDynamicData);
        createCurveVao(mCurveIndexData, mCurveStaticData);

        // Create animation controller.
        mpAnimationController = AnimationController::create(this, sceneData.meshStaticData, sceneData.meshDynamicData, sceneData.animations);

        // Must be placed after curve data/AABB creation.
        mpAnimationController->addAnimatedVertexCaches(std::move(sceneData.cachedCurves), std::move(sceneData.cachedMeshes));

        // Finalize scene.
        finalize();
    }

    Scene::SharedPtr Scene::create(const std::string& filename)
    {
        return SceneBuilder::create(filename)->getScene();
    }

    Scene::SharedPtr Scene::create(SceneData&& sceneData)
    {
        return Scene::SharedPtr(new Scene(std::move(sceneData)));
    }

    Shader::DefineList Scene::getDefaultSceneDefines()
    {
        Shader::DefineList defines;
        defines.add("SCENE_GEOMETRY_TYPES", "0");
        defines.add("SCENE_GRID_COUNT", "0");
        defines.add("SCENE_SDF_GRID_COUNT", "0");
        defines.add("SCENE_HAS_INDEXED_VERTICES", "0");
        defines.add("SCENE_HAS_16BIT_INDICES", "0");
        defines.add("SCENE_HAS_32BIT_INDICES", "0");

        defines.add(MaterialSystem::getDefaultDefines());

        return defines;
    }

    Shader::DefineList Scene::getSceneDefines() const
    {
        Shader::DefineList defines;
        defines.add("SCENE_GEOMETRY_TYPES", std::to_string((uint32_t)mGeometryTypes));
        defines.add("SCENE_GRID_COUNT", std::to_string(mGrids.size()));
        defines.add("SCENE_HAS_INDEXED_VERTICES", hasIndexBuffer() ? "1" : "0");
        defines.add("SCENE_HAS_16BIT_INDICES", mHas16BitIndices ? "1" : "0");
        defines.add("SCENE_HAS_32BIT_INDICES", mHas32BitIndices ? "1" : "0");

        defines.add(mHitInfo.getDefines());
        defines.add(mpMaterials->getDefines());
        defines.add(getSceneSDFGridDefines());

        return defines;
    }

    Shader::DefineList Scene::getSceneSDFGridDefines() const
    {
        Shader::DefineList defines;
        defines.add("SCENE_SDF_GRID_COUNT", std::to_string(mSDFGrids.size()));
        defines.add("SCENE_SDF_GRID_MAX_LOD_COUNT", std::to_string(mSDFGridMaxLODCount));

        defines.add("SCENE_SDF_GRID_IMPLEMENTATION_NDSDF", std::to_string((uint32_t)SDFGrid::Type::NormalizedDenseGrid));
        defines.add("SCENE_SDF_GRID_IMPLEMENTATION_SVS", std::to_string((uint32_t)SDFGrid::Type::SparseVoxelSet));
        defines.add("SCENE_SDF_GRID_IMPLEMENTATION_SBS", std::to_string((uint32_t)SDFGrid::Type::SparseBrickSet));
        defines.add("SCENE_SDF_GRID_IMPLEMENTATION_SVO", std::to_string((uint32_t)SDFGrid::Type::SparseVoxelOctree));

        defines.add("SCENE_SDF_NO_INTERSECTION_METHOD", std::to_string((uint32_t)SDFGridIntersectionMethod::None));
        defines.add("SCENE_SDF_NO_VOXEL_SOLVER", std::to_string((uint32_t)SDFGridIntersectionMethod::GridSphereTracing));
        defines.add("SCENE_SDF_VOXEL_SPHERE_TRACING", std::to_string((uint32_t)SDFGridIntersectionMethod::VoxelSphereTracing));

        defines.add("SCENE_SDF_NO_GRADIENT_EVALUATION_METHOD", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::None));
        defines.add("SCENE_SDF_GRADIENT_NUMERIC_DISCONTINUOUS", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::NumericDiscontinuous));
        defines.add("SCENE_SDF_GRADIENT_NUMERIC_CONTINUOUS", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::NumericContinuous));

        defines.add("SCENE_SDF_GRID_IMPLEMENTATION", std::to_string((uint32_t)mSDFGridConfig.implementation));
        defines.add("SCENE_SDF_VOXEL_INTERSECTION_METHOD", std::to_string((uint32_t)mSDFGridConfig.intersectionMethod));
        defines.add("SCENE_SDF_GRADIENT_EVALUATION_METHOD", std::to_string((uint32_t)mSDFGridConfig.gradientEvaluationMethod));
        defines.add("SCENE_SDF_SOLVER_MAX_ITERATION_COUNT", std::to_string(mSDFGridConfig.solverMaxIterations));
        defines.add("SCENE_SDF_OPTIMIZE_VISIBILITY_RAYS", mSDFGridConfig.optimizeVisibilityRays ? "1" : "0");

        return defines;
    }

    Program::TypeConformanceList Scene::getTypeConformances() const
    {
        return mpMaterials->getTypeConformances();
    }

    const LightCollection::SharedPtr& Scene::getLightCollection(RenderContext* pContext)
    {
        if (!mpLightCollection)
        {
            mpLightCollection = LightCollection::create(pContext, shared_from_this());
            mpLightCollection->setShaderData(mpSceneBlock["lightCollection"]);

            mSceneStats.emissiveMemoryInBytes = mpLightCollection->getMemoryUsageInBytes();
        }
        return mpLightCollection;
    }

    void Scene::rasterize(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars, RasterizerState::CullMode cullMode)
    {
        rasterize(pContext, pState, pVars, mFrontClockwiseRS[cullMode], mFrontCounterClockwiseRS[cullMode]);
    }

    void Scene::rasterize(RenderContext* pContext, GraphicsState* pState, GraphicsVars* pVars, const RasterizerState::SharedPtr& pRasterizerStateCW, const RasterizerState::SharedPtr& pRasterizerStateCCW)
    {
        FALCOR_PROFILE("rasterizeScene");

        pVars->setParameterBlock(kParameterBlockName, mpSceneBlock);

        auto pCurrentRS = pState->getRasterizerState();
        bool isIndexed = hasIndexBuffer();

        for (const auto& draw : mDrawArgs)
        {
            assert(draw.count > 0);

            // Set state.
            pState->setVao(draw.ibFormat == ResourceFormat::R16Uint ? mpVao16Bit : mpVao);

            if (draw.ccw) pState->setRasterizerState(pRasterizerStateCCW);
            else pState->setRasterizerState(pRasterizerStateCW);

            // Draw the primitives.
            if (isIndexed)
            {
                pContext->drawIndexedIndirect(pState, pVars, draw.count, draw.pBuffer.get(), 0, nullptr, 0);
            }
            else
            {
                pContext->drawIndirect(pState, pVars, draw.count, draw.pBuffer.get(), 0, nullptr, 0);
            }
        }

        pState->setRasterizerState(pCurrentRS);
    }

    uint32_t Scene::getRaytracingMaxAttributeSize() const
    {
        bool hasDisplacedMesh = hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh);
        if (hasDisplacedMesh) return 12;


        return 8;
    }

    void Scene::raytrace(RenderContext* pContext, RtProgram* pProgram, const std::shared_ptr<RtProgramVars>& pVars, uint3 dispatchDims)
    {
        FALCOR_PROFILE("raytraceScene");

        assert(pContext && pProgram && pVars);
        if (pVars->getRayTypeCount() > 0 && pVars->getGeometryCount() != getGeometryCount())
        {
            throw RuntimeError("RtProgramVars geometry count mismatch");
        }

        uint32_t rayTypeCount = pVars->getRayTypeCount();
        setRaytracingShaderData(pContext, pVars->getRootVar(), rayTypeCount);

        // Set ray type constant.
        pVars->getRootVar()["DxrPerFrame"]["rayTypeCount"] = rayTypeCount;

        pContext->raytrace(pProgram, pVars.get(), dispatchDims.x, dispatchDims.y, dispatchDims.z);
    }

    void Scene::createMeshVao(uint32_t drawCount, const std::vector<uint32_t>& indexData, const std::vector<PackedStaticVertexData>& staticData, const std::vector<DynamicVertexData>& dynamicData)
    {
        // Create the index buffer.
        size_t ibSize = sizeof(uint32_t) * indexData.size();
        if (ibSize > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Index buffer size exceeds 4GB");
        }

        Buffer::SharedPtr pIB = nullptr;
        if (ibSize > 0)
        {
            ResourceBindFlags ibBindFlags = Resource::BindFlags::Index | ResourceBindFlags::ShaderResource;
            pIB = Buffer::create(ibSize, ibBindFlags, Buffer::CpuAccess::None, indexData.data());
        }

        // Create the vertex data structured buffer.
        const size_t vertexCount = (uint32_t)staticData.size();
        size_t staticVbSize = sizeof(PackedStaticVertexData) * vertexCount;
        if (staticVbSize > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Vertex buffer size exceeds 4GB");
        }

        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Vertex;
        Buffer::SharedPtr pStaticBuffer = Buffer::createStructured(sizeof(PackedStaticVertexData), (uint32_t)vertexCount, vbBindFlags, Buffer::CpuAccess::None, nullptr, false);

        Vao::BufferVec pVBs(kVertexBufferCount);
        pVBs[kStaticDataBufferIndex] = pStaticBuffer;

        // Create the draw ID buffer.
        // This is only needed when rasterizing the scene.
        ResourceFormat drawIDFormat = drawCount <= (1 << 16) ? ResourceFormat::R16Uint : ResourceFormat::R32Uint;

        Buffer::SharedPtr pDrawIDBuffer = nullptr;
        if (drawIDFormat == ResourceFormat::R16Uint)
        {
            assert(drawCount <= (1 << 16));
            std::vector<uint16_t> drawIDs(drawCount);
            for (uint32_t i = 0; i < drawCount; i++) drawIDs[i] = i;
            pDrawIDBuffer = Buffer::create(drawCount * sizeof(uint16_t), ResourceBindFlags::Vertex, Buffer::CpuAccess::None, drawIDs.data());
        }
        else if (drawIDFormat == ResourceFormat::R32Uint)
        {
            std::vector<uint32_t> drawIDs(drawCount);
            for (uint32_t i = 0; i < drawCount; i++) drawIDs[i] = i;
            pDrawIDBuffer = Buffer::create(drawCount * sizeof(uint32_t), ResourceBindFlags::Vertex, Buffer::CpuAccess::None, drawIDs.data());
        }
        else should_not_get_here();

        assert(pDrawIDBuffer);
        pVBs[kDrawIdBufferIndex] = pDrawIDBuffer;

        // Create vertex layout.
        // The layout only initializes the vertex data and draw ID layout. The skinning data doesn't get passed into the vertex shader.
        VertexLayout::SharedPtr pLayout = VertexLayout::create();

        // Add the packed static vertex data layout.
        VertexBufferLayout::SharedPtr pStaticLayout = VertexBufferLayout::create();
        pStaticLayout->addElement(VERTEX_POSITION_NAME, offsetof(PackedStaticVertexData, position), ResourceFormat::RGB32Float, 1, VERTEX_POSITION_LOC);
        pStaticLayout->addElement(VERTEX_PACKED_NORMAL_TANGENT_NAME, offsetof(PackedStaticVertexData, packedNormalTangent), ResourceFormat::RGB32Float, 1, VERTEX_PACKED_NORMAL_TANGENT_LOC);
        pStaticLayout->addElement(VERTEX_TEXCOORD_NAME, offsetof(PackedStaticVertexData, texCrd), ResourceFormat::RG32Float, 1, VERTEX_TEXCOORD_LOC);
        pLayout->addBufferLayout(kStaticDataBufferIndex, pStaticLayout);

        // Add the draw ID layout.
        VertexBufferLayout::SharedPtr pInstLayout = VertexBufferLayout::create();
        pInstLayout->addElement(INSTANCE_DRAW_ID_NAME, 0, drawIDFormat, 1, INSTANCE_DRAW_ID_LOC);
        pInstLayout->setInputClass(VertexBufferLayout::InputClass::PerInstanceData, 1);
        pLayout->addBufferLayout(kDrawIdBufferIndex, pInstLayout);

        // Create the VAO objects.
        // Note that the global index buffer can be mixed 16/32-bit format.
        // For drawing the meshes we need separate VAOs for these cases.
        mpVao = Vao::create(Vao::Topology::TriangleList, pLayout, pVBs, pIB, ResourceFormat::R32Uint);
        mpVao16Bit = Vao::create(Vao::Topology::TriangleList, pLayout, pVBs, pIB, ResourceFormat::R16Uint);
    }

    void Scene::createCurveVao(const std::vector<uint32_t>& indexData, const std::vector<StaticCurveVertexData>& staticData)
    {
        if (indexData.empty() || staticData.empty()) return;

        // Create the index buffer.
        size_t ibSize = sizeof(uint32_t) * indexData.size();
        if (ibSize > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Curve index buffer size exceeds 4GB");
        }

        Buffer::SharedPtr pIB = nullptr;
        if (ibSize > 0)
        {
            ResourceBindFlags ibBindFlags = Resource::BindFlags::Index | ResourceBindFlags::ShaderResource;
            pIB = Buffer::create(ibSize, ibBindFlags, Buffer::CpuAccess::None, indexData.data());
        }

        // Create the vertex data as structured buffers.
        const size_t vertexCount = (uint32_t)staticData.size();
        size_t staticVbSize = sizeof(StaticCurveVertexData) * vertexCount;
        if (staticVbSize > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Curve vertex buffer exceeds 4GB");
        }

        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Vertex;
        // Also upload the curve vertex data.
        Buffer::SharedPtr pStaticBuffer = Buffer::createStructured(sizeof(StaticCurveVertexData), (uint32_t)vertexCount, vbBindFlags, Buffer::CpuAccess::None, staticData.data(), false);

        // Curves do not need DrawIDBuffer.
        Vao::BufferVec pVBs(kVertexBufferCount - 1);
        pVBs[kStaticDataBufferIndex] = pStaticBuffer;

        // Create vertex layout.
        // The layout only initializes the vertex data layout. The skinning data doesn't get passed into the vertex shader.
        VertexLayout::SharedPtr pLayout = VertexLayout::create();

        // Add the packed static vertex data layout.
        VertexBufferLayout::SharedPtr pStaticLayout = VertexBufferLayout::create();
        pStaticLayout->addElement(CURVE_VERTEX_POSITION_NAME, offsetof(StaticCurveVertexData, position), ResourceFormat::RGB32Float, 1, CURVE_VERTEX_POSITION_LOC);
        pStaticLayout->addElement(CURVE_VERTEX_RADIUS_NAME, offsetof(StaticCurveVertexData, radius), ResourceFormat::R32Float, 1, CURVE_VERTEX_RADIUS_LOC);
        pStaticLayout->addElement(CURVE_VERTEX_TEXCOORD_NAME, offsetof(StaticCurveVertexData, texCrd), ResourceFormat::RG32Float, 1, CURVE_VERTEX_TEXCOORD_LOC);
        pLayout->addBufferLayout(kStaticDataBufferIndex, pStaticLayout);

        // Create the VAO objects.
        mpCurveVao = Vao::create(Vao::Topology::LineStrip, pLayout, pVBs, pIB, ResourceFormat::R32Uint);
    }

    void Scene::setSDFGridConfig()
    {
        if (mSDFGrids.empty()) return;

        for (const SDFGrid::SharedPtr& pSDFGrid : mSDFGrids)
        {
            if (mSDFGridConfig.implementation == SDFGrid::Type::None)
            {
                mSDFGridConfig.implementation = pSDFGrid->getType();
            }
            else if (mSDFGridConfig.implementation != pSDFGrid->getType())
            {
                reportError("All SDF grids in the same scene must currently be of the same type.");
            }
        }

        // Set default SDF grid config and ompute allowed SDF grid UI settings list.

        switch (mSDFGridConfig.implementation)
        {
        case SDFGrid::Type::NormalizedDenseGrid:
        {
            mSDFGridConfig.intersectionMethod = SDFGridIntersectionMethod::VoxelSphereTracing;
            mSDFGridConfig.gradientEvaluationMethod = SDFGridGradientEvaluationMethod::NumericDiscontinuous;
            mSDFGridConfig.solverMaxIterations = 256;
            mSDFGridConfig.optimizeVisibilityRays = true;

            mSDFGridConfig.intersectionMethodList =
            {
                { uint32_t(SDFGridIntersectionMethod::GridSphereTracing), "Grid Sphere Tracing" },
                { uint32_t(SDFGridIntersectionMethod::VoxelSphereTracing), "Voxel Sphere Tracing" },
            };

            mSDFGridConfig.gradientEvaluationMethodList =
            {
                { uint32_t(SDFGridGradientEvaluationMethod::NumericDiscontinuous), "Numeric Discontinuous" },
                { uint32_t(SDFGridGradientEvaluationMethod::NumericContinuous), "Numeric Continuous" },
            };

            break;
        }
        case SDFGrid::Type::SparseVoxelSet:
        case SDFGrid::Type::SparseBrickSet:
        {
            mSDFGridConfig.intersectionMethod = SDFGridIntersectionMethod::VoxelSphereTracing;
            mSDFGridConfig.gradientEvaluationMethod = SDFGridGradientEvaluationMethod::NumericDiscontinuous;
            mSDFGridConfig.solverMaxIterations = 256;
            mSDFGridConfig.optimizeVisibilityRays = true;

            mSDFGridConfig.intersectionMethodList =
            {
                { uint32_t(SDFGridIntersectionMethod::VoxelSphereTracing), "Voxel Sphere Tracing" },
            };

            mSDFGridConfig.gradientEvaluationMethodList =
            {
                { uint32_t(SDFGridGradientEvaluationMethod::NumericDiscontinuous), "Numeric Discontinuous" },
            };

            break;
        case SDFGrid::Type::SparseVoxelOctree:
            mSDFGridConfig.intersectionMethod = SDFGridIntersectionMethod::VoxelSphereTracing;
            mSDFGridConfig.gradientEvaluationMethod = SDFGridGradientEvaluationMethod::NumericDiscontinuous;
            mSDFGridConfig.solverMaxIterations = 256;
            mSDFGridConfig.optimizeVisibilityRays = true;

            mSDFGridConfig.intersectionMethodList =
            {
                { uint32_t(SDFGridIntersectionMethod::VoxelSphereTracing), "Voxel Sphere Tracing" },
            };

            mSDFGridConfig.gradientEvaluationMethodList =
            {
                { uint32_t(SDFGridGradientEvaluationMethod::NumericDiscontinuous), "Numeric Discontinuous" },
            };
            break;

        }
        }
    }

    void Scene::initSDFGrids()
    {
        if (mSDFGridConfig.implementation == SDFGrid::Type::SparseBrickSet)
        {
            mSDFGridConfig.implementationData.SBS.virtualBrickCoordsBitCount = 0;
            mSDFGridConfig.implementationData.SBS.brickLocalVoxelCoordsBitCount = 0;
        }
        else if (mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelOctree)
        {
            mSDFGridConfig.implementationData.SVO.svoIndexBitCount = 0;
        }

        for (const SDFGrid::SharedPtr& pSDFGrid : mSDFGrids)
        {
            pSDFGrid->createResources(gpDevice->getRenderContext());

            if (mSDFGridConfig.implementation == SDFGrid::Type::SparseBrickSet)
            {
                const SDFSBS* pSBS = reinterpret_cast<const SDFSBS*>(pSDFGrid.get());
                mSDFGridConfig.implementationData.SBS.virtualBrickCoordsBitCount = std::max(mSDFGridConfig.implementationData.SBS.virtualBrickCoordsBitCount, pSBS->getVirtualBrickCoordsBitCount());
                mSDFGridConfig.implementationData.SBS.brickLocalVoxelCoordsBitCount = std::max(mSDFGridConfig.implementationData.SBS.brickLocalVoxelCoordsBitCount, pSBS->getBrickLocalVoxelCoordsBrickCount());
            }
            else if (mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelOctree)
            {
                const SDFSVO* pSVO = reinterpret_cast<const SDFSVO*>(pSDFGrid.get());
                mSDFGridConfig.implementationData.SVO.svoIndexBitCount = std::max(mSDFGridConfig.implementationData.SVO.svoIndexBitCount, pSVO->getSVOIndexBitCount());
            }
        }
    }

    void Scene::initResources()
    {
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("Scene/SceneBlock.slang", "", "main", getSceneDefines());
        ParameterBlockReflection::SharedConstPtr pReflection = pProgram->getReflector()->getParameterBlock(kParameterBlockName);
        assert(pReflection);

        mpSceneBlock = ParameterBlock::create(pReflection);
        mpMeshesBuffer = Buffer::createStructured(mpSceneBlock[kMeshBufferName], (uint32_t)mMeshDesc.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpMeshesBuffer->setName("Scene::mpMeshesBuffer");

        mpGeometryInstancesBuffer = Buffer::createStructured(mpSceneBlock[kGeometryInstanceBufferName], (uint32_t)mGeometryInstanceData.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
        mpGeometryInstancesBuffer->setName("Scene::mpGeometryInstancesBuffer");

        if (!mCurveDesc.empty())
        {
            mpCurvesBuffer = Buffer::createStructured(mpSceneBlock[kCurveBufferName], (uint32_t)mCurveDesc.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpCurvesBuffer->setName("Scene::mpCurvesBuffer");
        }

        if (!mLights.empty())
        {
            mpLightsBuffer = Buffer::createStructured(mpSceneBlock[kLightsBufferName], (uint32_t)mLights.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpLightsBuffer->setName("Scene::mpLightsBuffer");
        }

        if (!mGridVolumes.empty())
        {
            mpGridVolumesBuffer = Buffer::createStructured(mpSceneBlock[kGridVolumesBufferName], (uint32_t)mGridVolumes.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpGridVolumesBuffer->setName("Scene::mpGridVolumesBuffer");
        }
    }

    void Scene::uploadResources()
    {
        assert(mpAnimationController);

        // Upload geometry.
        mpMeshesBuffer->setBlob(mMeshDesc.data(), 0, sizeof(MeshDesc) * mMeshDesc.size());
        if (!mCurveDesc.empty()) mpCurvesBuffer->setBlob(mCurveDesc.data(), 0, sizeof(CurveDesc) * mCurveDesc.size());

        mpSceneBlock->setBuffer(kGeometryInstanceBufferName, mpGeometryInstancesBuffer);
        mpSceneBlock->setBuffer(kMeshBufferName, mpMeshesBuffer);
        mpSceneBlock->setBuffer(kCurveBufferName, mpCurvesBuffer);

        auto sdfGridsVar = mpSceneBlock[kSDFGridsArrayName];

        for (uint32_t i = 0; i < mSDFGrids.size(); i++)
        {
            const SDFGrid::SharedPtr& pGrid = mSDFGrids[i];
            pGrid->setShaderData(sdfGridsVar[i]);
        }

        mpSceneBlock->setBuffer(kLightsBufferName, mpLightsBuffer);
        mpSceneBlock->setBuffer(kGridVolumesBufferName, mpGridVolumesBuffer);
        if (hasIndexBuffer()) mpSceneBlock->setBuffer(kIndexBufferName, mpVao->getIndexBuffer());
        mpSceneBlock->setBuffer(kVertexBufferName, mpVao->getVertexBuffer(Scene::kStaticDataBufferIndex));
        mpSceneBlock->setBuffer(kPrevVertexBufferName, mpAnimationController->getPrevVertexData()); // Can be nullptr

        if (mpCurveVao != nullptr)
        {
            mpSceneBlock->setBuffer(kCurveIndexBufferName, mpCurveVao->getIndexBuffer());
            mpSceneBlock->setBuffer(kCurveVertexBufferName, mpCurveVao->getVertexBuffer(Scene::kStaticDataBufferIndex));
            mpSceneBlock->setBuffer(kPrevCurveVertexBufferName, mpAnimationController->getPrevCurveVertexData());
        }
    }

    void Scene::uploadSelectedCamera()
    {
        getCamera()->setShaderData(mpSceneBlock[kCamera]);
    }

    void Scene::updateBounds()
    {
        const auto& globalMatrices = mpAnimationController->getGlobalMatrices();

        mSceneBB = AABB();

        for (const auto& inst : mGeometryInstanceData)
        {
            const glm::mat4& transform = globalMatrices[inst.globalMatrixID];
            switch (inst.getType())
            {
            case GeometryType::TriangleMesh:
            case GeometryType::DisplacedTriangleMesh:
            {
                const AABB& meshBB = mMeshBBs[inst.geometryID];
                mSceneBB |= meshBB.transform(transform);
                break;
            }
            case GeometryType::Curve:
            {
                const AABB& curveBB = mCurveBBs[inst.geometryID];
                mSceneBB |= curveBB.transform(transform);
                break;
            }
            case GeometryType::SDFGrid:
            {
                float3x3 transform3x3 = glm::mat3(transform3x3);
                transform3x3[0] = glm::abs(transform3x3[0]);
                transform3x3[1] = glm::abs(transform3x3[1]);
                transform3x3[2] = glm::abs(transform3x3[2]);
                float3 center = transform[3];
                float3 halfExtent = transform3x3 * float3(0.5f);
                mSceneBB |= AABB(center - halfExtent, center + halfExtent);
                break;
            }
            }
        }

        for (const auto& aabb : mCustomPrimitiveAABBs)
        {
            mSceneBB |= aabb;
        }

        for (const auto& pGridVolume : mGridVolumes)
        {
            mSceneBB |= pGridVolume->getBounds();
        }
    }

    void Scene::updateGeometryInstances(bool forceUpdate)
    {
        bool dataChanged = false;
        const auto& globalMatrices = mpAnimationController->getGlobalMatrices();

        for (auto& inst : mGeometryInstanceData)
        {
            if (inst.getType() == GeometryType::TriangleMesh || inst.getType() == GeometryType::DisplacedTriangleMesh)
            {
                uint32_t prevFlags = inst.flags;

                const glm::mat4& transform = globalMatrices[inst.globalMatrixID];
                bool isTransformFlipped = doesTransformFlip(transform);
                bool isObjectFrontFaceCW = getMesh(inst.geometryID).isFrontFaceCW();
                bool isWorldFrontFaceCW = isObjectFrontFaceCW ^ isTransformFlipped;

                if (isTransformFlipped) inst.flags |= (uint32_t)GeometryInstanceFlags::TransformFlipped;
                else inst.flags &= ~(uint32_t)GeometryInstanceFlags::TransformFlipped;

                if (isObjectFrontFaceCW) inst.flags |= (uint32_t)GeometryInstanceFlags::IsObjectFrontFaceCW;
                else inst.flags &= ~(uint32_t)GeometryInstanceFlags::IsObjectFrontFaceCW;

                if (isWorldFrontFaceCW) inst.flags |= (uint32_t)GeometryInstanceFlags::IsWorldFrontFaceCW;
                else inst.flags &= ~(uint32_t)GeometryInstanceFlags::IsWorldFrontFaceCW;

                dataChanged |= (inst.flags != prevFlags);
            }
        }

        if (forceUpdate || dataChanged)
        {
            uint32_t byteSize = (uint32_t)(mGeometryInstanceData.size() * sizeof(GeometryInstanceData));
            mpGeometryInstancesBuffer->setBlob(mGeometryInstanceData.data(), 0, byteSize);
        }
    }

    Scene::UpdateFlags Scene::updateRaytracingAABBData(bool forceUpdate)
    {
        // This function updates the global list of AABBs for all procedural primitives.
        // TODO: Move this code to the GPU. Then the CPU copies of some buffers won't be needed anymore.
        Scene::UpdateFlags flags = Scene::UpdateFlags::None;

        size_t curveAABBCount = 0;
        for (const auto& curve : mCurveDesc) curveAABBCount += curve.indexCount;

        size_t customAABBCount = mCustomPrimitiveAABBs.size();
        size_t totalAABBCount = curveAABBCount + customAABBCount;

        if (totalAABBCount > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Procedural primitive count exceeds the maximum");
        }

        // If there are no procedural primitives, clear the CPU buffer and return.
        // We'll leave the GPU buffer to be lazily re-allocated when needed.
        if (totalAABBCount == 0)
        {
            mRtAABBRaw.clear();
            return flags;
        }

        mRtAABBRaw.resize(totalAABBCount);
        uint32_t offset = 0;

        size_t firstUpdated = std::numeric_limits<size_t>::max();
        size_t lastUpdated = 0;

        if (forceUpdate)
        {
            // Compute AABBs of curve segments.
            for (const auto& curve : mCurveDesc)
            {
                // Track range of updated AABBs.
                // TODO: Per-curve flag to indicate changes. For now assume all curves need updating.
                firstUpdated = std::min(firstUpdated, (size_t)offset);
                lastUpdated = std::max(lastUpdated, (size_t)offset + curve.indexCount);

                const auto* indexData = &mCurveIndexData[curve.ibOffset];
                const auto* staticData = &mCurveStaticData[curve.vbOffset];

                for (uint32_t j = 0; j < curve.indexCount; j++)
                {
                    AABB curveSegBB;
                    uint32_t v = indexData[j];

                    for (uint32_t k = 0; k <= curve.degree; k++)
                    {
                        curveSegBB.include(staticData[v + k].position - float3(staticData[v + k].radius));
                        curveSegBB.include(staticData[v + k].position + float3(staticData[v + k].radius));
                    }

                    mRtAABBRaw[offset++] = static_cast<RtAABB>(curveSegBB);
                }
                flags |= Scene::UpdateFlags::CurvesMoved;
            }
            assert(offset == curveAABBCount);
        }
        offset = (uint32_t)curveAABBCount;

        if (forceUpdate || mCustomPrimitivesChanged || mCustomPrimitivesMoved)
        {
            mCustomPrimitiveAABBOffset = offset;

            // Track range of updated AABBs.
            firstUpdated = std::min(firstUpdated, (size_t)offset);
            lastUpdated = std::max(lastUpdated, (size_t)offset + customAABBCount);

            for (auto& aabb : mCustomPrimitiveAABBs)
            {
                mRtAABBRaw[offset++] = static_cast<RtAABB>(aabb);
            }
            assert(offset == totalAABBCount);
            flags |= Scene::UpdateFlags::CustomPrimitivesMoved;
        }

        // Create/update GPU buffer. This is used in BLAS creation and also bound to the scene for lookup in shaders.
        // Requires unordered access and will be in Non-Pixel Shader Resource state.
        if (mpRtAABBBuffer == nullptr || mpRtAABBBuffer->getElementCount() < (uint32_t)mRtAABBRaw.size())
        {
            mpRtAABBBuffer = Buffer::createStructured(sizeof(RtAABB), (uint32_t)mRtAABBRaw.size(), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, mRtAABBRaw.data(), false);
            mpRtAABBBuffer->setName("Scene::mpRtAABBBuffer");

            // Bind the new buffer to the scene.
            assert(mpSceneBlock);
            mpSceneBlock->setBuffer(kProceduralPrimAABBBufferName, mpRtAABBBuffer);
        }
        else if (firstUpdated < lastUpdated)
        {
            size_t bytes = sizeof(RtAABB) * mRtAABBRaw.size();
            assert(mpRtAABBBuffer && mpRtAABBBuffer->getSize() >= bytes);

            // Update the modified range of the GPU buffer.
            size_t offset = firstUpdated * sizeof(RtAABB);
            bytes = (lastUpdated - firstUpdated) * sizeof(RtAABB);
            mpRtAABBBuffer->setBlob(mRtAABBRaw.data() + firstUpdated, offset, bytes);
        }

        return flags;
    }

    Scene::UpdateFlags Scene::updateDisplacement(bool forceUpdate)
    {
        if (!hasGeometryType(GeometryType::DisplacedTriangleMesh)) return UpdateFlags::None;

        // For now we assume that displaced meshes are static.
        // Create AABB and AABB update task buffers.
        if (!mDisplacement.pAABBBuffer)
        {
            mDisplacement.meshData.resize(mMeshDesc.size());
            mDisplacement.updateTasks.clear();

            uint32_t AABBOffset = 0;

            for (uint32_t meshID = 0; meshID < mMeshDesc.size(); ++meshID)
            {
                const auto& mesh = mMeshDesc[meshID];

                if (!mesh.isDisplaced())
                {
                    mDisplacement.meshData[meshID] = {};
                    continue;
                }

                uint32_t AABBCount = mesh.getTriangleCount();
                mDisplacement.meshData[meshID] = { AABBOffset, AABBCount };
                AABBOffset += AABBCount;

                DisplacementUpdateTask task;
                task.meshID = meshID;
                task.triangleIndex = 0;
                task.AABBIndex = mDisplacement.meshData[meshID].AABBOffset;
                task.count = mDisplacement.meshData[meshID].AABBCount;
                mDisplacement.updateTasks.push_back(task);
            }

            mDisplacement.pAABBBuffer = Buffer::createStructured(sizeof(RtAABB), AABBOffset, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

            assert(mDisplacement.updateTasks.size() < std::numeric_limits<uint32_t>::max());
            mDisplacement.pUpdateTasksBuffer = Buffer::createStructured((uint32_t)sizeof(DisplacementUpdateTask), (uint32_t)mDisplacement.updateTasks.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, mDisplacement.updateTasks.data());
        }

        assert(!mDisplacement.updateTasks.empty());

        // We cannot access the scene parameter block until its finalized.
        if (!mFinalized) return UpdateFlags::None;

        // Update the AABB data.
        if (!mDisplacement.pUpdatePass)
        {
            mDisplacement.pUpdatePass = ComputePass::create("Scene/Displacement/DisplacementUpdate.cs.slang", "main", getSceneDefines());
            mDisplacement.needsUpdate = true;
        }

        if (mDisplacement.needsUpdate)
        {
            // TODO: Only update objects with modified materials.

            FALCOR_PROFILE("updateDisplacement");

            mDisplacement.pUpdatePass->getVars()->setParameterBlock(kParameterBlockName, mpSceneBlock);

            auto var = mDisplacement.pUpdatePass->getRootVar()["CB"];
            var["gTaskCount"] = (uint32_t)mDisplacement.updateTasks.size();
            var["gTasks"] = mDisplacement.pUpdateTasksBuffer;
            var["gAABBs"] = mDisplacement.pAABBBuffer;

            mDisplacement.pUpdatePass->execute(gpDevice->getRenderContext(), uint3(DisplacementUpdateTask::kThreadCount, (uint32_t)mDisplacement.updateTasks.size(), 1));

            mCustomPrimitivesChanged = true; // Trigger a BVH update.
            mDisplacement.needsUpdate = false;
            return UpdateFlags::DisplacementChanged;
        }

        return UpdateFlags::None;
    }

    Scene::UpdateFlags Scene::updateProceduralPrimitives(bool forceUpdate)
    {
        // Update the AABB buffer.
        // The bounds are updated if any primitive has moved or been added/removed.
        Scene::UpdateFlags flags = updateRaytracingAABBData(forceUpdate);

        // Update the procedural primitives metadata.
        if (forceUpdate || mCustomPrimitivesChanged)
        {
            // Update the custom primitives buffer.
            if (!mCustomPrimitiveDesc.empty())
            {
                if (mpCustomPrimitivesBuffer == nullptr || mpCustomPrimitivesBuffer->getElementCount() < (uint32_t)mCustomPrimitiveDesc.size())
                {
                    mpCustomPrimitivesBuffer = Buffer::createStructured(mpSceneBlock[kCustomPrimitiveBufferName], (uint32_t)mCustomPrimitiveDesc.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, mCustomPrimitiveDesc.data(), false);
                    mpCustomPrimitivesBuffer->setName("Scene::mpCustomPrimitivesBuffer");

                    // Bind the buffer to the scene.
                    assert(mpSceneBlock);
                    mpSceneBlock->setBuffer(kCustomPrimitiveBufferName, mpCustomPrimitivesBuffer);
                }
                else
                {
                    size_t bytes = sizeof(CustomPrimitiveDesc) * mCustomPrimitiveDesc.size();
                    assert(mpCustomPrimitivesBuffer && mpCustomPrimitivesBuffer->getSize() >= bytes);
                    mpCustomPrimitivesBuffer->setBlob(mCustomPrimitiveDesc.data(), 0, bytes);
                }
            }

            // Update scene constants.
            uint32_t customPrimitiveInstanceOffset = getGeometryInstanceCount();
            uint32_t customPrimitiveInstanceCount = getCustomPrimitiveCount();

            auto var = mpSceneBlock->getRootVar();
            var["customPrimitiveInstanceOffset"] = customPrimitiveInstanceOffset;
            var["customPrimitiveInstanceCount"] = customPrimitiveInstanceCount;
            var["customPrimitiveAABBOffset"] = mCustomPrimitiveAABBOffset;

            flags |= Scene::UpdateFlags::GeometryChanged;
        }

        return flags;
    }

    void Scene::updateGeometryTypes()
    {
        mGeometryTypes = GeometryTypeFlags(0);
        if (getMeshCount() > 0) mGeometryTypes |= GeometryTypeFlags::TriangleMesh;
        auto hasDisplaced = std::any_of(mMeshDesc.begin(), mMeshDesc.end(), [](const auto& mesh) { return mesh.isDisplaced(); });
        if (hasDisplaced) mGeometryTypes |= GeometryTypeFlags::DisplacedTriangleMesh;
        if (getCurveCount() > 0) mGeometryTypes |= GeometryTypeFlags::Curve;
        if (getSDFGridCount() > 0) mGeometryTypes |= GeometryTypeFlags::SDFGrid;
        if (getCustomPrimitiveCount() > 0) mGeometryTypes |= GeometryTypeFlags::Custom;
    }

    void Scene::finalize()
    {
        // Prepare the materials.
        // This step is necessary for setting up scene defines, which are used below when creating scene resources.
        // TODO: Remove this when unbounded descriptor arrays are supported (#1321).
        assert(mpMaterials);
        mpMaterials->finalize();

        // Prepare and upload resources.
        // The order of these calls is important as there are dependencies between them.
        updateGeometryTypes();
        initSDFGrids();
        mHitInfo.init(*this, mUseCompressedHitInfo);
        initResources(); // Requires scene defines
        mpAnimationController->animate(gpDevice->getRenderContext(), 0); // Requires Scene block to exist
        updateGeometry(true); // Requires scene defines
        updateGeometryInstances(true);

        updateBounds();
        createDrawList();
        if (mCameras.size() == 0)
        {
            // Create a new camera to use in the event of a scene with no cameras
            mCameras.push_back(Camera::create());
            resetCamera();
        }
        setCameraController(mCamCtrlType);
        initializeCameras();
        uploadSelectedCamera();
        addViewpoint();
        updateLights(true);
        updateGridVolumes(true);
        updateEnvMap(true);
        updateMaterials(true);
        uploadResources(); // Upload data after initialization is complete

        updateGeometryStats();
        updateMaterialStats();
        updateLightStats();
        updateGridVolumeStats();
        prepareUI();

        mFinalized = true;
    }

    void Scene::initializeCameras()
    {
        for (auto& camera : mCameras)
        {
            updateAnimatable(*camera, *mpAnimationController, true);
            camera->beginFrame();
        }
    }

    void Scene::prepareUI()
    {
        for (uint32_t camId = 0; camId < (uint32_t)mCameras.size(); camId++)
        {
            mCameraList.push_back({ camId, mCameras[camId]->getName() });
        }
    }

    void Scene::updateGeometryStats()
    {
        auto& s = mSceneStats;

        s.meshCount = getMeshCount();
        s.meshInstanceCount = 0;
        s.meshInstanceOpaqueCount = 0;
        s.transformCount = getAnimationController()->getGlobalMatrices().size();
        s.uniqueVertexCount = 0;
        s.uniqueTriangleCount = 0;
        s.instancedVertexCount = 0;
        s.instancedTriangleCount = 0;
        s.curveCount = getCurveCount();
        s.curveInstanceCount = 0;
        s.uniqueCurvePointCount = 0;
        s.uniqueCurveSegmentCount = 0;
        s.instancedCurvePointCount = 0;
        s.instancedCurveSegmentCount = 0;
        s.sdfGridCount = getSDFGridCount();
        s.sdfGridDescriptorCount = getSDFGridDescCount();
        s.sdfGridInstancesCount = 0;

        s.customPrimitiveCount = getCustomPrimitiveCount();

        for (uint32_t instanceID = 0; instanceID < getGeometryInstanceCount(); instanceID++)
        {
            const auto& instance = getGeometryInstance(instanceID);
            switch (instance.getType())
            {
            case GeometryType::TriangleMesh:
            case GeometryType::DisplacedTriangleMesh:
            {
                s.meshInstanceCount++;
                const auto& mesh = getMesh(instance.geometryID);
                s.instancedVertexCount += mesh.vertexCount;
                s.instancedTriangleCount += mesh.getTriangleCount();

                auto pMaterial = getMaterial(instance.materialID);
                if (pMaterial->isOpaque()) s.meshInstanceOpaqueCount++;
                break;
            }
            case GeometryType::Curve:
            {
                s.curveInstanceCount++;
                const auto& curve = getCurve(instance.geometryID);
                s.instancedCurvePointCount += curve.vertexCount;
                s.instancedCurveSegmentCount += curve.getSegmentCount();
                break;
            }
            case GeometryType::SDFGrid:
            {
                s.sdfGridInstancesCount++;
                break;
            }
            }
        }

        for (uint32_t meshID = 0; meshID < getMeshCount(); meshID++)
        {
            const auto& mesh = getMesh(meshID);
            s.uniqueVertexCount += mesh.vertexCount;
            s.uniqueTriangleCount += mesh.getTriangleCount();
        }

        for (uint32_t curveID = 0; curveID < getCurveCount(); curveID++)
        {
            const auto& curve = getCurve(curveID);
            s.uniqueCurvePointCount += curve.vertexCount;
            s.uniqueCurveSegmentCount += curve.getSegmentCount();
        }

        // Calculate memory usage.
        const auto& pIB = mpVao->getIndexBuffer();
        const auto& pVB = mpVao->getVertexBuffer(kStaticDataBufferIndex);
        const auto& pDrawID = mpVao->getVertexBuffer(kDrawIdBufferIndex);

        s.indexMemoryInBytes = 0;
        s.vertexMemoryInBytes = 0;
        s.geometryMemoryInBytes = 0;
        s.animationMemoryInBytes = 0;

        s.indexMemoryInBytes += pIB ? pIB->getSize() : 0;
        s.vertexMemoryInBytes += pVB ? pVB->getSize() : 0;

        s.curveIndexMemoryInBytes = 0;
        s.curveVertexMemoryInBytes = 0;

        if (mpCurveVao != nullptr)
        {
            const auto& pCurveIB = mpCurveVao->getIndexBuffer();
            const auto& pCurveVB = mpCurveVao->getVertexBuffer(kStaticDataBufferIndex);

            s.curveIndexMemoryInBytes += pCurveIB ? pCurveIB->getSize() : 0;
            s.curveVertexMemoryInBytes += pCurveVB ? pCurveVB->getSize() : 0;
        }

        s.sdfGridMemoryInBytes = 0;

        for (const SDFGrid::SharedPtr& pSDFGrid : mSDFGrids)
        {
            s.sdfGridMemoryInBytes += pSDFGrid->getSize();
        }

        s.geometryMemoryInBytes += mpGeometryInstancesBuffer ? mpGeometryInstancesBuffer->getSize() : 0;
        s.geometryMemoryInBytes += mpMeshesBuffer ? mpMeshesBuffer->getSize() : 0;
        s.geometryMemoryInBytes += mpCustomPrimitivesBuffer ? mpCustomPrimitivesBuffer->getSize() : 0;
        s.geometryMemoryInBytes += mpRtAABBBuffer ? mpRtAABBBuffer->getSize() : 0;
        s.geometryMemoryInBytes += pDrawID ? pDrawID->getSize() : 0;
        for (const auto& draw : mDrawArgs)
        {
            assert(draw.pBuffer);
            s.geometryMemoryInBytes += draw.pBuffer->getSize();
        }
        s.geometryMemoryInBytes += mpCurvesBuffer ? mpCurvesBuffer->getSize() : 0;

        s.animationMemoryInBytes += getAnimationController()->getMemoryUsageInBytes();
    }

    void Scene::updateMaterialStats()
    {
        mSceneStats.materials = mpMaterials->getStats();
    }

    void Scene::updateRaytracingBLASStats()
    {
        auto& s = mSceneStats;

        s.blasGroupCount = mBlasGroups.size();
        s.blasCount = mBlasData.size();
        s.blasCompactedCount = 0;
        s.blasOpaqueCount = 0;
        s.blasGeometryCount = 0;
        s.blasOpaqueGeometryCount = 0;
        s.blasMemoryInBytes = 0;
        s.blasScratchMemoryInBytes = 0;

        for (const auto& blas : mBlasData)
        {
            if (blas.useCompaction) s.blasCompactedCount++;
            s.blasMemoryInBytes += blas.blasByteSize;

            // Count number of opaque geometries in BLAS.
            uint64_t opaque = 0;
            for (const auto& desc : blas.geomDescs)
            {
                if (is_set(desc.flags, RtGeometryFlags::Opaque)) opaque++;
            }

            if (opaque == blas.geomDescs.size()) s.blasOpaqueCount++;
            s.blasGeometryCount += blas.geomDescs.size();
            s.blasOpaqueGeometryCount += opaque;
        }
        if (mpBlasScratch) s.blasScratchMemoryInBytes += mpBlasScratch->getSize();
        if (mpBlasStaticWorldMatrices) s.blasScratchMemoryInBytes += mpBlasStaticWorldMatrices->getSize();
    }

    void Scene::updateRaytracingTLASStats()
    {
        auto& s = mSceneStats;

        s.tlasCount = 0;
        s.tlasMemoryInBytes = 0;
        s.tlasScratchMemoryInBytes = 0;

        for (const auto& [i, tlas] : mTlasCache)
        {
            if (tlas.pTlasBuffer)
            {
                s.tlasMemoryInBytes += tlas.pTlasBuffer->getSize();
                s.tlasCount++;
            }
            if (tlas.pInstanceDescs) s.tlasScratchMemoryInBytes += tlas.pInstanceDescs->getSize();
        }
        if (mpTlasScratch) s.tlasScratchMemoryInBytes += mpTlasScratch->getSize();
    }

    void Scene::updateLightStats()
    {
        auto& s = mSceneStats;

        s.activeLightCount = mActiveLightCount;
        s.totalLightCount = mLights.size();
        s.pointLightCount = 0;
        s.directionalLightCount = 0;
        s.rectLightCount = 0;
        s.sphereLightCount = 0;
        s.distantLightCount = 0;

        for (const auto& light : mLights)
        {
            switch (light->getType())
            {
            case LightType::Point:
                s.pointLightCount++;
                break;
            case LightType::Directional:
                s.directionalLightCount++;
                break;
            case LightType::Rect:
                s.rectLightCount++;
                break;
            case LightType::Sphere:
                s.sphereLightCount++;
                break;
            case LightType::Distant:
                s.distantLightCount++;
                break;
            }
        }

        s.lightsMemoryInBytes = mpLightsBuffer ? mpLightsBuffer->getSize() : 0;
    }

    void Scene::updateGridVolumeStats()
    {
        auto& s = mSceneStats;

        s.gridVolumeCount = mGridVolumes.size();
        s.gridVolumeMemoryInBytes = mpGridVolumesBuffer ? mpGridVolumesBuffer->getSize() : 0;

        s.gridCount = mGrids.size();
        s.gridVoxelCount = 0;
        s.gridMemoryInBytes = 0;

        for (const auto& pGrid : mGrids)
        {
            s.gridVoxelCount += pGrid->getVoxelCount();
            s.gridMemoryInBytes += pGrid->getGridSizeInBytes();
        }
    }

    bool Scene::updateAnimatable(Animatable& animatable, const AnimationController& controller, bool force)
    {
        uint32_t nodeID = animatable.getNodeID();

        // It is possible for this to be called on an object with no associated node in the scene graph (kInvalidNode),
        // e.g. non-animated lights. This check ensures that we return immediately instead of trying to check
        // matrices for a non-existent node.
        if (nodeID == kInvalidNode) return false;

        if (force || (animatable.hasAnimation() && animatable.isAnimated()))
        {
            if (!controller.isMatrixChanged(nodeID) && !force) return false;

            glm::mat4 transform = controller.getGlobalMatrices()[nodeID];
            animatable.updateFromAnimation(transform);
            return true;
        }
        return false;
    }

    Scene::UpdateFlags Scene::updateSelectedCamera(bool forceUpdate)
    {
        auto camera = mCameras[mSelectedCamera];

        if (forceUpdate || (camera->hasAnimation() && camera->isAnimated()))
        {
            updateAnimatable(*camera, *mpAnimationController, forceUpdate);
        }
        else
        {
            mpCamCtrl->update();
        }

        UpdateFlags flags = UpdateFlags::None;
        auto cameraChanges = camera->beginFrame();
        if (mCameraSwitched || cameraChanges != Camera::Changes::None)
        {
            uploadSelectedCamera();
            if (is_set(cameraChanges, Camera::Changes::Movement)) flags |= UpdateFlags::CameraMoved;
            if ((cameraChanges & (~Camera::Changes::Movement)) != Camera::Changes::None) flags |= UpdateFlags::CameraPropertiesChanged;
            if (mCameraSwitched) flags |= UpdateFlags::CameraSwitched;
        }
        mCameraSwitched = false;
        return flags;
    }

    Scene::UpdateFlags Scene::updateLights(bool forceUpdate)
    {
        Light::Changes combinedChanges = Light::Changes::None;

        // Animate lights and get list of changes.
        for (const auto& light : mLights)
        {
            if (light->isActive())
            {
                updateAnimatable(*light, *mpAnimationController, forceUpdate);
            }

            auto changes = light->beginFrame();
            combinedChanges |= changes;
        }

        // Update changed lights.
        mActiveLightCount = 0;

        for (const auto& light : mLights)
        {
            if (!light->isActive()) continue;

            auto changes = light->getChanges();
            if (changes != Light::Changes::None || is_set(combinedChanges, Light::Changes::Active) || forceUpdate)
            {
                // TODO: This is slow since the buffer is not CPU writable. Copy into CPU buffer and upload once instead.
                mpLightsBuffer->setElement(mActiveLightCount, light->getData());
            }

            mActiveLightCount++;
        }

        if (combinedChanges != Light::Changes::None || forceUpdate)
        {
            mpSceneBlock["lightCount"] = mActiveLightCount;
            updateLightStats();
        }

        // Compute update flags.
        UpdateFlags flags = UpdateFlags::None;
        if (is_set(combinedChanges, Light::Changes::Intensity)) flags |= UpdateFlags::LightIntensityChanged;
        if (is_set(combinedChanges, Light::Changes::Position)) flags |= UpdateFlags::LightsMoved;
        if (is_set(combinedChanges, Light::Changes::Direction)) flags |= UpdateFlags::LightsMoved;
        if (is_set(combinedChanges, Light::Changes::Active)) flags |= UpdateFlags::LightCountChanged;
        const Light::Changes otherChanges = ~(Light::Changes::Intensity | Light::Changes::Position | Light::Changes::Direction | Light::Changes::Active);
        if ((combinedChanges & otherChanges) != Light::Changes::None) flags |= UpdateFlags::LightPropertiesChanged;

        return flags;
    }

    Scene::UpdateFlags Scene::updateGridVolumes(bool forceUpdate)
    {
        GridVolume::UpdateFlags combinedUpdates = GridVolume::UpdateFlags::None;

        // Update animations and get combined updates.
        for (const auto& pGridVolume : mGridVolumes)
        {
            updateAnimatable(*pGridVolume, *mpAnimationController, forceUpdate);
            combinedUpdates |= pGridVolume->getUpdates();
        }

        // Early out if no volumes have changed.
        if (!forceUpdate && combinedUpdates == GridVolume::UpdateFlags::None) return UpdateFlags::None;

        // Upload grids.
        if (forceUpdate)
        {
            auto var = mpSceneBlock["grids"];
            for (size_t i = 0; i < mGrids.size(); ++i)
            {
                mGrids[i]->setShaderData(var[i]);
            }
        }

        // Upload volumes and clear updates.
        uint32_t volumeIndex = 0;
        for (const auto& pGridVolume : mGridVolumes)
        {
            if (forceUpdate || pGridVolume->getUpdates() != GridVolume::UpdateFlags::None)
            {
                // Fetch copy of volume data.
                auto data = pGridVolume->getData();
                data.densityGrid = pGridVolume->getDensityGrid() ? mGridIDs.at(pGridVolume->getDensityGrid()) : kInvalidGrid;
                data.emissionGrid = pGridVolume->getEmissionGrid() ? mGridIDs.at(pGridVolume->getEmissionGrid()) : kInvalidGrid;
                // Merge grid and volume transforms.
                const auto& densityGrid = pGridVolume->getDensityGrid();
                if (densityGrid)
                {
                    data.transform = data.transform * densityGrid->getTransform();
                    data.invTransform = densityGrid->getInvTransform() * data.invTransform;
                }
                mpGridVolumesBuffer->setElement(volumeIndex, data);
            }
            pGridVolume->clearUpdates();
            volumeIndex++;
        }

        mpSceneBlock["gridVolumeCount"] = (uint32_t)mGridVolumes.size();

        UpdateFlags flags = UpdateFlags::None;
        if (is_set(combinedUpdates, GridVolume::UpdateFlags::TransformChanged)) flags |= UpdateFlags::GridVolumesMoved;
        if (is_set(combinedUpdates, GridVolume::UpdateFlags::PropertiesChanged)) flags |= UpdateFlags::GridVolumePropertiesChanged;
        if (is_set(combinedUpdates, GridVolume::UpdateFlags::GridsChanged)) flags |= UpdateFlags::GridVolumeGridsChanged;
        if (is_set(combinedUpdates, GridVolume::UpdateFlags::BoundsChanged)) flags |= UpdateFlags::GridVolumeBoundsChanged;

        return flags;
    }

    Scene::UpdateFlags Scene::updateEnvMap(bool forceUpdate)
    {
        UpdateFlags flags = UpdateFlags::None;

        if (mpEnvMap)
        {
            auto envMapChanges = mpEnvMap->beginFrame();
            if (envMapChanges != EnvMap::Changes::None || mEnvMapChanged || forceUpdate)
            {
                if (envMapChanges != EnvMap::Changes::None) flags |= UpdateFlags::EnvMapPropertiesChanged;
                mpEnvMap->setShaderData(mpSceneBlock[kEnvMap]);
            }
        }
        mSceneStats.envMapMemoryInBytes = mpEnvMap ? mpEnvMap->getMemoryUsageInBytes() : 0;

        if (mEnvMapChanged)
        {
            flags |= UpdateFlags::EnvMapChanged;
            mEnvMapChanged = false;
        }

        return flags;
    }

    Scene::UpdateFlags Scene::updateMaterials(bool forceUpdate)
    {
        // Update material system.
        Material::UpdateFlags materialUpdates = mpMaterials->update(forceUpdate);

        UpdateFlags flags = UpdateFlags::None;
        if (forceUpdate || materialUpdates != Material::UpdateFlags::None)
        {
            flags |= UpdateFlags::MaterialsChanged;

            // Bind materials parameter block to scene.
            mpSceneBlock->setParameterBlock(kMaterialsBlockName, mpMaterials->getParameterBlock());

            // If displacement parameters have changed, we need to trigger displacement update.
            if (is_set(materialUpdates, Material::UpdateFlags::DisplacementChanged))
            {
                mDisplacement.needsUpdate = true;
            }

            updateMaterialStats();
        }

        return flags;
    }

    Scene::UpdateFlags Scene::updateGeometry(bool forceUpdate)
    {
        UpdateFlags flags = updateProceduralPrimitives(forceUpdate);
        flags |= updateDisplacement(forceUpdate);

        if (forceUpdate || mCustomPrimitivesChanged)
        {
            updateGeometryTypes();
            updateGeometryStats();

            // Clear any previous BLAS data. This will trigger a full BLAS/TLAS rebuild.
            // TODO: Support partial rebuild of just the procedural primitives.
            mBlasData.clear();
        }

        mCustomPrimitivesMoved = false;
        mCustomPrimitivesChanged = false;
        return flags;
    }

    Scene::UpdateFlags Scene::update(RenderContext* pContext, double currentTime)
    {
        // Run scene update callback.
        if (mUpdateCallback) mUpdateCallback(shared_from_this(), currentTime);

        mUpdates = UpdateFlags::None;

        if (mpAnimationController->animate(pContext, currentTime))
        {
            mUpdates |= UpdateFlags::SceneGraphChanged;
            for (const auto& inst : mGeometryInstanceData)
            {
                if (mpAnimationController->isMatrixChanged(inst.globalMatrixID))
                {
                    mUpdates |= UpdateFlags::GeometryMoved;
                }
            }

            // We might end up setting the flag even if curves haven't changed (if looping is disabled for example).
            if (mpAnimationController->hasAnimatedCurveCaches()) mUpdates |= UpdateFlags::CurvesMoved;
        }

        for (const auto& pGridVolume : mGridVolumes)
        {
            pGridVolume->updatePlayback(currentTime);
        }

        mUpdates |= updateSelectedCamera(false);
        mUpdates |= updateLights(false);
        mUpdates |= updateGridVolumes(false);
        mUpdates |= updateEnvMap(false);
        mUpdates |= updateMaterials(false);
        mUpdates |= updateGeometry(false);
        pContext->flush();

        if (is_set(mUpdates, UpdateFlags::GeometryMoved))
        {
            mTlasCache.clear();
            updateGeometryInstances(false);
        }

        // Update existing BLASes if skinned animation and/or procedural primitives moved.
        bool skinnedAnimation = mHasSkinnedMesh && is_set(mUpdates, UpdateFlags::SceneGraphChanged);
        bool updateProcedural = is_set(mUpdates, UpdateFlags::CurvesMoved) || is_set(mUpdates, UpdateFlags::CustomPrimitivesMoved);
        bool blasUpdateRequired = skinnedAnimation || updateProcedural;

        if (!mBlasData.empty() && blasUpdateRequired)
        {
            mTlasCache.clear();
            buildBlas(pContext);
        }

        // Update light collection
        if (mpLightCollection && mpLightCollection->update(pContext))
        {
            mUpdates |= UpdateFlags::LightCollectionChanged;
            mSceneStats.emissiveMemoryInBytes = mpLightCollection->getMemoryUsageInBytes();
        }
        else if (!mpLightCollection)
        {
            mSceneStats.emissiveMemoryInBytes = 0;
        }

        if (mRenderSettings != mPrevRenderSettings)
        {
            mUpdates |= UpdateFlags::RenderSettingsChanged;
            mPrevRenderSettings = mRenderSettings;
        }

        if (mSDFGridConfig != mPrevSDFGridConfig)
        {
            mUpdates |= UpdateFlags::SDFGridConfigChanged;
            mPrevSDFGridConfig = mSDFGridConfig;
        }

        return mUpdates;
    }

    void Scene::renderUI(Gui::Widgets& widget)
    {
        if (mpAnimationController->hasAnimations())
        {
            bool isEnabled = mpAnimationController->isEnabled();
            if (widget.checkbox("Animate Scene", isEnabled)) mpAnimationController->setEnabled(isEnabled);

            if (auto animGroup = widget.group("Animations"))
            {
                mpAnimationController->renderUI(animGroup);
            }
        }

        auto camera = mCameras[mSelectedCamera];
        if (camera->hasAnimation())
        {
            bool isAnimated = camera->isAnimated();
            if (widget.checkbox("Animate Camera", isAnimated)) camera->setIsAnimated(isAnimated);
        }

        if (widget.var("Camera Speed", mCameraSpeed, 0.f, std::numeric_limits<float>::max(), 0.01f))
        {
            mpCamCtrl->setCameraSpeed(mCameraSpeed);
        }

        if (mCameraList.size() > 1)
        {
            uint32_t camIndex = mSelectedCamera;
            if (widget.dropdown("Selected Camera", mCameraList, camIndex)) selectCamera(camIndex);
        }

        if (widget.button("Add Viewpoint")) addViewpoint();

        if (mViewpoints.size() > 1)
        {
            if (widget.button("Remove Viewpoint", true)) removeViewpoint();

            static uint32_t animationLength = 30;
            widget.var("Animation Length", animationLength, 1u, 120u);

            if (widget.button("Save Viewpoints"))
            {
                static const FileDialogFilterVec kFileExtensionFilters = { { "txt", "Text Files"} };
                std::string filename = "cameraPath.txt";
                if (saveFileDialog(kFileExtensionFilters, filename))
                {
                    std::ofstream file(filename, std::ios::out);
                    if (file.is_open())
                    {
                        for (uint32_t i = 0; i < mViewpoints.size(); i++)
                        {
                            const Viewpoint& vp = mViewpoints[i];
                            float timePoint = animationLength * float(i) / mViewpoints.size();

                            file << timePoint << ", Transform(";
                            file << "position = float3(" << vp.position.x << ", " << vp.position.y << ", " << vp.position.z << "), ";
                            file << "target = float3(" << vp.target.x << ", " << vp.target.y << ", " << vp.target.z << "), ";
                            file << "up = float3(" << vp.up.x << ", " << vp.up.y << ", " << vp.up.z << "))" << std::endl;
                        }

                        const Viewpoint& vp = mViewpoints[0];
                        file << animationLength << ", Transform(";
                        file << "position = float3(" << vp.position.x << ", " << vp.position.y << ", " << vp.position.z << "), ";
                        file << "target = float3(" << vp.target.x << ", " << vp.target.y << ", " << vp.target.z << "), ";
                        file << "up = float3(" << vp.up.x << ", " << vp.up.y << ", " << vp.up.z << "))" << std::endl;

                        file.close();
                    }
                }
            }

            Gui::DropdownList viewpoints;
            viewpoints.push_back({ 0, "Default Viewpoint" });
            for (uint32_t viewId = 1; viewId < (uint32_t)mViewpoints.size(); viewId++)
            {
                viewpoints.push_back({ viewId, "Viewpoint " + std::to_string(viewId) });
            }
            uint32_t viewIndex = mCurrentViewpoint;
            if (widget.dropdown("Viewpoints", viewpoints, viewIndex)) selectViewpoint(viewIndex);
        }

        if (auto cameraGroup = widget.group("Camera"))
        {
            camera->renderUI(cameraGroup);
        }

        if (auto renderSettingsGroup = widget.group("Render Settings"))
        {
            renderSettingsGroup.checkbox("Use environment light", mRenderSettings.useEnvLight);
            renderSettingsGroup.tooltip("This enables using the environment map as a distant light source.", true);

            renderSettingsGroup.checkbox("Use analytic lights", mRenderSettings.useAnalyticLights);
            renderSettingsGroup.tooltip("This enables using analytic lights.", true);

            renderSettingsGroup.checkbox("Use emissive", mRenderSettings.useEmissiveLights);
            renderSettingsGroup.tooltip("This enables using emissive triangles as lights.", true);

            renderSettingsGroup.checkbox("Use grid volumes", mRenderSettings.useGridVolumes);
            renderSettingsGroup.tooltip("This enables rendering of grid volumes.", true);
        }

        if (mSDFGridConfig.implementation != SDFGrid::Type::None)
        {
            if (auto sdfGridConfigGroup = widget.group("SDF Grid Settings"))
            {
                sdfGridConfigGroup.dropdown("Intersection Method", mSDFGridConfig.intersectionMethodList, reinterpret_cast<uint32_t&>(mSDFGridConfig.intersectionMethod));
                sdfGridConfigGroup.dropdown("Gradient Evaluation Method", mSDFGridConfig.gradientEvaluationMethodList, reinterpret_cast<uint32_t&>(mSDFGridConfig.gradientEvaluationMethod));
                sdfGridConfigGroup.var("Solver Max Iteration Count", mSDFGridConfig.solverMaxIterations, 0u, 512u, 1u);
                sdfGridConfigGroup.checkbox("Optimize Visibility Rays", mSDFGridConfig.optimizeVisibilityRays);
            }
        }

        if (auto envMapGroup = widget.group("EnvMap"))
        {
            if (envMapGroup.button("Load"))
            {
                std::string filename;
                if (openFileDialog(Bitmap::getFileDialogFilters(ResourceFormat::RGBA32Float), filename)) loadEnvMap(filename);
            }

            if (mpEnvMap && envMapGroup.button("Clear", true)) setEnvMap(nullptr);

            if (mpEnvMap) mpEnvMap->renderUI(envMapGroup);
        }

        if (auto lightsGroup = widget.group("Lights"))
        {
            uint32_t lightID = 0;
            for (auto& light : mLights)
            {
                auto name = std::to_string(lightID) + ": " + light->getName();
                if (auto lightGroup = lightsGroup.group(name))
                {
                    light->renderUI(lightGroup);
                }
                lightID++;
            }
        }

        if (auto materialsGroup = widget.group("Materials"))
        {
            mpMaterials->renderUI(materialsGroup);
        }

        if (auto volumesGroup = widget.group("Grid volumes"))
        {
            uint32_t volumeID = 0;
            for (auto& pGridVolume : mGridVolumes)
            {
                auto name = std::to_string(volumeID) + ": " + pGridVolume->getName();
                if (auto volumeGroup = volumesGroup.group(name))
                {
                    pGridVolume->renderUI(volumeGroup);
                }
                volumeID++;
            }
        }

        if (auto statsGroup = widget.group("Statistics"))
        {
            const auto& s = mSceneStats;
            const double bytesPerTexel = s.materials.textureTexelCount > 0 ? (double)s.materials.textureMemoryInBytes / s.materials.textureTexelCount : 0.0;

            std::ostringstream oss;
            oss << "Filename: " << mFilename << std::endl;
            oss << "Bounds: (" << mSceneBB.minPoint.x << "," << mSceneBB.minPoint.y << "," << mSceneBB.minPoint.z << ")-(" << mSceneBB.maxPoint.x << "," << mSceneBB.maxPoint.y << "," << mSceneBB.maxPoint.z << ")" << std::endl;
            oss << "Total scene memory: " << formatByteSize(s.getTotalMemory()) << std::endl;

            // Geometry stats.
            oss << "Geometry stats:" << std::endl
                << "  Mesh count: " << s.meshCount << std::endl
                << "  Mesh instance count (total): " << s.meshInstanceCount << std::endl
                << "  Mesh instance count (opaque): " << s.meshInstanceOpaqueCount << std::endl
                << "  Mesh instance count (non-opaque): " << (s.meshInstanceCount - s.meshInstanceOpaqueCount) << std::endl
                << "  Transform matrix count: " << s.transformCount << std::endl
                << "  Unique triangle count: " << s.uniqueTriangleCount << std::endl
                << "  Unique vertex count: " << s.uniqueVertexCount << std::endl
                << "  Instanced triangle count: " << s.instancedTriangleCount << std::endl
                << "  Instanced vertex count: " << s.instancedVertexCount << std::endl
                << "  Index  buffer memory: " << formatByteSize(s.indexMemoryInBytes) << std::endl
                << "  Vertex buffer memory: " << formatByteSize(s.vertexMemoryInBytes) << std::endl
                << "  Geometry data memory: " << formatByteSize(s.geometryMemoryInBytes) << std::endl
                << "  Animation data memory: " << formatByteSize(s.animationMemoryInBytes) << std::endl
                << "  Curve count: " << s.curveCount << std::endl
                << "  Curve instance count: " << s.curveInstanceCount << std::endl
                << "  Unique curve segment count: " << s.uniqueCurveSegmentCount << std::endl
                << "  Unique curve point count: " << s.uniqueCurvePointCount << std::endl
                << "  Instanced curve segment count: " << s.instancedCurveSegmentCount << std::endl
                << "  Instanced curve point count: " << s.instancedCurvePointCount << std::endl
                << "  Curve index buffer memory: " << formatByteSize(s.curveIndexMemoryInBytes) << std::endl
                << "  Curve vertex buffer memory: " << formatByteSize(s.curveVertexMemoryInBytes) << std::endl
                << "  SDF grid count: " << s.sdfGridCount << std::endl
                << "  SDF grid descriptor count: " << s.sdfGridDescriptorCount << std::endl
                << "  SDF grid instances count: " << s.sdfGridInstancesCount << std::endl
                << "  SDF grid memory: " << formatByteSize(s.sdfGridMemoryInBytes) << std::endl
                << "  Custom primitive count: " << s.customPrimitiveCount << std::endl
                << std::endl;

            // Raytracing stats.
            oss << "Raytracing stats:" << std::endl
                << "  BLAS groups: " << s.blasGroupCount << std::endl
                << "  BLAS count (total): " << s.blasCount << std::endl
                << "  BLAS count (compacted): " << s.blasCompactedCount << std::endl
                << "  BLAS count (opaque): " << s.blasOpaqueCount << std::endl
                << "  BLAS count (non-opaque): " << (s.blasCount - s.blasOpaqueCount) << std::endl
                << "  BLAS geometries (total): " << s.blasGeometryCount << std::endl
                << "  BLAS geometries (opaque): " << s.blasOpaqueGeometryCount << std::endl
                << "  BLAS geometries (non-opaque): " << (s.blasGeometryCount - s.blasOpaqueGeometryCount) << std::endl
                << "  BLAS memory (final): " << formatByteSize(s.blasMemoryInBytes) << std::endl
                << "  BLAS memory (scratch): " << formatByteSize(s.blasScratchMemoryInBytes) << std::endl
                << "  TLAS count: " << s.tlasCount << std::endl
                << "  TLAS memory (final): " << formatByteSize(s.tlasMemoryInBytes) << std::endl
                << "  TLAS memory (scratch): " << formatByteSize(s.tlasScratchMemoryInBytes) << std::endl
                << std::endl;

            // Material stats.
            oss << "Materials stats:" << std::endl
                << "  Material count (total): " << s.materials.materialCount << std::endl
                << "  Material count (opaque): " << s.materials.materialOpaqueCount << std::endl
                << "  Material count (non-opaque): " << (s.materials.materialCount - s.materials.materialOpaqueCount) << std::endl
                << "  Material memory: " << formatByteSize(s.materials.materialMemoryInBytes) << std::endl
                << "  Texture count (total): " << s.materials.textureCount << std::endl
                << "  Texture count (compressed): " << s.materials.textureCompressedCount << std::endl
                << "  Texture texel count: " << s.materials.textureTexelCount << std::endl
                << "  Texture memory: " << formatByteSize(s.materials.textureMemoryInBytes) << std::endl
                << "  Bytes/texel (average): " << std::fixed << std::setprecision(2) << bytesPerTexel << std::endl
                << std::endl;

            // Analytic light stats.
            oss << "Analytic light stats:" << std::endl
                << "  Active light count: " << s.activeLightCount << std::endl
                << "  Total light count: " << s.totalLightCount << std::endl
                << "  Point light count: " << s.pointLightCount << std::endl
                << "  Directional light count: " << s.directionalLightCount << std::endl
                << "  Rect light count: " << s.rectLightCount << std::endl
                << "  Sphere light count: " << s.sphereLightCount << std::endl
                << "  Distant light count: " << s.distantLightCount << std::endl
                << "  Analytic lights memory: " << formatByteSize(s.lightsMemoryInBytes) << std::endl
                << std::endl;

            // Emissive light stats.
            oss << "Emissive light stats:" << std::endl;
            if (mpLightCollection)
            {
                const auto& stats = mpLightCollection->getStats();
                oss << "  Active triangle count: " << stats.trianglesActive << std::endl
                    << "  Active uniform triangle count: " << stats.trianglesActiveUniform << std::endl
                    << "  Active textured triangle count: " << stats.trianglesActiveTextured << std::endl
                    << "  Details:" << std::endl
                    << "    Total mesh count: " << stats.meshLightCount << std::endl
                    << "    Textured mesh count: " << stats.meshesTextured << std::endl
                    << "    Total triangle count: " << stats.triangleCount << std::endl
                    << "    Texture triangle count: " << stats.trianglesTextured << std::endl
                    << "    Culled triangle count: " << stats.trianglesCulled << std::endl
                    << "  Emissive lights memory: " << formatByteSize(s.emissiveMemoryInBytes) << std::endl;
            }
            else
            {
                oss << "  N/A" << std::endl;
            }
            oss << std::endl;

            // Environment map stats.
            oss << "Environment map:" << std::endl;
            if (mpEnvMap)
            {
                oss << "  Filename: " << mpEnvMap->getFilename() << std::endl
                    << "  Resolution: " << mpEnvMap->getEnvMap()->getWidth() << "x" << mpEnvMap->getEnvMap()->getHeight() << std::endl
                    << "  Texture memory: " << formatByteSize(s.envMapMemoryInBytes) << std::endl;
            }
            else
            {
                oss << "  N/A" << std::endl;
            }
            oss << std::endl;

            // Grid volume stats.
            oss << "Grid volume stats:" << std::endl
                << "  Grid volume count: " << s.gridVolumeCount << std::endl
                << "  Grid volume memory: " << formatByteSize(s.gridVolumeMemoryInBytes) << std::endl
                << std::endl;

            // Grid stats.
            oss << "Grid stats:" << std::endl
                << "  Grid count: " << s.gridCount << std::endl
                << "  Grid voxel count: " << s.gridVoxelCount << std::endl
                << "  Grid memory: " << formatByteSize(s.gridMemoryInBytes) << std::endl
                << std::endl;

            if (statsGroup.button("Print to log")) logInfo("\n" + oss.str());

            statsGroup.text(oss.str());
        }

        // Filtering mode
        // Camera controller
    }

    bool Scene::useEnvBackground() const
    {
        return mpEnvMap != nullptr;
    }

    bool Scene::useEnvLight() const
    {
        return mRenderSettings.useEnvLight && mpEnvMap != nullptr && mpEnvMap->getIntensity() > 0.f;
    }

    bool Scene::useAnalyticLights() const
    {
        return mRenderSettings.useAnalyticLights && mLights.empty() == false && mActiveLightCount > 0;
    }

    bool Scene::useEmissiveLights() const
    {
        return mRenderSettings.useEmissiveLights && mpLightCollection != nullptr && mpLightCollection->getActiveLightCount() > 0;
    }

    bool Scene::useGridVolumes() const
    {
        return mRenderSettings.useGridVolumes && mGridVolumes.empty() == false;
    }

    void Scene::setCamera(const Camera::SharedPtr& pCamera)
    {
        auto it = std::find(mCameras.begin(), mCameras.end(), pCamera);
        if (it != mCameras.end())
        {
            selectCamera((uint32_t)std::distance(mCameras.begin(), it));
        }
        else if (pCamera)
        {
            logWarning("Selected camera " + pCamera->getName() + " does not exist.");
        }
    }

    void Scene::selectCamera(uint32_t index)
    {
        if (index == mSelectedCamera) return;
        if (index >= mCameras.size())
        {
            logWarning("Selected camera index " + std::to_string(index) + " is invalid.");
            return;
        }

        mSelectedCamera = index;
        mCameraSwitched = true;
        setCameraController(mCamCtrlType);
    }

    void Scene::resetCamera(bool resetDepthRange)
    {
        auto camera = getCamera();
        float radius = mSceneBB.radius();
        camera->setPosition(mSceneBB.center());
        camera->setTarget(mSceneBB.center() + float3(0, 0, -1));
        camera->setUpVector(float3(0, 1, 0));

        if (resetDepthRange)
        {
            float nearZ = std::max(0.1f, radius / 750.0f);
            float farZ = radius * 50;
            camera->setDepthRange(nearZ, farZ);
        }
    }

    void Scene::setCameraSpeed(float speed)
    {
        mCameraSpeed = clamp(speed, 0.f, std::numeric_limits<float>::max());
        mpCamCtrl->setCameraSpeed(speed);
    }

    void Scene::addViewpoint()
    {
        auto camera = getCamera();
        addViewpoint(camera->getPosition(), camera->getTarget(), camera->getUpVector(), mSelectedCamera);
    }

    void Scene::addViewpoint(const float3& position, const float3& target, const float3& up, uint32_t cameraIndex)
    {
        Viewpoint viewpoint = { cameraIndex, position, target, up };
        mViewpoints.push_back(viewpoint);
        mCurrentViewpoint = (uint32_t)mViewpoints.size() - 1;
    }

    void Scene::removeViewpoint()
    {
        if (mCurrentViewpoint == 0)
        {
            logWarning("Cannot remove default viewpoint");
            return;
        }
        mViewpoints.erase(mViewpoints.begin() + mCurrentViewpoint);
        mCurrentViewpoint = std::min(mCurrentViewpoint, (uint32_t)mViewpoints.size() - 1);
    }

    void Scene::selectViewpoint(uint32_t index)
    {
        if (index >= mViewpoints.size())
        {
            logWarning("Viewpoint does not exist");
            return;
        }

        auto& viewpoint = mViewpoints[index];
        selectCamera(viewpoint.index);
        auto camera = getCamera();
        camera->setPosition(viewpoint.position);
        camera->setTarget(viewpoint.target);
        camera->setUpVector(viewpoint.up);
        mCurrentViewpoint = index;
    }

    uint32_t Scene::getGeometryCount() const
    {
        // The BLASes currently hold the geometries in the order: meshes, curves, SDF grids, custom primitives.
        // We calculate the total number of geometries as the sum of the respective kind.

        size_t totalGeometries = mMeshDesc.size() + mCurveDesc.size() + mCustomPrimitiveDesc.size() + getSDFGridGeometryCount();
        assert(totalGeometries < std::numeric_limits<uint32_t>::max());
        return (uint32_t)totalGeometries;
    }

    std::vector<uint32_t> Scene::getGeometryIDs(GeometryType type) const
    {
        std::vector<uint32_t> geometryIDs;

        uint32_t geometryCount = getGeometryCount();
        for (uint32_t geometryID = 0; geometryID < geometryCount; geometryID++)
        {
            if (getGeometryType(geometryID) == type) geometryIDs.push_back(geometryID);
        }

        return geometryIDs;
    }

    Scene::GeometryType Scene::getGeometryType(uint32_t geometryID) const
    {
        // Map global geometry ID to which type of geometry it represents.
        if (geometryID < mMeshDesc.size()) return mMeshDesc[geometryID].isDisplaced() ? GeometryType::DisplacedTriangleMesh : GeometryType::TriangleMesh;
        else if (geometryID < mMeshDesc.size() + mCurveDesc.size()) return GeometryType::Curve;
        else if (geometryID < mMeshDesc.size() + mCurveDesc.size() + mSDFGridDesc.size()) return GeometryType::SDFGrid;
        else if (geometryID < mMeshDesc.size() + mCurveDesc.size() + mSDFGridDesc.size() + mCustomPrimitiveDesc.size()) return GeometryType::Custom;
        else throw ArgumentError("'geometryID' is invalid.");
    }

    uint32_t Scene::getSDFGridGeometryCount() const
    {
        switch (mSDFGridConfig.implementation)
        {
        case SDFGrid::Type::None:
            return 0;
        case SDFGrid::Type::NormalizedDenseGrid:
        case SDFGrid::Type::SparseVoxelOctree:
            return mSDFGrids.empty() ? 0 : 1;
        case SDFGrid::Type::SparseVoxelSet:
        case SDFGrid::Type::SparseBrickSet:
            return (uint32_t)mSDFGrids.size();
        default:
            should_not_get_here();
            return 0;
        }
    }

    Material::SharedPtr Scene::getGeometryMaterial(uint32_t geometryID) const
    {
        if (geometryID < mMeshDesc.size())
        {
            return mpMaterials->getMaterial(mMeshDesc[geometryID].materialID);
        }
        geometryID -= (uint32_t)mMeshDesc.size();

        if (geometryID < mCurveDesc.size())
        {
            return mpMaterials->getMaterial(mCurveDesc[geometryID].materialID);
        }
        geometryID -= (uint32_t)mCurveDesc.size();

        if (geometryID < mSDFGridDesc.size())
        {
            return mpMaterials->getMaterial(mSDFGridDesc[geometryID].materialID);
        }
        geometryID -= (uint32_t)mSDFGridDesc.size();

        if (geometryID < mCustomPrimitiveDesc.size())
        {
            return nullptr;
        }
        geometryID -= (uint32_t)mCustomPrimitiveDesc.size();

        throw ArgumentError("'geometryID' is invalid.");
    }

    uint32_t Scene::getCustomPrimitiveIndex(uint32_t geometryID) const
    {
        if (getGeometryType(geometryID) != GeometryType::Custom)
        {
            throw ArgumentError("'geometryID' ({}) does not refer to a custom primitive.", geometryID);
        }

        size_t customPrimitiveOffset = mMeshDesc.size() + mCurveDesc.size() + mSDFGridDesc.size();
        assert(geometryID >= (uint32_t)customPrimitiveOffset && geometryID < getGeometryCount());
        return geometryID - (uint32_t)customPrimitiveOffset;
    }

    const CustomPrimitiveDesc& Scene::getCustomPrimitive(uint32_t index) const
    {
        if (index >= getCustomPrimitiveCount())
        {
            throw ArgumentError("'index' ({}) is out of range.", index);
        }
        return mCustomPrimitiveDesc[index];
    }

    const AABB& Scene::getCustomPrimitiveAABB(uint32_t index) const
    {
        if (index >= getCustomPrimitiveCount())
        {
            throw ArgumentError("'index' ({}) is out of range.", index);
        }
        return mCustomPrimitiveAABBs[index];
    }

    uint32_t Scene::addCustomPrimitive(uint32_t userID, const AABB& aabb)
    {
        // Currently each custom primitive has exactly one AABB. This may change in the future.
        assert(mCustomPrimitiveDesc.size() == mCustomPrimitiveAABBs.size());
        if (mCustomPrimitiveAABBs.size() > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Custom primitive count exceeds the maximum");
        }

        const uint32_t index = (uint32_t)mCustomPrimitiveDesc.size();

        CustomPrimitiveDesc desc = {};
        desc.userID = userID;
        desc.aabbOffset = (uint32_t)mCustomPrimitiveAABBs.size();

        mCustomPrimitiveDesc.push_back(desc);
        mCustomPrimitiveAABBs.push_back(aabb);
        mCustomPrimitivesChanged = true;

        return index;
    }

    void Scene::removeCustomPrimitives(uint32_t first, uint32_t last)
    {
        if (first > last || last > getCustomPrimitiveCount())
        {
            throw ArgumentError("'first' ({}) and 'last' ({}) is not a valid range of custom primitives.", first, last);
        }

        if (first == last) return;

        mCustomPrimitiveDesc.erase(mCustomPrimitiveDesc.begin() + first, mCustomPrimitiveDesc.begin() + last);
        mCustomPrimitiveAABBs.erase(mCustomPrimitiveAABBs.begin() + first, mCustomPrimitiveAABBs.begin() + last);

        // Update AABB offsets for all subsequent primitives.
        // The offset is currently redundant since there is one AABB per primitive. This may change in the future.
        for (uint32_t i = first; i < mCustomPrimitiveDesc.size(); i++)
        {
            mCustomPrimitiveDesc[i].aabbOffset = i;
        }

        mCustomPrimitivesChanged = true;
    }

    void Scene::updateCustomPrimitive(uint32_t index, const AABB& aabb)
    {
        if (index >= getCustomPrimitiveCount())
        {
            throw ArgumentError("'index' ({}) is out of range.", index);
        }

        if (mCustomPrimitiveAABBs[index] != aabb)
        {
            mCustomPrimitiveAABBs[index] = aabb;
            mCustomPrimitivesMoved = true;
        }
    }

    GridVolume::SharedPtr Scene::getGridVolumeByName(const std::string& name) const
    {
        for (const auto& v : mGridVolumes)
        {
            if (v->getName() == name) return v;
        }

        return nullptr;
    }

    Light::SharedPtr Scene::getLightByName(const std::string& name) const
    {
        for (const auto& l : mLights)
        {
            if (l->getName() == name) return l;
        }

        return nullptr;
    }

    void Scene::toggleAnimations(bool animate)
    {
        for (auto& light : mLights) light->setIsAnimated(animate);
        for (auto& camera : mCameras) camera->setIsAnimated(animate);
        mpAnimationController->setEnabled(animate);
    }

    void Scene::setBlasUpdateMode(UpdateMode mode)
    {
        if (mode != mBlasUpdateMode) mRebuildBlas = true;
        mBlasUpdateMode = mode;
    }

    void Scene::createDrawList()
    {
        // This function creates argument buffers for draw indirect calls to rasterize the scene.
        // The updateGeometryInstances() function must have been called before so that the flags are accurate.
        //
        // Note that we create four draw buffers to handle all combinations of:
        // 1) mesh is using 16- or 32-bit indices,
        // 2) mesh triangle winding is CW or CCW after transformation.
        //
        // TODO: Update the draw args if a mesh undergoes animation that flips the winding.

        mDrawArgs.clear();

        // Helper to create the draw-indirect buffer.
        auto createDrawBuffer = [this](const auto& drawMeshes, bool ccw, ResourceFormat ibFormat = ResourceFormat::Unknown)
        {
            if (drawMeshes.size() > 0)
            {
                DrawArgs draw;
                draw.pBuffer = Buffer::create(sizeof(drawMeshes[0]) * drawMeshes.size(), Resource::BindFlags::IndirectArg, Buffer::CpuAccess::None, drawMeshes.data());
                draw.pBuffer->setName("Scene draw buffer");
                assert(drawMeshes.size() <= std::numeric_limits<uint32_t>::max());
                draw.count = (uint32_t)drawMeshes.size();
                draw.ccw = ccw;
                draw.ibFormat = ibFormat;
                mDrawArgs.push_back(draw);
            }
        };

        if (hasIndexBuffer())
        {
            std::vector<DrawIndexedArguments> drawClockwiseMeshes[2], drawCounterClockwiseMeshes[2];

            uint32_t instanceID = 0;
            for (const auto& instance : mGeometryInstanceData)
            {
                if (instance.getType() != GeometryType::TriangleMesh) continue;

                const auto& mesh = mMeshDesc[instance.geometryID];
                bool use16Bit = mesh.use16BitIndices();

                DrawIndexedArguments draw;
                draw.IndexCountPerInstance = mesh.indexCount;
                draw.InstanceCount = 1;
                draw.StartIndexLocation = mesh.ibOffset * (use16Bit ? 2 : 1);
                draw.BaseVertexLocation = mesh.vbOffset;
                draw.StartInstanceLocation = instanceID++;

                int i = use16Bit ? 0 : 1;
                (instance.isWorldFrontFaceCW()) ? drawClockwiseMeshes[i].push_back(draw) : drawCounterClockwiseMeshes[i].push_back(draw);
            }

            createDrawBuffer(drawClockwiseMeshes[0], false, ResourceFormat::R16Uint);
            createDrawBuffer(drawClockwiseMeshes[1], false, ResourceFormat::R32Uint);
            createDrawBuffer(drawCounterClockwiseMeshes[0], true, ResourceFormat::R16Uint);
            createDrawBuffer(drawCounterClockwiseMeshes[1], true, ResourceFormat::R32Uint);
        }
        else
        {
            std::vector<DrawArguments> drawClockwiseMeshes, drawCounterClockwiseMeshes;

            uint32_t instanceID = 0;
            for (const auto& instance : mGeometryInstanceData)
            {
                if (instance.getType() != GeometryType::TriangleMesh) continue;

                const auto& mesh = mMeshDesc[instance.geometryID];
                assert(mesh.indexCount == 0);

                DrawArguments draw;
                draw.VertexCountPerInstance = mesh.vertexCount;
                draw.InstanceCount = 1;
                draw.StartVertexLocation = mesh.vbOffset;
                draw.StartInstanceLocation = instanceID++;

                (instance.isWorldFrontFaceCW()) ? drawClockwiseMeshes.push_back(draw) : drawCounterClockwiseMeshes.push_back(draw);
            }

            createDrawBuffer(drawClockwiseMeshes, false);
            createDrawBuffer(drawCounterClockwiseMeshes, true);
        }
    }

    void Scene::initGeomDesc(RenderContext* pContext)
    {
        assert(mBlasData.empty());

        const VertexBufferLayout::SharedConstPtr& pVbLayout = mpVao->getVertexLayout()->getBufferLayout(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pVb = mpVao->getVertexBuffer(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pIb = mpVao->getIndexBuffer();
        const auto& globalMatrices = mpAnimationController->getGlobalMatrices();

        // Normally static geometry is already pre-transformed to world space by the SceneBuilder,
        // but if that isn't the case, we let DXR transform static geometry as part of the BLAS build.
        // For this we need the GPU address of the transform matrix of each mesh in row-major format.
        // Since glm uses column-major format we lazily create a buffer with the transposed matrices.
        // Note that this is sufficient to do once only as the transforms for static meshes can't change.
        // TODO: Use AnimationController's matrix buffer directly when we've switched to a row-major matrix library.
        auto getStaticMatricesBuffer = [&]()
        {
            if (!mpBlasStaticWorldMatrices)
            {
                std::vector<glm::mat4> transposedMatrices;
                transposedMatrices.reserve(globalMatrices.size());
                for (const auto& m : globalMatrices) transposedMatrices.push_back(glm::transpose(m));

                uint32_t float4Count = (uint32_t)transposedMatrices.size() * 4;
                mpBlasStaticWorldMatrices = Buffer::createStructured(sizeof(float4), float4Count, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, transposedMatrices.data(), false);
                mpBlasStaticWorldMatrices->setName("Scene::mpBlasStaticWorldMatrices");

                // Transition the resource to non-pixel shader state as expected by DXR.
                pContext->resourceBarrier(mpBlasStaticWorldMatrices.get(), Resource::State::NonPixelShader);
            }
            return mpBlasStaticWorldMatrices;
        };

        assert(mMeshGroups.size() > 0);
        uint32_t totalBlasCount = (uint32_t)mMeshGroups.size() + (mCurveDesc.empty() ? 0 : 1) + getSDFGridGeometryCount() + (mCustomPrimitiveDesc.empty() ? 0 : 1); // If there are procedural primitives, they are all placed in one more BLAS.

        mBlasData.resize(totalBlasCount);
        mRebuildBlas = true;
        mHasSkinnedMesh = false;
        mHasAnimatedVertexCache = false;

        // Iterate over the mesh groups. One BLAS will be created for each group.
        // Each BLAS may contain multiple geometries.
        for (size_t i = 0; i < mMeshGroups.size(); i++)
        {
            const auto& meshList = mMeshGroups[i].meshList;
            const bool isStatic = mMeshGroups[i].isStatic;
            const bool isDisplaced = mMeshGroups[i].isDisplaced;
            auto& blas = mBlasData[i];
            auto& geomDescs = blas.geomDescs;
            geomDescs.resize(meshList.size());
            blas.hasProceduralPrimitives = false;

            // Track what types of triangle winding exist in the final BLAS.
            // The SceneBuilder should have ensured winding is consistent, but keeping the check here as a safeguard.
            uint32_t triangleWindings = 0; // bit 0 indicates CW, bit 1 CCW.

            for (size_t j = 0; j < meshList.size(); j++)
            {
                const uint32_t meshID = meshList[j];
                const MeshDesc& mesh = mMeshDesc[meshID];
                bool frontFaceCW = mesh.isFrontFaceCW();
                blas.hasSkinnedMesh |= mesh.hasDynamicData();

                RtGeometryDesc& desc = geomDescs[j];

                if (!isDisplaced)
                {
                    desc.type = RtGeometryType::Triangles;
                    desc.content.triangles.transform3x4 = 0; // The default is no transform

                    if (isStatic)
                    {
                        // Static meshes will be pre-transformed when building the BLAS.
                        // Lookup the matrix ID here. If it is an identity matrix, no action is needed.
                        assert(mMeshIdToInstanceIds[meshID].size() == 1);
                        uint32_t instanceID = mMeshIdToInstanceIds[meshID][0];
                        assert(instanceID < mGeometryInstanceData.size());
                        uint32_t matrixID = mGeometryInstanceData[instanceID].globalMatrixID;

                        if (globalMatrices[matrixID] != glm::identity<glm::mat4>())
                        {
                            // Get the GPU address of the transform in row-major format.
                            desc.content.triangles.transform3x4 = getStaticMatricesBuffer()->getGpuAddress() + matrixID * 64ull;

                            if (glm::determinant(globalMatrices[matrixID]) < 0.f) frontFaceCW = !frontFaceCW;
                        }
                    }
                    triangleWindings |= frontFaceCW ? 1 : 2;

                    // If this is an opaque mesh, set the opaque flag
                    auto pMaterial = mpMaterials->getMaterial(mesh.materialID);
                    desc.flags = pMaterial->isOpaque() ? RtGeometryFlags::Opaque : RtGeometryFlags::None;

                    // Set the position data
                    desc.content.triangles.vertexData = pVb->getGpuAddress() + (mesh.vbOffset * pVbLayout->getStride());
                    desc.content.triangles.vertexStride = pVbLayout->getStride();
                    desc.content.triangles.vertexCount = mesh.vertexCount;
                    desc.content.triangles.vertexFormat = pVbLayout->getElementFormat(0);

                    // Set index data
                    if (pIb)
                    {
                        // The global index data is stored in a dword array.
                        // Each mesh specifies whether its indices are in 16-bit or 32-bit format.
                        ResourceFormat ibFormat = mesh.use16BitIndices() ? ResourceFormat::R16Uint : ResourceFormat::R32Uint;
                        desc.content.triangles.indexData = pIb->getGpuAddress() + mesh.ibOffset * sizeof(uint32_t);
                        desc.content.triangles.indexCount = mesh.indexCount;
                        desc.content.triangles.indexFormat = ibFormat;
                    }
                    else
                    {
                        assert(mesh.indexCount == 0);
                        desc.content.triangles.indexData = NULL;
                        desc.content.triangles.indexCount = 0;
                        desc.content.triangles.indexFormat = ResourceFormat::Unknown;
                    }
                }
                else
                {
                    // Displaced triangle mesh, requires custom intersection.
                    desc.type = RtGeometryType::ProcedurePrimitives;
                    desc.flags = RtGeometryFlags::Opaque;

                    desc.content.proceduralAABBs.count = mDisplacement.meshData[meshID].AABBCount;
                    uint64_t bbStartOffset = mDisplacement.meshData[meshID].AABBOffset * sizeof(RtAABB);
                    desc.content.proceduralAABBs.data = mDisplacement.pAABBBuffer->getGpuAddress() + bbStartOffset;
                    desc.content.proceduralAABBs.stride = sizeof(RtAABB);
                }
            }

            mHasSkinnedMesh |= blas.hasSkinnedMesh;
            assert(!(isStatic && mHasSkinnedMesh));

            if (triangleWindings == 0x3)
            {
                logWarning("Mesh group " + std::to_string(i) + " has mixed triangle winding. Back/front face culling won't work correctly.");
            }
        }

        // Procedural primitives other than displaced triangle meshes and SDF grids are placed in two BLASes at the end.
        // The geometries in these BLASes are using the following layout:
        //
        //  +----------+----------+-----+----------+
        //  |          |          |     |          |
        //  |  Curve0  |  Curve1  | ... |  CurveM  |
        //  |          |          |     |          |
        //  |          |          |     |          |
        //  +----------+----------+-----+----------+
        // SDF grids either create a shared BLAS or one BLAS per SDF grid:
        //  +----------+          +----------+ +----------+     +----------+
        //  |          |          |          | |          |     |          |
        //  | SDFGrid  |          | SDFGrid0 | | SDFGrid1 | ... | SDFGridN |
        //  |  Shared  |    or    |          | |          |     |          |
        //  | Geometry |          |          | |          |     |          |
        //  +----------+          +----------+ +----------+     +----------+
        //
        //  +----------+----------+-----+----------+
        //  |          |          |     |          |
        //  |  Custom  |  Custom  | ... |  Custom  |
        //  |  Prim0   |  Prim1   |     |  PrimN   |
        //  |          |          |     |          |
        //  +----------+----------+-----+----------+
        //
        // Each procedural primitive indexes a range of AABBs in a global AABB buffer.
        //
        size_t blasDataIndex = mMeshGroups.size();
        uint64_t bbAddressOffset = 0;
        if (!mCurveDesc.empty())
        {
            assert(mpRtAABBBuffer && mpRtAABBBuffer->getElementCount() >= mRtAABBRaw.size());

            auto& blas = mBlasData[blasDataIndex++];
            blas.geomDescs.resize(mCurveDesc.size());
            blas.hasProceduralPrimitives = true;

            blas.hasAnimatedVertexCache |= mpAnimationController->hasAnimatedVertexCaches();
            mHasAnimatedVertexCache |= blas.hasAnimatedVertexCache;

            uint32_t geomIndexOffset = 0;

            for (const auto& curve : mCurveDesc)
            {
                // One geometry desc per curve.
                RtGeometryDesc& desc = blas.geomDescs[geomIndexOffset++];

                desc.type = RtGeometryType::ProcedurePrimitives;
                desc.flags = RtGeometryFlags::Opaque;
                desc.content.proceduralAABBs.count = curve.indexCount;
                desc.content.proceduralAABBs.data = mpRtAABBBuffer->getGpuAddress() + bbAddressOffset;
                desc.content.proceduralAABBs.stride = sizeof(RtAABB);

                bbAddressOffset += sizeof(RtAABB) * curve.indexCount;
            }
        }

        if (!mSDFGrids.empty())
        {
            if (mSDFGridConfig.implementation == SDFGrid::Type::NormalizedDenseGrid ||
                mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelOctree)
            {
                // All ND SDF Grid instances share the same BLAS and AABB buffer.
                const SDFGrid::SharedPtr& pSDFGrid = mSDFGrids.back();

                auto& blas = mBlasData[blasDataIndex];
                blas.hasProceduralPrimitives = true;
                blas.geomDescs.resize(1);

                RtGeometryDesc& desc = blas.geomDescs.back();
                desc.type = RtGeometryType::ProcedurePrimitives;
                desc.flags = RtGeometryFlags::Opaque;
                desc.content.proceduralAABBs.count = pSDFGrid->getAABBCount();
                desc.content.proceduralAABBs.data = pSDFGrid->getAABBBuffer()->getGpuAddress();
                desc.content.proceduralAABBs.stride = sizeof(RtAABB);

                blasDataIndex++;
            }
            else if (mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelSet ||
                     mSDFGridConfig.implementation == SDFGrid::Type::SparseBrickSet)
            {
                for (uint32_t s = 0; s < mSDFGrids.size(); s++)
                {
                    const SDFGrid::SharedPtr& pSDFGrid = mSDFGrids[s];

                    auto& blas = mBlasData[blasDataIndex + s];
                    blas.hasProceduralPrimitives = true;
                    blas.geomDescs.resize(1);

                    RtGeometryDesc& desc = blas.geomDescs.back();
                    desc.type = RtGeometryType::ProcedurePrimitives;
                    desc.flags = RtGeometryFlags::Opaque;
                    desc.content.proceduralAABBs.count = pSDFGrid->getAABBCount();
                    desc.content.proceduralAABBs.data = pSDFGrid->getAABBBuffer()->getGpuAddress();
                    desc.content.proceduralAABBs.stride = sizeof(RtAABB);

                    assert(desc.content.proceduralAABBs.count > 0);
                }

                blasDataIndex += mSDFGrids.size();
            }
        }

        if (!mCustomPrimitiveDesc.empty())
        {
            assert(mpRtAABBBuffer && mpRtAABBBuffer->getElementCount() >= mRtAABBRaw.size());

            auto& blas = mBlasData.back();
            blas.geomDescs.resize(mCustomPrimitiveDesc.size());
            blas.hasProceduralPrimitives = true;

            uint32_t geomIndexOffset = 0;

            for (const auto& customPrim : mCustomPrimitiveDesc)
            {
                RtGeometryDesc& desc = blas.geomDescs[geomIndexOffset++];
                desc.type = RtGeometryType::ProcedurePrimitives;
                desc.flags = RtGeometryFlags::None;

                desc.content.proceduralAABBs.count = 1; // Currently only one AABB per user-defined prim supported
                desc.content.proceduralAABBs.data = mpRtAABBBuffer->getGpuAddress() + bbAddressOffset;
                desc.content.proceduralAABBs.stride = sizeof(RtAABB);

                bbAddressOffset += sizeof(RtAABB);
            }
        }

        // Verify that the total geometry count matches the expectation.
        size_t totalGeometries = 0;
        for (const auto& blas : mBlasData) totalGeometries += blas.geomDescs.size();
        if (totalGeometries != getGeometryCount()) throw RuntimeError("Total geometry count mismatch");
    }

    void Scene::preparePrebuildInfo(RenderContext* pContext)
    {
        for (auto& blas : mBlasData)
        {
            // Determine how BLAS build/update should be done.
            // The default choice is to compact all static BLASes and those that don't need to be rebuilt every frame.
            // For all other BLASes, compaction just adds overhead.
            // TODO: Add compaction on/off switch for profiling.
            // TODO: Disable compaction for skinned meshes if update performance becomes a problem.
            blas.updateMode = mBlasUpdateMode;
            blas.useCompaction = (!blas.hasSkinnedMesh && !blas.hasAnimatedVertexCache) || blas.updateMode != UpdateMode::Rebuild;

            // Setup build parameters.
            RtAccelerationStructureBuildInputs& inputs = blas.buildInputs;
            inputs.kind = RtAccelerationStructureKind::BottomLevel;
            inputs.descCount = (uint32_t)blas.geomDescs.size();
            inputs.geometryDescs = blas.geomDescs.data();
            inputs.flags = RtAccelerationStructureBuildFlags::None;

            // Add necessary flags depending on settings.
            if (blas.useCompaction)
            {
                inputs.flags |= RtAccelerationStructureBuildFlags::AllowCompaction;
            }
            if ((blas.hasSkinnedMesh || blas.hasProceduralPrimitives) && blas.updateMode == UpdateMode::Refit)
            {
                inputs.flags |= RtAccelerationStructureBuildFlags::AllowUpdate;
            }
            // Set optional performance hints.
            // TODO: Set FAST_BUILD for skinned meshes if update/rebuild performance becomes a problem.
            // TODO: Add FAST_TRACE on/off switch for profiling. It is disabled by default as it is scene-dependent.
            //if (!blas.hasSkinnedMesh)
            //{
            //    inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            //}

            // Get prebuild info.
            blas.prebuildInfo = RtAccelerationStructure::getPrebuildInfo(inputs);

            // Figure out the padded allocation sizes to have proper alignment.
            assert(blas.prebuildInfo.resultDataMaxSize > 0);
            blas.resultByteSize = align_to(kAccelerationStructureByteAlignment, blas.prebuildInfo.resultDataMaxSize);

            uint64_t scratchByteSize = std::max(blas.prebuildInfo.scratchDataSize, blas.prebuildInfo.updateScratchDataSize);
            blas.scratchByteSize = align_to(kAccelerationStructureByteAlignment, scratchByteSize);
        }
    }

    void Scene::computeBlasGroups()
    {
        mBlasGroups.clear();
        uint64_t groupSize = 0;

        for (uint32_t blasId = 0; blasId < mBlasData.size(); blasId++)
        {
            auto& blas = mBlasData[blasId];
            size_t blasSize = blas.resultByteSize + blas.scratchByteSize;

            // Start new BLAS group on first iteration or if group size would exceed the target.
            if (groupSize == 0 || groupSize + blasSize > kMaxBLASBuildMemory)
            {
                mBlasGroups.push_back({});
                groupSize = 0;
            }

            // Add BLAS to current group.
            assert(mBlasGroups.size() > 0);
            auto& group = mBlasGroups.back();
            group.blasIndices.push_back(blasId);
            blas.blasGroupIndex = (uint32_t)mBlasGroups.size() - 1;

            // Update data offsets and sizes.
            blas.resultByteOffset = group.resultByteSize;
            blas.scratchByteOffset = group.scratchByteSize;
            group.resultByteSize += blas.resultByteSize;
            group.scratchByteSize += blas.scratchByteSize;

            groupSize += blasSize;
        }

        // Validation that all offsets and sizes are correct.
        uint64_t totalResultSize = 0;
        uint64_t totalScratchSize = 0;
        std::set<uint32_t> blasIDs;

        for (size_t blasGroupIndex = 0; blasGroupIndex < mBlasGroups.size(); blasGroupIndex++)
        {
            uint64_t resultSize = 0;
            uint64_t scratchSize = 0;

            const auto& group = mBlasGroups[blasGroupIndex];
            assert(!group.blasIndices.empty());

            for (auto blasId : group.blasIndices)
            {
                assert(blasId < mBlasData.size());
                const auto& blas = mBlasData[blasId];

                assert(blasIDs.insert(blasId).second);
                assert(blas.blasGroupIndex == blasGroupIndex);

                assert(blas.resultByteSize > 0);
                assert(blas.resultByteOffset == resultSize);
                resultSize += blas.resultByteSize;

                assert(blas.scratchByteSize > 0);
                assert(blas.scratchByteOffset == scratchSize);
                scratchSize += blas.scratchByteSize;

                assert(blas.blasByteOffset == 0);
                assert(blas.blasByteSize == 0);
            }

            assert(resultSize == group.resultByteSize);
            assert(scratchSize == group.scratchByteSize);
        }
        assert(blasIDs.size() == mBlasData.size());
    }

    void Scene::buildBlas(RenderContext* pContext)
    {
        FALCOR_PROFILE("buildBlas");

        if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::Raytracing))
        {
            throw RuntimeError("Raytracing is not supported by the current device");
        }

        // Add barriers for the VB and IB which will be accessed by the build.
        const Buffer::SharedPtr& pVb = mpVao->getVertexBuffer(kStaticDataBufferIndex);
        const Buffer::SharedPtr& pIb = mpVao->getIndexBuffer();
        pContext->resourceBarrier(pVb.get(), Resource::State::NonPixelShader);
        if (pIb) pContext->resourceBarrier(pIb.get(), Resource::State::NonPixelShader);
        if (mpRtAABBBuffer) pContext->resourceBarrier(mpRtAABBBuffer.get(), Resource::State::NonPixelShader);

        if (!mSDFGrids.empty())
        {
            if (mSDFGridConfig.implementation == SDFGrid::Type::NormalizedDenseGrid ||
                mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelOctree)
            {
                pContext->resourceBarrier(mSDFGrids.back()->getAABBBuffer().get(), Resource::State::NonPixelShader);
            }
            else if (mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelSet ||
                     mSDFGridConfig.implementation == SDFGrid::Type::SparseBrickSet)
            {
                for (const SDFGrid::SharedPtr& pSDFGrid : mSDFGrids)
                {
                    pContext->resourceBarrier(pSDFGrid->getAABBBuffer().get(), Resource::State::NonPixelShader);
                }
            }
        }

        if (mpCurveVao)
        {
            const Buffer::SharedPtr& pCurveVb = mpCurveVao->getVertexBuffer(kStaticDataBufferIndex);
            const Buffer::SharedPtr& pCurveIb = mpCurveVao->getIndexBuffer();
            pContext->resourceBarrier(pCurveVb.get(), Resource::State::NonPixelShader);
            pContext->resourceBarrier(pCurveIb.get(), Resource::State::NonPixelShader);
        }

        // On the first time, or if a full rebuild is necessary we will:
        // - Update all build inputs and prebuild info
        // - Compute BLAS groups
        // - Calculate total intermediate buffer sizes
        // - Build all BLASes into an intermediate buffer
        // - Calculate total compacted buffer size
        // - Compact/clone all BLASes to their final location

        if (mRebuildBlas)
        {
            logInfo("Initiating BLAS build for " + std::to_string(mBlasData.size()) + " mesh groups");

            // Invalidate any previous TLASes as they won't be valid anymore.
            mTlasCache.clear();

            // Compute pre-build info per BLAS and organize the BLASes into groups
            // in order to limit GPU memory usage during BLAS build.
            preparePrebuildInfo(pContext);
            computeBlasGroups();

            logInfo("BLAS build split into " + std::to_string(mBlasGroups.size()) + " groups");

            // Compute the required maximum size of the result and scratch buffers.
            uint64_t resultByteSize = 0;
            uint64_t scratchByteSize = 0;
            size_t maxBlasCount = 0;

            for (const auto& group : mBlasGroups)
            {
                resultByteSize = std::max(resultByteSize, group.resultByteSize);
                scratchByteSize = std::max(scratchByteSize, group.scratchByteSize);
                maxBlasCount = std::max(maxBlasCount, group.blasIndices.size());
            }
            assert(resultByteSize > 0 && scratchByteSize > 0);

            logInfo("BLAS build result buffer size: " + formatByteSize(resultByteSize));
            logInfo("BLAS build scratch buffer size: " + formatByteSize(scratchByteSize));

            // Allocate result and scratch buffers.
            // The scratch buffer we'll retain because it's needed for subsequent rebuilds and updates.
            // TODO: Save memory by reducing the scratch buffer to the minimum required for the dynamic objects.
            if (mpBlasScratch == nullptr || mpBlasScratch->getSize() < scratchByteSize)
            {
                mpBlasScratch = Buffer::create(scratchByteSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
                mpBlasScratch->setName("Scene::mpBlasScratch");
            }

            Buffer::SharedPtr pResultBuffer = Buffer::create(resultByteSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
            assert(pResultBuffer && mpBlasScratch);

            // Create post-build info pool for readback.
            RtAccelerationStructurePostBuildInfoPool::Desc compactedSizeInfoPoolDesc;
            compactedSizeInfoPoolDesc.queryType = RtAccelerationStructurePostBuildInfoQueryType::CompactedSize;
            compactedSizeInfoPoolDesc.elementCount = (uint32_t)maxBlasCount;
            RtAccelerationStructurePostBuildInfoPool::SharedPtr compactedSizeInfoPool = RtAccelerationStructurePostBuildInfoPool::create(compactedSizeInfoPoolDesc);

            RtAccelerationStructurePostBuildInfoPool::Desc currentSizeInfoPoolDesc;
            currentSizeInfoPoolDesc.queryType = RtAccelerationStructurePostBuildInfoQueryType::CurrentSize;
            currentSizeInfoPoolDesc.elementCount = (uint32_t)maxBlasCount;
            RtAccelerationStructurePostBuildInfoPool::SharedPtr currentSizeInfoPool = RtAccelerationStructurePostBuildInfoPool::create(currentSizeInfoPoolDesc);

            bool hasSkinnedMesh = false;
            bool hasProceduralPrimitives = false;

            mBlasObjects.resize(mBlasData.size());

            // Iterate over BLAS groups. For each group build and compact all BLASes.
            for (size_t blasGroupIndex = 0; blasGroupIndex < mBlasGroups.size(); blasGroupIndex++)
            {
                auto& group = mBlasGroups[blasGroupIndex];

                // Allocate array to hold intermediate blases for the group.
                std::vector<RtAccelerationStructure::SharedPtr> intermediateBlases(group.blasIndices.size());

                // Insert barriers. The buffers are now ready to be written.
                pContext->uavBarrier(pResultBuffer.get());
                pContext->uavBarrier(mpBlasScratch.get());

                // Reset the post-build info pools to receive new info.
                compactedSizeInfoPool->reset(pContext);
                currentSizeInfoPool->reset(pContext);

                // Build the BLASes into the intermediate result buffer.
                // We output post-build info in order to find out the final size requirements.
                for (size_t i = 0; i < group.blasIndices.size(); ++i)
                {
                    const uint32_t blasId = group.blasIndices[i];
                    const auto& blas = mBlasData[blasId];

                    hasSkinnedMesh |= blas.hasSkinnedMesh;
                    hasProceduralPrimitives |= blas.hasProceduralPrimitives;

                    RtAccelerationStructure::Desc createDesc = {};
                    createDesc.setBuffer(pResultBuffer, blas.resultByteOffset, blas.resultByteSize);
                    createDesc.setKind(RtAccelerationStructureKind::BottomLevel);
                    auto blasObject = RtAccelerationStructure::create(createDesc);
                    intermediateBlases[i] = blasObject;

                    RtAccelerationStructure::BuildDesc asDesc = {};
                    asDesc.inputs = blas.buildInputs;
                    asDesc.scratchData = mpBlasScratch->getGpuAddress() + blas.scratchByteOffset;
                    asDesc.dest = blasObject.get();

                    // Need to find out the post-build compacted BLAS size to know the final allocation size.
                    RtAccelerationStructurePostBuildInfoDesc postbuildInfoDesc = {};
                    if (blas.useCompaction)
                    {
                        postbuildInfoDesc.type = RtAccelerationStructurePostBuildInfoQueryType::CompactedSize;
                        postbuildInfoDesc.index = (uint32_t)i;
                        postbuildInfoDesc.pool = compactedSizeInfoPool.get();
                    }
                    else
                    {
                        postbuildInfoDesc.type = RtAccelerationStructurePostBuildInfoQueryType::CurrentSize;
                        postbuildInfoDesc.index = (uint32_t)i;
                        postbuildInfoDesc.pool = currentSizeInfoPool.get();
                    }

                    pContext->buildAccelerationStructure(asDesc, 1, &postbuildInfoDesc);
                }

                // Read back the calculated final size requirements for each BLAS.

                group.finalByteSize = 0;
                for (size_t i = 0; i < group.blasIndices.size(); i++)
                {
                    const uint32_t blasId = group.blasIndices[i];
                    auto& blas = mBlasData[blasId];

                    // Check the size. Upon failure a zero size may be reported.
                    uint64_t byteSize = 0;
                    if (blas.useCompaction)
                    {
                        byteSize = compactedSizeInfoPool->getElement(pContext, (uint32_t)i);
                    }
                    else
                    {
                        byteSize = currentSizeInfoPool->getElement(pContext, (uint32_t)i);
                        // For platforms that does not support current size query, use prebuild size.
                        if (byteSize == 0)
                        {
                            byteSize = blas.prebuildInfo.resultDataMaxSize;
                        }
                    }
                    assert(byteSize <= blas.prebuildInfo.resultDataMaxSize);
                    if (byteSize == 0) throw RuntimeError("Acceleration structure build failed for BLAS index {}", blasId);

                    blas.blasByteSize = align_to(kAccelerationStructureByteAlignment, byteSize);
                    blas.blasByteOffset = group.finalByteSize;
                    group.finalByteSize += blas.blasByteSize;
                }
                assert(group.finalByteSize > 0);

                logInfo("BLAS group " + std::to_string(blasGroupIndex) + " final size: " + formatByteSize(group.finalByteSize));

                // Allocate final BLAS buffer.
                auto& pBlas = group.pBlas;
                if (pBlas == nullptr || pBlas->getSize() < group.finalByteSize)
                {
                    pBlas = Buffer::create(group.finalByteSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
                    pBlas->setName("Scene::mBlasGroups[" + std::to_string(blasGroupIndex) + "].pBlas");
                }
                else
                {
                    // If we didn't need to reallocate, just insert a barrier so it's safe to use.
                    pContext->uavBarrier(pBlas.get());
                }

                // Insert barrier. The result buffer is now ready to be consumed.
                // TOOD: This is probably not necessary since we flushed above, but it's not going to hurt.
                pContext->uavBarrier(pResultBuffer.get());

                // Compact/clone all BLASes to their final location.
                for (size_t i = 0; i < group.blasIndices.size(); ++i)
                {
                    const uint32_t blasId = group.blasIndices[i];
                    auto& blas = mBlasData[blasId];

                    RtAccelerationStructure::Desc blasDesc = {};
                    blasDesc.setBuffer(pBlas, blas.blasByteOffset, blas.blasByteSize);
                    blasDesc.setKind(RtAccelerationStructureKind::BottomLevel);
                    mBlasObjects[blasId] = RtAccelerationStructure::create(blasDesc);

                    pContext->copyAccelerationStructure(
                        mBlasObjects[blasId].get(),
                        intermediateBlases[i].get(),
                        blas.useCompaction ? RenderContext::RtAccelerationStructureCopyMode::Compact : RenderContext::RtAccelerationStructureCopyMode::Clone);
                }

                // Insert barrier. The BLAS buffer is now ready for use.
                pContext->uavBarrier(pBlas.get());
            }

            // Release scratch buffer if there is no animated content. We will not need it.
            if (!hasSkinnedMesh && !hasProceduralPrimitives) mpBlasScratch.reset();

            updateRaytracingBLASStats();
            mRebuildBlas = false;
            return;
        }

        // If we get here, all BLASes have previously been built and compacted. We will:
        // - Skip the ones that have no animated geometries.
        // - Update or rebuild in-place the ones that are animated.

        assert(!mRebuildBlas);
        bool updateProcedural = is_set(mUpdates, UpdateFlags::CurvesMoved) || is_set(mUpdates, UpdateFlags::CustomPrimitivesMoved);

        for (const auto& group : mBlasGroups)
        {
            // Determine if any BLAS in the group needs to be updated.
            bool needsUpdate = false;
            for (uint32_t blasId : group.blasIndices)
            {
                const auto& blas = mBlasData[blasId];
                if (blas.hasProceduralPrimitives && updateProcedural) needsUpdate = true;
                if (!blas.hasProceduralPrimitives && blas.hasSkinnedMesh) needsUpdate = true;
            }

            if (!needsUpdate) continue;

            // At least one BLAS in the group needs to be updated.
            // Insert barriers. The buffers are now ready to be written.
            auto& pBlas = group.pBlas;
            assert(pBlas && mpBlasScratch);
            pContext->uavBarrier(pBlas.get());
            pContext->uavBarrier(mpBlasScratch.get());

            // Iterate over all BLASes in group.
            for (uint32_t blasId : group.blasIndices)
            {
                const auto& blas = mBlasData[blasId];

                // Skip BLASes that do not need to be updated.
                if (blas.hasProceduralPrimitives && !updateProcedural) continue;
                if (!blas.hasProceduralPrimitives && !blas.hasSkinnedMesh) continue;

                // Rebuild/update BLAS.
                RtAccelerationStructure::BuildDesc asDesc = {};
                asDesc.inputs = blas.buildInputs;
                asDesc.scratchData = mpBlasScratch->getGpuAddress() + blas.scratchByteOffset;
                asDesc.dest = mBlasObjects[blasId].get();

                if (blas.updateMode == UpdateMode::Refit)
                {
                    // Set source address to destination address to update in place.
                    asDesc.source = asDesc.dest;
                    asDesc.inputs.flags |= RtAccelerationStructureBuildFlags::PerformUpdate;
                }
                else
                {
                    // We'll rebuild in place. The BLAS should not be compacted, check that size matches prebuild info.
                    assert(blas.blasByteSize == blas.prebuildInfo.resultDataMaxSize);
                }
                pContext->buildAccelerationStructure(asDesc, 0, nullptr);
            }

            // Insert barrier. The BLAS buffer is now ready for use.
            pContext->uavBarrier(pBlas.get());
        }
    }

    void Scene::fillInstanceDesc(std::vector<RtInstanceDesc>& instanceDescs, uint32_t rayCount, bool perMeshHitEntry) const
    {
        instanceDescs.clear();
        uint32_t instanceContributionToHitGroupIndex = 0;
        uint32_t instanceID = 0;

        for (size_t i = 0; i < mMeshGroups.size(); i++)
        {
            const auto& meshList = mMeshGroups[i].meshList;
            const bool isStatic = mMeshGroups[i].isStatic;

            assert(mBlasData[i].blasGroupIndex < mBlasGroups.size());
            const auto& pBlas = mBlasGroups[mBlasData[i].blasGroupIndex].pBlas;
            assert(pBlas);

            RtInstanceDesc desc = {};
            desc.accelerationStructure = pBlas->getGpuAddress() + mBlasData[i].blasByteOffset;
            desc.instanceMask = 0xFF;
            desc.instanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;

            instanceContributionToHitGroupIndex += rayCount * (uint32_t)meshList.size();

            // We expect all meshes in a group to have identical triangle winding. Verify that assumption here.
            assert(!meshList.empty());
            const bool frontFaceCW = mMeshDesc[meshList[0]].isFrontFaceCW();
            for (size_t i = 1; i < meshList.size(); i++)
            {
                assert(mMeshDesc[meshList[i]].isFrontFaceCW() == frontFaceCW);
            }

            // Set the triangle winding for the instance if it differs from the default.
            // The default in DXR is that a triangle is front facing if its vertices appear clockwise
            // from the ray origin, in object space in a left-handed coordinate system.
            // Note that Falcor uses a right-handed coordinate system, so we have to invert the flag.
            // Since these winding direction rules are defined in object space, they are unaffected by instance transforms.
            if (frontFaceCW) desc.flags = desc.flags | RtGeometryInstanceFlags::TriangleFrontCounterClockwise;

            // From the scene builder we can expect the following:
            //
            // If BLAS is marked as static:
            // - The meshes are pre-transformed to world-space.
            // - The meshes are guaranteed to be non-instanced, so only one INSTANCE_DESC with an identity transform is needed.
            //
            // If BLAS is not marked as static:
            // - The meshes are guaranteed to be non-instanced or be identically instanced, one INSTANCE_DESC per TLAS instance is needed.
            // - The global matrices are the same for all meshes in an instance.
            //
            assert(!meshList.empty());
            size_t instanceCount = mMeshIdToInstanceIds[meshList[0]].size();

            assert(instanceCount > 0);
            for (size_t instanceIdx = 0; instanceIdx < instanceCount; instanceIdx++)
            {
                // Validate that the ordering is matching our expectations:
                // InstanceID() + GeometryIndex() should look up the correct mesh instance.
                for (uint32_t geometryIndex = 0; geometryIndex < (uint32_t)meshList.size(); geometryIndex++)
                {
                    const auto& instances = mMeshIdToInstanceIds[meshList[geometryIndex]];
                    assert(instances.size() == instanceCount);
                    assert(instances[instanceIdx] == instanceID + geometryIndex);
                }

                desc.instanceID = instanceID;
                instanceID += (uint32_t)meshList.size();

                glm::mat4 transform4x4 = glm::identity<glm::mat4>();
                if (!isStatic)
                {
                    // For non-static meshes, the matrices for all meshes in an instance are guaranteed to be the same.
                    // Just pick the matrix from the first mesh.
                    const uint32_t matrixId = mGeometryInstanceData[desc.instanceID].globalMatrixID;
                    transform4x4 = transpose(mpAnimationController->getGlobalMatrices()[matrixId]);

                    // Verify that all meshes have matching tranforms.
                    for (uint32_t geometryIndex = 0; geometryIndex < (uint32_t)meshList.size(); geometryIndex++)
                    {
                        assert(matrixId == mGeometryInstanceData[desc.instanceID + geometryIndex].globalMatrixID);
                    }
                }
                std::memcpy(desc.transform, &transform4x4, sizeof(desc.transform));

                // Verify that instance data has the correct instanceIndex and geometryIndex.
                for (uint32_t geometryIndex = 0; geometryIndex < (uint32_t)meshList.size(); geometryIndex++)
                {
                    assert((uint32_t)instanceDescs.size() == mGeometryInstanceData[desc.instanceID + geometryIndex].instanceIndex);
                    assert(geometryIndex == mGeometryInstanceData[desc.instanceID + geometryIndex].geometryIndex);
                }

                instanceDescs.push_back(desc);
            }
        }

        uint32_t totalBlasCount = (uint32_t)mMeshGroups.size() + (mCurveDesc.empty() ? 0 : 1) + getSDFGridGeometryCount() + (mCustomPrimitiveDesc.empty() ? 0 : 1);
        assert((uint32_t)mBlasData.size() == totalBlasCount);

        size_t blasDataIndex = mMeshGroups.size();
        // One instance for curves.
        if (!mCurveDesc.empty())
        {
            const auto& blasData = mBlasData[blasDataIndex++];
            assert(blasData.blasGroupIndex < mBlasGroups.size());
            const auto& pBlas = mBlasGroups[blasData.blasGroupIndex].pBlas;
            assert(pBlas);

            RtInstanceDesc desc = {};
            desc.accelerationStructure = pBlas->getGpuAddress() + blasData.blasByteOffset;
            desc.instanceMask = 0xFF;
            desc.instanceID = instanceID;
            instanceID += (uint32_t)mCurveDesc.size();

            // Start procedural primitive hit group after the triangle hit groups.
            desc.instanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;

            instanceContributionToHitGroupIndex += rayCount * (uint32_t)mCurveDesc.size();

            // For cached curves, the matrices for all curves in an instance are guaranteed to be the same.
            // Just pick the matrix from the first curve.
            auto it = std::find_if(mGeometryInstanceData.begin(), mGeometryInstanceData.end(), [](const auto& inst) { return inst.getType() == GeometryType::Curve; });
            assert(it != mGeometryInstanceData.end());
            const uint32_t matrixId = it->globalMatrixID;
            desc.setTransform(mpAnimationController->getGlobalMatrices()[matrixId]);

            // Verify that instance data has the correct instanceIndex and geometryIndex.
            for (uint32_t geometryIndex = 0; geometryIndex < (uint32_t)mCurveDesc.size(); geometryIndex++)
            {
                assert((uint32_t)instanceDescs.size() == mGeometryInstanceData[desc.instanceID + geometryIndex].instanceIndex);
                assert(geometryIndex == mGeometryInstanceData[desc.instanceID + geometryIndex].geometryIndex);
            }

            instanceDescs.push_back(desc);
        }

        // One instance per SDF grid instance.
        if (!mSDFGrids.empty())
        {
            bool sdfGridInstancesHaveUniqueBLASes = true;
            switch (mSDFGridConfig.implementation)
            {
            case SDFGrid::Type::NormalizedDenseGrid:
            case SDFGrid::Type::SparseVoxelOctree:
                sdfGridInstancesHaveUniqueBLASes = false;
                break;
            case SDFGrid::Type::SparseVoxelSet:
            case SDFGrid::Type::SparseBrickSet:
                sdfGridInstancesHaveUniqueBLASes = true;
                break;
            default:
                should_not_get_here();
            }

            for (const GeometryInstanceData& instance : mGeometryInstanceData)
            {
                if (instance.getType() != GeometryType::SDFGrid) continue;

                const BlasData& blasData = mBlasData[blasDataIndex + (sdfGridInstancesHaveUniqueBLASes ? instance.geometryID : 0)];
                const auto& pBlas = mBlasGroups[blasData.blasGroupIndex].pBlas;

                RtInstanceDesc desc = {};
                desc.accelerationStructure = pBlas->getGpuAddress() + blasData.blasByteOffset;
                desc.instanceMask = 0xFF;
                desc.instanceID = instanceID;
                instanceID++;

                // Start SDF grid hit group after the curve hit groups.
                desc.instanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;

                desc.setTransform(mpAnimationController->getGlobalMatrices()[instance.globalMatrixID]);

                // Verify that instance data has the correct instanceIndex and geometryIndex.
                assert((uint32_t)instanceDescs.size() == instance.instanceIndex);
                assert(0 == instance.geometryIndex);

                instanceDescs.push_back(desc);
            }

            blasDataIndex += (sdfGridInstancesHaveUniqueBLASes ? mSDFGrids.size() : 1);
            instanceContributionToHitGroupIndex += rayCount * (uint32_t)mSDFGridDesc.size();
        }

        // One instance with identity transform for custom primitives.
        if (!mCustomPrimitiveDesc.empty())
        {
            assert(mBlasData.back().blasGroupIndex < mBlasGroups.size());
            const auto& pBlas = mBlasGroups[mBlasData.back().blasGroupIndex].pBlas;
            assert(pBlas);

            RtInstanceDesc desc = {};
            desc.accelerationStructure = pBlas->getGpuAddress() + mBlasData.back().blasByteOffset;
            desc.instanceMask = 0xFF;
            desc.instanceID = instanceID;
            instanceID += (uint32_t)mCustomPrimitiveDesc.size();

            // Start procedural primitive hit group after the curve hit group.
            desc.instanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;

            instanceContributionToHitGroupIndex += rayCount * (uint32_t)mCustomPrimitiveDesc.size();

            glm::mat4 identityMat = glm::identity<glm::mat4>();
            std::memcpy(desc.transform, &identityMat, sizeof(desc.transform));
            instanceDescs.push_back(desc);
        }
    }

    void Scene::buildTlas(RenderContext* pContext, uint32_t rayCount, bool perMeshHitEntry)
    {
        FALCOR_PROFILE("buildTlas");

        TlasData tlas;
        auto it = mTlasCache.find(rayCount);
        if (it != mTlasCache.end()) tlas = it->second;

        fillInstanceDesc(mInstanceDescs, rayCount, perMeshHitEntry);

        RtAccelerationStructureBuildInputs inputs = {};
        inputs.kind = RtAccelerationStructureKind::TopLevel;
        inputs.descCount = (uint32_t)mInstanceDescs.size();
        inputs.flags = RtAccelerationStructureBuildFlags::None;

        // Add build flags for dynamic scenes if TLAS should be updating instead of rebuilt
        if ((mpAnimationController->hasAnimations() || mpAnimationController->hasAnimatedVertexCaches()) && mTlasUpdateMode == UpdateMode::Refit)
        {
            inputs.flags |= RtAccelerationStructureBuildFlags::AllowUpdate;

            // If TLAS has been built already and it was built with ALLOW_UPDATE
            if (tlas.pTlasObject != nullptr && tlas.updateMode == UpdateMode::Refit) inputs.flags |= RtAccelerationStructureBuildFlags::PerformUpdate;
        }

        tlas.updateMode = mTlasUpdateMode;

        // On first build for the scene, create scratch buffer and cache prebuild info. As long as INSTANCE_DESC count doesn't change, we can reuse these
        if (mpTlasScratch == nullptr)
        {
            // Prebuild
            mTlasPrebuildInfo = RtAccelerationStructure::getPrebuildInfo(inputs);
            mpTlasScratch = Buffer::create(mTlasPrebuildInfo.scratchDataSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            mpTlasScratch->setName("Scene::mpTlasScratch");

            // #SCENE This isn't guaranteed according to the spec, and the scratch buffer being stored should be sized differently depending on update mode
            assert(mTlasPrebuildInfo.updateScratchDataSize <= mTlasPrebuildInfo.scratchDataSize);
        }

        // Setup GPU buffers
        RtAccelerationStructure::BuildDesc asDesc = {};
        asDesc.inputs = inputs;

        // If first time building this TLAS
        if (tlas.pTlasObject == nullptr)
        {
            assert(tlas.pInstanceDescs == nullptr); // Instance desc should also be null if no TLAS
            tlas.pTlasBuffer = Buffer::create(mTlasPrebuildInfo.resultDataMaxSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
            tlas.pTlasBuffer->setName("Scene TLAS buffer");
            tlas.pInstanceDescs = Buffer::create((uint32_t)mInstanceDescs.size() * sizeof(RtInstanceDesc), Buffer::BindFlags::None, Buffer::CpuAccess::Write, mInstanceDescs.data());
            tlas.pInstanceDescs->setName("Scene instance descs buffer");

            RtAccelerationStructure::Desc asCreateDesc = {};
            asCreateDesc.setKind(RtAccelerationStructureKind::TopLevel);
            asCreateDesc.setBuffer(tlas.pTlasBuffer, 0, mTlasPrebuildInfo.resultDataMaxSize);
            tlas.pTlasObject = RtAccelerationStructure::create(asCreateDesc);
        }
        // Else update instance descs and barrier TLAS buffers
        else
        {
            assert(mpAnimationController->hasAnimations() || mpAnimationController->hasAnimatedVertexCaches());
            pContext->uavBarrier(tlas.pTlasBuffer.get());
            pContext->uavBarrier(mpTlasScratch.get());
            tlas.pInstanceDescs->setBlob(mInstanceDescs.data(), 0, inputs.descCount * sizeof(RtInstanceDesc));
            asDesc.source = tlas.pTlasObject.get(); // Perform the update in-place
        }

        assert((inputs.descCount != 0) && tlas.pInstanceDescs->getApiHandle() && tlas.pTlasBuffer->getApiHandle() && mpTlasScratch->getApiHandle());

        asDesc.inputs.instanceDescs = tlas.pInstanceDescs->getGpuAddress();
        asDesc.scratchData = mpTlasScratch->getGpuAddress();
        asDesc.dest = tlas.pTlasObject.get();

        // Set the source buffer to update in place if this is an update
        if ((inputs.flags & RtAccelerationStructureBuildFlags::PerformUpdate) != RtAccelerationStructureBuildFlags::None) asDesc.source = asDesc.dest;

        // Create TLAS
        pContext->resourceBarrier(tlas.pInstanceDescs.get(), Resource::State::NonPixelShader);
        pContext->buildAccelerationStructure(asDesc, 0, nullptr);
        pContext->uavBarrier(tlas.pTlasBuffer.get());

        mTlasCache[rayCount] = tlas;
        updateRaytracingTLASStats();
    }

    void Scene::setRaytracingShaderData(RenderContext* pContext, const ShaderVar& var, uint32_t rayTypeCount)
    {
        // On first execution or if BLASes need to be rebuilt, create BLASes for all geometries.
        if (mBlasData.empty())
        {
            initGeomDesc(pContext);
            buildBlas(pContext);
        }

        // On first execution, when meshes have moved, when there's a new ray type count, or when a BLAS has changed, create/update the TLAS
        //
        // The raytracing shader table has one hit record per ray type and geometry. We need to know the ray type count in order to setup the indexing properly.
        // Note that for DXR 1.1 ray queries, the shader table is not used and the ray type count doesn't matter and can be set to zero.
        //
        auto tlasIt = mTlasCache.find(rayTypeCount);
        if (tlasIt == mTlasCache.end())
        {
            // We need a hit entry per mesh right now to pass GeometryIndex()
            buildTlas(pContext, rayTypeCount, true);

            // If new TLAS was just created, get it so the iterator is valid
            if (tlasIt == mTlasCache.end()) tlasIt = mTlasCache.find(rayTypeCount);
        }

        assert(mpSceneBlock);

        // Bind TLAS.
        mpSceneBlock["rtAccel"].setAccelerationStructure(tlasIt->second.pTlasObject);

        // Bind Scene parameter block.
        getCamera()->setShaderData(mpSceneBlock[kCamera]); // TODO REMOVE: Shouldn't be needed anymore?
        var[kParameterBlockName] = mpSceneBlock;
    }

    std::vector<uint32_t> Scene::getMeshBlasIDs() const
    {
        const uint32_t invalidID = uint32_t(-1);
        std::vector<uint32_t> blasIDs(mMeshDesc.size(), invalidID);

        for (uint32_t blasID = 0; blasID < (uint32_t)mMeshGroups.size(); blasID++)
        {
            for (auto meshID : mMeshGroups[blasID].meshList)
            {
                assert(meshID < blasIDs.size());
                blasIDs[meshID] = blasID;
            }
        }

        for (auto blasID : blasIDs) assert(blasID != invalidID);
        return blasIDs;
    }

    uint32_t Scene::getParentNodeID(uint32_t nodeID) const
    {
        if (nodeID >= mSceneGraph.size()) throw ArgumentError("'nodeID' ({}) is out of range", nodeID);
        return mSceneGraph[nodeID].parent;
    }

    void Scene::nullTracePass(RenderContext* pContext, const uint2& dim)
    {
        if (!gpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        {
            logWarning("Raytracing Tier 1.1 is not supported by the current device");
            return;
        }

        RtAccelerationStructureBuildInputs inputs = {};
        inputs.kind = RtAccelerationStructureKind::TopLevel;
        inputs.descCount = 0;
        inputs.flags = RtAccelerationStructureBuildFlags::None;

        RtAccelerationStructurePrebuildInfo prebuildInfo = RtAccelerationStructure::getPrebuildInfo(inputs);

        auto pScratch = Buffer::create(prebuildInfo.scratchDataSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
        auto pTlasBuffer = Buffer::create(prebuildInfo.resultDataMaxSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);

        RtAccelerationStructure::Desc createDesc = {};
        createDesc.setKind(RtAccelerationStructureKind::TopLevel);
        createDesc.setBuffer(pTlasBuffer, 0, prebuildInfo.resultDataMaxSize);
        RtAccelerationStructure::SharedPtr tlasObject = RtAccelerationStructure::create(createDesc);

        RtAccelerationStructure::BuildDesc asDesc = {};
        asDesc.inputs = inputs;
        asDesc.scratchData = pScratch->getGpuAddress();
        asDesc.dest = tlasObject.get();

        pContext->buildAccelerationStructure(asDesc, 0, nullptr);
        pContext->uavBarrier(pTlasBuffer.get());

        Program::Desc desc;
        desc.addShaderLibrary("Scene/NullTrace.cs.slang").csEntry("main").setShaderModel("6_5");
        auto pass = ComputePass::create(desc);
        pass["gOutput"] = Texture::create2D(dim.x, dim.y, ResourceFormat::R8Uint, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess);
        pass["gTlas"].setAccelerationStructure(tlasObject);

        for (size_t i = 0; i < 100; i++)
        {
            pass->execute(pContext, uint3(dim, 1));
        }
    }

    void Scene::setEnvMap(EnvMap::SharedPtr pEnvMap)
    {
        if (mpEnvMap == pEnvMap) return;
        mpEnvMap = pEnvMap;
        mEnvMapChanged = true;
    }

    void Scene::loadEnvMap(const std::string& filename)
    {
        EnvMap::SharedPtr pEnvMap = EnvMap::create(filename);
        setEnvMap(pEnvMap);
    }

    void Scene::setCameraAspectRatio(float ratio)
    {
        getCamera()->setAspectRatio(ratio);
    }

    void Scene::setCameraController(CameraControllerType type)
    {
        if (!mCameraSwitched && mCamCtrlType == type && mpCamCtrl) return;

        auto camera = getCamera();
        switch (type)
        {
        case CameraControllerType::FirstPerson:
            mpCamCtrl = FirstPersonCameraController::create(camera);
            break;
        case CameraControllerType::Orbiter:
            mpCamCtrl = OrbiterCameraController::create(camera);
            ((OrbiterCameraController*)mpCamCtrl.get())->setModelParams(mSceneBB.center(), mSceneBB.radius(), 3.5f);
            break;
        case CameraControllerType::SixDOF:
            mpCamCtrl = SixDoFCameraController::create(camera);
            break;
        default:
            should_not_get_here();
        }
        mpCamCtrl->setCameraSpeed(mCameraSpeed);
    }

    bool Scene::onMouseEvent(const MouseEvent& mouseEvent)
    {
        return mpCamCtrl->onMouseEvent(mouseEvent);
    }

    bool Scene::onKeyEvent(const KeyboardEvent& keyEvent)
    {
        if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            if (!(keyEvent.mods.isAltDown || keyEvent.mods.isCtrlDown || keyEvent.mods.isShiftDown))
            {
                if (keyEvent.key == KeyboardEvent::Key::F3)
                {
                    addViewpoint();
                    return true;
                }
            }
        }
        return mpCamCtrl->onKeyEvent(keyEvent);
    }

    std::string Scene::getScript(const std::string& sceneVar)
    {
        std::string c;

        // Render settings.
        c += ScriptWriter::makeSetProperty(sceneVar, kRenderSettings, mRenderSettings);

        // Animations.
        if (hasAnimation() && !isAnimated())
        {
            c += ScriptWriter::makeSetProperty(sceneVar, kAnimated, false);
        }
        for (size_t i = 0; i < mLights.size(); ++i)
        {
            const auto& light = mLights[i];
            if (light->hasAnimation() && !light->isAnimated())
            {
                c += ScriptWriter::makeSetProperty(sceneVar + "." + kGetLight + "(" + std::to_string(i) + ").", kAnimated, false);
            }
        }

        // Camera.
        if (mSelectedCamera != 0)
        {
            c += sceneVar + "." + kCamera + " = " + sceneVar + "." + kCameras + "[" + std::to_string(mSelectedCamera) + "]\n";
        }
        c += getCamera()->getScript(sceneVar + "." + kCamera);

        // Camera speed.
        c += ScriptWriter::makeSetProperty(sceneVar, kCameraSpeed, mCameraSpeed);

        // Viewpoints.
        if (hasSavedViewpoints())
        {
            for (size_t i = 1; i < mViewpoints.size(); i++)
            {
                auto v = mViewpoints[i];
                c += ScriptWriter::makeMemberFunc(sceneVar, kAddViewpoint, v.position, v.target, v.up, v.index);
            }
        }

        return c;
    }

    pybind11::dict Scene::SceneStats::toPython() const
    {
        pybind11::dict d;

        // Geometry stats
        d["meshCount"] = meshCount;
        d["meshInstanceCount"] = meshInstanceCount;
        d["meshInstanceOpaqueCount"] = meshInstanceOpaqueCount;
        d["transformCount"] = transformCount;
        d["uniqueTriangleCount"] = uniqueTriangleCount;
        d["uniqueVertexCount"] = uniqueVertexCount;
        d["instancedTriangleCount"] = instancedTriangleCount;
        d["instancedVertexCount"] = instancedVertexCount;
        d["indexMemoryInBytes"] = indexMemoryInBytes;
        d["vertexMemoryInBytes"] = vertexMemoryInBytes;
        d["geometryMemoryInBytes"] = geometryMemoryInBytes;
        d["animationMemoryInBytes"] = animationMemoryInBytes;

        // Curve stats
        d["curveCount"] = curveCount;
        d["curveInstanceCount"] = curveInstanceCount;
        d["uniqueCurveSegmentCount"] = uniqueCurveSegmentCount;
        d["uniqueCurvePointCount"] = uniqueCurvePointCount;
        d["instancedCurveSegmentCount"] = instancedCurveSegmentCount;
        d["instancedCurvePointCount"] = instancedCurvePointCount;
        d["curveIndexMemoryInBytes"] = curveIndexMemoryInBytes;
        d["curveVertexMemoryInBytes"] = curveVertexMemoryInBytes;

        // SDF grid stats
        d["sdfGridCount"] = sdfGridCount;
        d["sdfGridDescriptorCount"] = sdfGridDescriptorCount;
        d["sdfGridInstancesCount"] = sdfGridInstancesCount;
        d["sdfGridMemoryInBytes"] = sdfGridMemoryInBytes;

        // Custom primitive stats
        d["customPrimitiveCount"] = customPrimitiveCount;

        // Material stats
        d["materialCount"] = materials.materialCount;
        d["materialOpaqueCount"] = materials.materialOpaqueCount;
        d["materialMemoryInBytes"] = materials.materialMemoryInBytes;
        d["textureCount"] = materials.textureCount;
        d["textureCompressedCount"] = materials.textureCompressedCount;
        d["textureTexelCount"] = materials.textureTexelCount;
        d["textureMemoryInBytes"] = materials.textureMemoryInBytes;

        // Raytracing stats
        d["blasGroupCount"] = blasGroupCount;
        d["blasCount"] = blasCount;
        d["blasCompactedCount"] = blasCompactedCount;
        d["blasOpaqueCount"] = blasOpaqueCount;
        d["blasGeometryCount"] = blasGeometryCount;
        d["blasOpaqueGeometryCount"] = blasOpaqueGeometryCount;
        d["blasMemoryInBytes"] = blasMemoryInBytes;
        d["blasScratchMemoryInBytes"] = blasScratchMemoryInBytes;
        d["tlasCount"] = tlasCount;
        d["tlasMemoryInBytes"] = tlasMemoryInBytes;
        d["tlasScratchMemoryInBytes"] = tlasScratchMemoryInBytes;

        // Light stats
        d["activeLightCount"] = activeLightCount;
        d["totalLightCount"] = totalLightCount;
        d["pointLightCount"] = pointLightCount;
        d["directionalLightCount"] = directionalLightCount;
        d["rectLightCount"] = rectLightCount;
        d["sphereLightCount"] = sphereLightCount;
        d["distantLightCount"] = distantLightCount;
        d["lightsMemoryInBytes"] = lightsMemoryInBytes;
        d["envMapMemoryInBytes"] = envMapMemoryInBytes;
        d["emissiveMemoryInBytes"] = emissiveMemoryInBytes;

        // Volume stats
        d["gridVolumeCount"] = gridVolumeCount;
        d["gridVolumeMemoryInBytes"] = gridVolumeMemoryInBytes;

        // Grid stats
        d["gridCount"] = gridCount;
        d["gridVoxelCount"] = gridVoxelCount;
        d["gridMemoryInBytes"] = gridMemoryInBytes;

        return d;
    }

    FALCOR_SCRIPT_BINDING(Scene)
    {
        pybind11::class_<Scene, Scene::SharedPtr> scene(m, "Scene");

        scene.def_property_readonly(kStats.c_str(), [](const Scene* pScene) { return pScene->getSceneStats().toPython(); });
        scene.def_property_readonly(kBounds.c_str(), &Scene::getSceneBounds, pybind11::return_value_policy::copy);
        scene.def_property(kCamera.c_str(), &Scene::getCamera, &Scene::setCamera);
        scene.def_property(kEnvMap.c_str(), &Scene::getEnvMap, &Scene::setEnvMap);
        scene.def_property_readonly(kAnimations.c_str(), &Scene::getAnimations);
        scene.def_property_readonly(kCameras.c_str(), &Scene::getCameras);
        scene.def_property_readonly(kLights.c_str(), &Scene::getLights);
        scene.def_property_readonly(kMaterials.c_str(), &Scene::getMaterials);
        scene.def_property_readonly(kGridVolumes.c_str(), &Scene::getGridVolumes);
        scene.def_property_readonly("volumes", &Scene::getGridVolumes); // PYTHONDEPRECATED
        scene.def_property(kCameraSpeed.c_str(), &Scene::getCameraSpeed, &Scene::setCameraSpeed);
        scene.def_property(kAnimated.c_str(), &Scene::isAnimated, &Scene::setIsAnimated);
        scene.def_property(kLoopAnimations.c_str(), &Scene::isLooped, &Scene::setIsLooped);
        scene.def_property(kRenderSettings.c_str(), pybind11::overload_cast<void>(&Scene::getRenderSettings, pybind11::const_), &Scene::setRenderSettings);
        scene.def_property(kUpdateCallback.c_str(), &Scene::getUpdateCallback, &Scene::setUpdateCallback);

        scene.def(kSetEnvMap.c_str(), &Scene::loadEnvMap, "filename"_a);
        scene.def(kGetLight.c_str(), &Scene::getLight, "index"_a);
        scene.def(kGetLight.c_str(), &Scene::getLightByName, "name"_a);
        scene.def(kGetMaterial.c_str(), &Scene::getMaterial, "index"_a);
        scene.def(kGetMaterial.c_str(), &Scene::getMaterialByName, "name"_a);
        scene.def(kGetGridVolume.c_str(), &Scene::getGridVolume, "index"_a);
        scene.def(kGetGridVolume.c_str(), &Scene::getGridVolumeByName, "name"_a);
        scene.def("getVolume", &Scene::getGridVolume, "index"_a); // PYTHONDEPRECATED
        scene.def("getVolume", &Scene::getGridVolumeByName, "name"_a); // PYTHONDEPRECATED

        // Viewpoints
        scene.def(kAddViewpoint.c_str(), pybind11::overload_cast<>(&Scene::addViewpoint)); // add current camera as viewpoint
        scene.def(kAddViewpoint.c_str(), pybind11::overload_cast<const float3&, const float3&, const float3&, uint32_t>(&Scene::addViewpoint), "position"_a, "target"_a, "up"_a, "cameraIndex"_a = 0); // add specified viewpoint
        scene.def(kRemoveViewpoint.c_str(), &Scene::removeViewpoint); // remove the selected viewpoint
        scene.def(kSelectViewpoint.c_str(), &Scene::selectViewpoint, "index"_a); // select a viewpoint by index

        // RenderSettings
        ScriptBindings::SerializableStruct<Scene::RenderSettings> renderSettings(m, "SceneRenderSettings");
#define field(f_) field(#f_, &Scene::RenderSettings::f_)
        renderSettings.field(useEnvLight);
        renderSettings.field(useAnalyticLights);
        renderSettings.field(useEmissiveLights);
        renderSettings.field(useGridVolumes);
#undef field
    }
}
