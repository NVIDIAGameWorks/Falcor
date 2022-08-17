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
#pragma once

#include "Utils.h"
#include "USDHelpers.h"
#include "PreviewSurfaceConverter.h"
#include "Scene/SceneIDs.h"
#include "Scene/SceneBuilder.h"
#include "Scene/Animation/Animation.h"
#include "Scene/Curves/CurveTessellation.h"
#include "Utils/Math/Vector.h"
#include "Utils/Math/Matrix.h"
#include "Utils/Timing/TimeReport.h"
#include "Utils/Scripting/Dictionary.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
END_DISABLE_USD_WARNINGS

#include <glm/gtx/euler_angles.hpp>

#include <execution>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>


using namespace pxr;

namespace Falcor
{
    // Object Types

    /** Represents an instance of a UsdGeomGprim in the scene. In practice, currently limited to types UsdGeomMesh and UsdGeomBasisCurves.
    */
    struct GeomInstance
    {
        std::string name;                               ///< Instance name.
        UsdPrim prim;                                   ///< Reference to mesh prim.
        rmcv::mat4 xform;                                 ///< Instance world transform
        rmcv::mat4 bindTransform;                         ///< If mesh is skinned, its bind-time transform.
        NodeID parentID;                                ///< SceneBuilder node ID of parent
    };

    /** Represents an instance of a prototype in the scene.
    */
    struct PrototypeInstance
    {
        std::string name;                               ///< Instance name.
        UsdPrim protoPrim;                              ///< Reference to prototype prim.
        NodeID parentID{ NodeID::kInvalidID };          ///< SceneBuilder parent node id.
        rmcv::mat4 xform = rmcv::mat4(1.f);                 ///< Instance transformation.
        std::vector<Animation::Keyframe> keyframes;     ///< Keyframes for animated instance transformation, if any.
    };

    /** Mesh processing task parameters
    */
    struct MeshProcessingTask
    {
        uint32_t meshId;    ///< Mesh ID
        uint32_t sampleIdx; ///< Temporal sample index
    };

    /** Represents a mesh in the USD scene.
        During mesh processing, meshIDs is initialized with a list of scene builder mesh IDs.
        A Mesh can contain multiple procssed meshes/meshIDs when a UsdGeomMesh has one or more
        UsdGeomSubset children, since we break up such subsets into multiple Falcor meshes.
    */

    using ProcessedMeshList = std::vector<SceneBuilder::ProcessedMesh>;
    using MeshAttributeIndicesList = std::vector<SceneBuilder::MeshAttributeIndices>;

    struct Mesh
    {
        UsdPrim prim;                       ///< UsdGeomMesh prim, or UsdGeomBasisCurves prim to be tessellated into into a mesh.
        std::vector<double> timeSamples;    ///< Animation time samples.

        // Per GeomSubset
        ProcessedMeshList processedMeshes;          ///< Temporary list of pre-processed meshes
        std::vector<CachedMesh> cachedMeshes;       ///< Keyframe data for vertex-animated meshes per processed mesh
        std::vector<MeshID> meshIDs;                ///< List of scene builder mesh IDs.
        MeshAttributeIndicesList attributeIndices;  ///< For time-sampled meshes, list of attribute indices describing how mesh was processed
    };

    /** Represents a curvePrim in the USD scene.
        During curve processing, geometryID is initialized with scene builder curve or mesh ID.
    */
    struct Curve
    {
        static const uint32_t kInvalidID = std::numeric_limits<uint32_t>::max();

        UsdPrim curvePrim;                                          ///< Curve prim.
        CurveTessellationMode tessellationMode;                     ///< Curve tessellation mode.

        CurveOrMeshID geometryID{ CurveOrMeshID::kInvalidID };      ///< Geometry ID (curve or mesh, depending on tessellation mode).

        std::vector<double> timeSamples;                            ///< Time samples for animation.
        std::vector<SceneBuilder::ProcessedCurve> processedCurves;  ///< List of pre-processed curves per keyframe.

        SceneBuilder::ProcessedMesh processedMesh;                  ///< Pre-processed mesh of the first keyframe (valid only for PolyTube tessellation mode).
    };

    /** Represents a prototype in the scene.
        A prototype is a scene subgraph that is instantiated as a whole.
    */
    struct PrototypeGeom
    {
        struct AnimationKeyframes
        {
            std::vector<Animation::Keyframe> keyframes;
            NodeID targetNodeID{ NodeID::kInvalidID };
        };

        UsdPrim protoPrim;                                          ///< Prototype prim.
        std::vector<GeomInstance> geomInstances;                    ///< Geom instances making up the prototype.
        std::vector<PrototypeInstance> prototypeInstances;          ///< Prototype instances contained in the prototype.
        std::vector<SceneBuilder::Node> nodes;                      ///< Prototype subgraph nodes.
        std::vector<AnimationKeyframes> animations;                 ///< Animations targeting subgraph nodes, if any.
        std::vector<NodeID> nodeStack;                              ///< Current node stack.
        double timeCodesPerSecond;                                  ///< For use when creating animations.

        PrototypeGeom(const UsdPrim& prim, double timeCodesPerSecond)
            : protoPrim(prim)
            , timeCodesPerSecond(timeCodesPerSecond)
        {
            SceneBuilder::Node node;
            node.name = prim.GetPath().GetString();
            node.transform = rmcv::mat4(1.f);
            node.parent = NodeID::Invalid();
            nodes.push_back(node);
            nodeStack.push_back(NodeID{ 0 });
        }

        void addGeomInstance(const std::string& name, UsdPrim prim, const rmcv::mat4& xform, const rmcv::mat4& bindXform)
        {
            geomInstances.push_back(GeomInstance{name, prim, xform, bindXform, nodeStack.back()});
        }

        void addPrototypeInstance(const PrototypeInstance& inst)
        {
            prototypeInstances.push_back(inst);
        }

        void pushNode(const UsdGeomXformable& prim)
        {
            if (prim.TransformMightBeTimeVarying())
            {
                addAnimation(prim);
            }
            else
            {
                // The node stack should at least contain the root node.
                FALCOR_ASSERT(nodeStack.size() > 0);
                rmcv::mat4 localTransform;
                bool resets = getLocalTransform(prim, localTransform);

                SceneBuilder::Node node;
                node.name = prim.GetPath().GetString();
                node.transform = localTransform;
                node.parent = resets ? NodeID{ 0 } : nodeStack.back();
                NodeID nodeID{ nodes.size() };
                nodes.push_back(node);
                nodeStack.push_back(nodeID);
            }
        }

        void popNode()
        {
            nodeStack.pop_back();
        }

        void addAnimation(const UsdGeomXformable& xformable);
    };

    /** Represents data described by a SkelRoot, which can contain multiple Skeleton prims.
        A Skeleton object is created for every SkelRoot and represents all of its relevant children,
        which currently are Skeleton and SkelAnimation prims.
    */
    struct Skeleton
    {
        // Data described by a UsdSkeleton prim
        struct SubSkeleton
        {
            // Affected prims
            std::unordered_map<UsdObject, std::vector<uint32_t>, UsdObjHash> skinnedMeshes; ///< Map from a prim to its joint index mapping data

            // Skeleton Data
            std::vector<SceneBuilder::Node> bones;           ///< Local transforms of each bone
            std::vector<Animation::SharedPtr> animations;    ///< Animations per bone
            NodeID::IntType nodeOffset{ NodeID::kInvalidID };///< Where this skeleton's nodes start in the SceneBuilder
        };

        std::string name;                                   ///< SkelRoot prim name
        NodeID parentID{ NodeID::kInvalidID };              ///< Parent node ID
        NodeID nodeID{ NodeID::kInvalidID };                ///< Node ID of skeleton world transform in the scene graph
        std::vector<SubSkeleton> subskeletons;
    };

    // ImporterContext

    // Importer data and helper functions
    struct ImporterContext
    {
        ImporterContext(const std::filesystem::path& path, UsdStageRefPtr pStage, SceneBuilder& builder, const Dictionary& dict, TimeReport& timeReport, bool useInstanceProxies = false);

        // Get pointer to default material for the given prim, based on its type, creating it if it doesn't already exist.
        // Thread-safe.
        Material::SharedPtr getDefaultMaterial(const UsdPrim& prim);


        // Return a pointer to the material to use for the given UsdShadeMaterial. Thread-safe.
        Material::SharedPtr resolveMaterial(const UsdPrim& prim, const UsdShadeMaterial& material, const std::string& primName);

        // Return the UsdShadeMaterial bound to the given prim
        template <class T>
        UsdShadeMaterial getBoundMaterial(const T& prim)
        {
            return UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial(&bindingsCache, &collQueryCache);
        }

        // Meshes
        void addMesh(const UsdPrim& prim);
        const Mesh& getMesh(const UsdPrim& meshPrim) { return meshes[geomMap.at(meshPrim)]; }
        void addGeomInstance(const std::string& name, const UsdPrim& prim, const rmcv::mat4& xform, const rmcv::mat4& bindxform);

        // Prototypes
        void createPrototype(const UsdPrim& rootPrim);
        bool hasPrototype(const UsdPrim& protoPrim) const;
        const PrototypeGeom& getPrototypeGeom(const UsdPrim& protoPrim) { return prototypeGeoms[prototypeGeomMap.at(protoPrim)]; }
        void addPrototypeInstance(const PrototypeInstance& inst);

        // Curves
        void addCurve(const UsdPrim& curvePrim);
        const Curve& getCurve(const UsdPrim& curvePrim) { return curves[curveMap.at(curvePrim)]; }
        void addCurveInstance(const std::string& name, const UsdPrim& curvePrim, const rmcv::mat4& xform, NodeID parentID);
        void addCachedCurve(Curve& curve);

        // Add USD Objects
        void createEnvMap(const UsdPrim& lightPrim);
        void addLight(const UsdPrim& lightPrim, Light::SharedPtr pLight, NodeID parentId);
        void createDistantLight(const UsdPrim& lightPrim);
        void createRectLight(const UsdPrim& lightPrim);
        void createSphereLight(const UsdPrim& lightPrim);
        void createDiskLight(const UsdPrim& lightPrim);
        void createMeshedDiskLight(const UsdPrim& lightPrim);
        bool createCamera(const UsdPrim& cameraPrim);
        void createPointInstances(const UsdPrim& prim, PrototypeGeom* proto = nullptr);
        void createSkeleton(const UsdPrim& prim);

        // Animation

        // Create animation from time-sampled transforms on a prim, such as for rigid body animations.
        NodeID createAnimation(const UsdGeomXformable& xformable);

        // Initialize a vector of keyframes, one for each instance in a point instancer.
        // Returns false, and does not initialize keyframes, if the instance transforms are not animated.
        // Returns true otherwise.
        bool createPointInstanceKeyframes(const UsdGeomPointInstancer& instancer, std::vector<std::vector<Animation::Keyframe>>& keyframes);

        // Transforms

        /** Set the stage's root transform, which corrects for differences in scene unit and up direction.
            Additionally creates a node in the SceneBuilder that other objects can reference when importing.

            This should only be called once during importer initialization and only when the nodeStack is empty.
        */
        void setRootXform(const rmcv::mat4& xform);
        rmcv::mat4 getLocalToWorldXform(const UsdGeomXformable& prim, UsdTimeCode time = UsdTimeCode::EarliestTime());
        size_t getNodeStackDepth() const { return nodeStack.size(); }
        void pushNode(const UsdGeomXformable& prim);
        void popNode() { nodeStack.pop_back(); }
        NodeID getRootNodeID() const { return nodeStack[nodeStackStartDepth.back()]; }
        rmcv::mat4 getGeomBindTransform(const UsdPrim& usdPrim) const;

        /** Start a new node stack.

            Rather than maintain a stack-of-stacks, we emulate pushing a new stack by pushing and popping
            the 'invalid' (top-level) node on the single stack. This makes any traversal bugs epsilon more difficult
            to track down, but makes the code much cleaner. (Any mismatched node stack pushes and pops are
            due soley to bugs in this code; they can't be caused by malformed USD alone.)

            We also maintain a stack of the starting depths of each pushed stack to aid in error checking, and to
            allow determining the current root node id.
        */
        void pushNodeStack(NodeID nodeID = NodeID::Invalid());           ///< Start a new node stack using the given nodeID as the root

        // Pop the current node stack
        void popNodeStack();

        /** Perform additional operations required after scene has been traversed.

            Some object types may require data from other associated objects in the
            scene and cannot be added directly to the SceneBuilder during traversal.
        */
        void finalize();

        std::filesystem::path path;                                                                  ///< Path of the USD stage being imported.
        UsdStageRefPtr pStage;                                                                       ///< USD stage being imported.
        const Dictionary& dict;                                                                      ///< Input map from material path to material short name.
        std::map<std::string, std::string> localDict;                                                ///< Local input map from material path to
        TimeReport& timeReport;                                                                      ///< Timer object to use when importing.
        SceneBuilder& builder;                                                                       ///< Scene builder for this import session.
        std::vector<NodeID> nodeStack;                                                               ///< Stack of SceneBuilder node IDs
        std::vector<size_t> nodeStackStartDepth;                                                     ///< Stack depth at time of new node stack creation
        float metersPerUnit = .01f;                                                                  ///< Meters per unit scene distance.
        double timeCodesPerSecond = 24.0;                                                            ///< Time code unit scaling for time-sampled data. USD default is 24.
        rmcv::mat4 rootXform;                                                                          ///< Pseudoroot xform, correcting for world unit scaling and up vector orientation.
        NodeID rootXformNodeId{ NodeID::kInvalidID };                                                ///< Get the node ID containing the scene root transform in the builder's scene graph
        bool useInstanceProxies = false;                                                             ///< If true, traverse instances as if they were non-instances (debugging feature).

        std::unordered_map<std::string, Material::SharedPtr> materialMap;                            ///< Created material instances, indexed by material instance name.
        std::unordered_map<float3, Material::SharedPtr, Float3Hash> defaultMaterialMap;              ///< Default materials, indexed by base color.
        std::unordered_map<float3, Material::SharedPtr, Float3Hash> defaultCurveMaterialMap;         ///< Default curve materials, indexed by absorption coefficient.
        std::mutex materialMutex;                                                                    ///< Mutex to protect access to material maps.

        std::vector<Mesh> meshes;                                                                    ///< List of meshes.
        std::vector<PrototypeGeom> prototypeGeoms;                                                   ///< List of prototype geometries.
        std::vector<GeomInstance> geomInstances;                                                     ///< List of geom instances.
        std::vector<MeshProcessingTask> meshTasks;                                                   ///< List of mesh processing tasks (non time-sampled, and first time-samples)
        std::vector<MeshProcessingTask> meshKeyframeTasks;                                           ///< List of processing tasks for time-sampled mesh vertex data
        std::vector<PrototypeInstance> prototypeInstances;                                           ///< List of prototype instances.
        std::unordered_map<UsdObject, size_t, UsdObjHash> geomMap;                                   ///< Map from prim to mesh.
        std::unordered_map<UsdObject, size_t, UsdObjHash> prototypeGeomMap;                          ///< Map from prim to prototype mesh.
        std::vector<Skeleton> skeletons;                                                             ///< List of skeletons. One per SkelRoot prim.
        std::unordered_map<UsdObject, std::pair<size_t, size_t>, UsdObjHash> meshSkelMap;            ///< Mesh prim to the location of its skeleton in the "skeletons" data structure.

        std::vector<Curve> curves;                                                                   ///< List of curves.
        std::vector<GeomInstance> curveInstances;                                                    ///< List of curve instances.
        std::unordered_map<UsdObject, size_t, UsdObjHash> curveMap;                                  ///< Map from prim to curve.
        std::vector<CachedCurve> cachedCurves;                                                       ///< List of animated curve vertex caches.

        UsdShadeMaterialBindingAPI::CollectionQueryCache collQueryCache;                             ///< Material collection binding cache
        UsdShadeMaterialBindingAPI::BindingsCache bindingsCache;                                     ///< Material binding cache
        UsdSkelCache skelCache;

        std::unique_ptr<PreviewSurfaceConverter> mpPreviewSurfaceConverter;                          ///< UsdPreviewSurface to Falcor::Material converter instance
};
}
