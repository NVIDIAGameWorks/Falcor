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
#pragma once
#include "LightBVH.h"
#include "Utils/Math/BBox.h"
#include "Utils/Math/Vector.h"
#include "Utils/UI/Gui.h"
#include <limits>
#include <vector>

namespace Falcor
{
    /** Utility class for building 2-way light BVH on the CPU.

        The building process can be customized via the |Options|,
        which are also available in the GUI via the |renderUI()| function.

        TODO: Rename all things triangle* to light* as the BVH class can be used for other types.
    */
    class dlldecl LightBVHBuilder
    {
    public:
        using SharedPtr = std::shared_ptr<LightBVHBuilder>;
        using SharedConstPtr = std::shared_ptr<const LightBVHBuilder>;

        enum class SplitHeuristic : uint32_t
        {
            Equal = 0u,         ///< Split the input into two equal partitions.
            BinnedSAH = 1u,     ///< Split the input according to SAH; the input is binned for speeding up the SAH computation.
            BinnedSAOH = 2u,    ///< Split the input according to SAOH (Estévez Conty et al, 2018); the input is binned for speeding up the SAOH computation.
        };

        /** Light BVH builder configuration options.
            Note if you change options, please update SCRIPT_BINDING in LightBVHBuilder.cpp
        */
        struct Options
        {
            SplitHeuristic splitHeuristicSelection = SplitHeuristic::BinnedSAOH; ///< Which splitting heuristic to use when building.
            uint32_t       maxTriangleCountPerLeaf = 10;                         ///< How many triangles to store at most per leaf node.
            uint32_t       binCount = 16;                                        ///< How many bins to use when building the BVH.
            float          volumeEpsilon = 1e-3f;                                ///< If a node has an AABB which is 0 along one (or more) of its dimensions, use this epsilon instead for that dimension. Only used when 'useVolumeOverSA' is enabled.
            bool           splitAlongLargest = false;                            ///< Rather than computing a split along each of the 3 dimensions and selecting the best one, only compute the split along the largest dimension.
            bool           useVolumeOverSA = false;                              ///< Use the volume rather than the surface area of the AABB, when computing a split cost.
            bool           useLeafCreationCost = true;                           ///< Set to true to avoid splitting when the cost is higher than the cost of creating a leaf node. Only used when 'createLeavesASAP' is disabled.
            bool           createLeavesASAP = true;                              ///< Rather than creating a leaf only once splitting stops, create it as soon as we can.
            bool           allowRefitting = true;                                ///< Rather than always rebuilding the BVH from scratch, keep the hierarchy but update the bounds and lighting cones.
            bool           usePreintegration = true;                             ///< Use pre-integration for culling out emissive triangles and use their flux when computing the splits. Only valid when using the BinnedSAOH split heuristic.
            bool           useLightingCones = true;                              ///< Use lighting cones when computing the splits. Only valid when using the BinnedSAOH split heuristic.
        };

        /** Creates a new object.
            \param[in] options The options to use for building the BVH.
        */
        static SharedPtr create(const Options& options);

        /** Build the BVH.
            \param[in,out] bvh The light BVH to build.
        */
        void build(LightBVH& bvh);

        virtual bool renderUI(Gui::Widgets& widget);

        const Options& getOptions() const { return mOptions; }

    protected:
        struct Range
        {
            uint32_t begin;
            uint32_t end;

            Range(uint32_t _begin, uint32_t _end) : begin(_begin), end(_end) { assert(begin <= end); }
            constexpr uint32_t middle() const noexcept { return (begin + end) / 2; }
            constexpr uint32_t length() const noexcept { return end - begin; }
        };

        struct SplitResult
        {
            uint32_t axis = std::numeric_limits<uint32_t>::max();
            uint32_t triangleIndex = std::numeric_limits<uint32_t>::max();

            bool isValid() const
            {
                return axis != std::numeric_limits<uint32_t>::max() &&
                    triangleIndex != std::numeric_limits<uint32_t>::max();
            }
        };

        struct TriangleSortData
        {
            BBox bounds;                                    ///< World-space bounding box for the light source(s).
            float3 center = {};                             ///< Center point.
            float3 coneDirection = {};                      ///< Light emission normal direction.
            float cosConeAngle = 1.f;                       ///< Cosine normal bounding cone (half) angle.
            float flux = 0.f;                               ///< Precomputed triangle flux (note, this takes doublesidedness into account).
            uint32_t triangleIndex = MeshLightData::kInvalidIndex; ///< Index into global triangle list.
        };

        struct BuildingData
        {
            std::vector<PackedNode>& nodes;                 ///< BVH nodes generated by the builder.
            std::vector<TriangleSortData> trianglesData;    ///< Compact list of triangles to include in build.
            std::vector<uint32_t> triangleIndices;          ///< Triangle indices sorted by leaf node. Each leaf node refers to a contiguous array of triangle indices.
            std::vector<uint64_t> triangleBitmasks;         ///< Array containing the per triangle bit pattern retracing the tree traversal to reach the triangle: 0=left child, 1=right child; this array gets filled in during the build process. Indexed by global triangle index.
            float currentNodeFlux = 0.f;                    ///< Used by computeSAOHSplit() as the leaf creation cost.

            BuildingData(std::vector<PackedNode>& bvhNodes) : nodes(bvhNodes) {}
        };

        /** Compute the split according to a specified heuristic.
            \param[in] data Prepared light data.
            \param[in] triangleRange Range of triangles to process.
            \param[in] nodeBounds Bounds for the node to be splitted.
            \param[in] parameters Various parameters defining how the building should occur.
        */
        using SplitHeuristicFunction = std::function<SplitResult(const BuildingData& data, const Range& triangleRange, const BBox& nodeBounds, const Options& parameters)>;

        LightBVHBuilder(const Options& options);

        /** Renders the UI with builder options.
        */
        bool renderOptions(Gui::Widgets& widget, Options& options) const;

        /** Recursive BVH build.
            \param[in] splitHeuristic The splitting heuristic to be used.
            \param[in] bitmask Bit pattern retracing the tree traversal to reach the node to be built: 0=left child, 1=right child.
            \param[in] depth Depth of the node to be built
            \param[in] triangleRange Range of triangles to process.
            \param[in,out] data Prepared light data.
            \return Index of the allocated node.
        */
        uint32_t buildInternal(const Options& options, const SplitHeuristicFunction& splitHeuristic, uint64_t bitmask, uint32_t depth, const Range& triangleRange, BuildingData& data);

        /** Recursive computation of lighting cones for all internal nodes.
            \param[in] nodeIndex Index of the current node.
            \param[in,out] data Updated node data.
            \param[out] cosConeAngle Cosine of the cone angle of the lighting cone for the current node, or kInvalidCosConeAngle if the cone is invalid.
            \return direction of the lighting cone for the current node.
        */
        float3 computeLightingConesInternal(const uint32_t nodeIndex, BuildingData& data, float& cosConeAngle);

        /** Compute lighting cone for a range of triangles.
            \param[in] triangleRange Range of triangles to process.
            \param[in] data Prepared light data.
            \param[out] cosTheta Cosine of the cone angle.
            \return Direction of the lighting cone.
        */
        static float3 computeLightingCone(const Range& triangleRange, const BuildingData& data, float& cosTheta);

        // See the documentation of SplitHeuristicFunction.
        static SplitResult computeSplitWithEqual(const BuildingData& /*data*/, const Range& triangleRange, const BBox& nodeBounds, const Options& /*parameters*/);
        static SplitResult computeSplitWithBinnedSAH(const BuildingData& data, const Range& triangleRange, const BBox& nodeBounds, const Options& parameters);
        static SplitResult computeSplitWithBinnedSAOH(const BuildingData& data, const Range& triangleRange, const BBox& nodeBounds, const Options& parameters);

        static SplitHeuristicFunction getSplitFunction(SplitHeuristic heuristic);

        // Configuration
        Options mOptions;
    };
}
