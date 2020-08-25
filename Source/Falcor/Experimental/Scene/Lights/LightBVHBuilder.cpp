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
#include "stdafx.h"
#include "LightBVHBuilder.h"
#include <algorithm>

namespace
{
    using namespace Falcor;

    // Define the maximum supported BVH tree depth.
    // The limitation comes from the need to store the traversal path to each node in a bit mask.
    const uint32_t kMaxBVHDepth = 64;

    // Define the maximum supported leaf triangle count and offsets.
    const uint32_t kMaxLeafTriangleCount = 1 << PackedNode::kTriangleCountBits;
    const uint32_t kMaxLeafTriangleOffset = 1 << PackedNode::kTriangleOffsetBits;

    inline float safeACos(float v)
    {
        return std::acos(glm::clamp(v, -1.0f, 1.0f));
    }

    /** Returns sin(a) based on cos(a) for a in [0,pi].
    */
    inline float sinFromCos(float cosAngle)
    {
        return std::sqrt(std::max(0.f, 1.f - cosAngle * cosAngle));
    }

    /** Given a bounding cone specified by direction and cosine spread angle,
        compute the minimum cone angle that includes a second bounding cone.
        If either cone is invalid or the result is larger than pi, the resulting
        cone is marked as invalid.
        TODO: Move to utility header and add unit test.
        \return The cosine of the spread angle for the new cone.
    */
    float computeCosConeAngle(const float3& coneDir, const float cosTheta, const float3& otherConeDir, const float cosOtherTheta)
    {
        float cosResult = kInvalidCosConeAngle;
        if (cosTheta != kInvalidCosConeAngle && cosOtherTheta != kInvalidCosConeAngle)
        {
            const float cosDiffTheta = glm::dot(coneDir, otherConeDir);
            const float sinDiffTheta = sinFromCos(cosDiffTheta);
            const float sinOtherTheta = sinFromCos(cosOtherTheta);

            // Rotate (cosDiffTheta, sinDiffTheta) counterclockwise by the other cone's spread angle.
            float cosTotalTheta = cosOtherTheta * cosDiffTheta - sinOtherTheta * sinDiffTheta;
            float sinTotalTheta = sinOtherTheta * cosDiffTheta + cosOtherTheta * sinDiffTheta;

            // If the total angle is less than pi, store the new cone angle.
            // Otherwise, the bounding cone will be deactivated because it would represent the whole sphere.
            if (sinTotalTheta > 0.f)
            {
                cosResult = std::min(cosTheta, cosTotalTheta);
            }
        }
        return cosResult;
    }

    /** Given two cones specified by direction vectors and the cosine of
        their spread angles, returns a cone that bounds both of them. This
        is what was used previously; the cones it returns aren't as tight as
        those given by coneUnion().
    */
    float3 coneUnionOld(float3 aDir, float aCosTheta, float3 bDir, float bCosTheta, float& cosResult)
    {
        float3 dir = aDir + bDir;
        if (aCosTheta == kInvalidCosConeAngle || bCosTheta == kInvalidCosConeAngle || dir == float3(0.0f))
        {
            cosResult = kInvalidCosConeAngle;
            return float3(0.0f);
        }

        dir = glm::normalize(dir);

        const float aDiff = safeACos(glm::dot(dir, aDir));
        const float bDiff = safeACos(glm::dot(dir, bDir));
        cosResult = std::cos(std::max(aDiff + std::acos(aCosTheta), bDiff + std::acos(bCosTheta)));
        return dir;
    }

    /** Given two cones specified by direction vectors and the cosine of
        their spread angles, returns a cone that bounds both of
        them. Algorithm 1 in the 2018 Sony EGSR light sampling paper.
    */
    float3 coneUnion(float3 aDir, float aCosTheta, float3 bDir, float bCosTheta, float& cosResult)
    {
        if (aCosTheta == kInvalidCosConeAngle || bCosTheta == kInvalidCosConeAngle)
        {
            cosResult = kInvalidCosConeAngle;
            return float3(0.0f);
        }

        // Swap if necessary so that aTheta > bTheta. Note that the test is
        // reversed since we're testing the cosine of the angles.
        if (bCosTheta < aCosTheta)
        {
            std::swap(aDir, bDir);
            std::swap(aCosTheta, bCosTheta);
        }

        // TODO: this could be optimized to use fewer trig functions.
        const float theta = safeACos(glm::dot(aDir, bDir));
        const float aTheta = safeACos(aCosTheta), bTheta = safeACos(bCosTheta);
        if (std::min(theta + bTheta, glm::pi<float>()) <= aTheta)
        {
            // a encloses b and we're done.
            cosResult = aCosTheta;
            return aDir;
        }

        // Merge the two cones. First compute the spread angle of the cone
        // that will fit all of them.
        float oTheta = (theta + aTheta + bTheta) / 2;
        if (oTheta > glm::pi<float>())
        {
            cosResult = kInvalidCosConeAngle;
            return float3(0.0f);
        }

        // Rotate a's axis toward b just enough so that that oTheta covers
        // both cones.
        const float rTheta = oTheta - aTheta;
        const float3 rDir = glm::cross(aDir, bDir);
        float3 dir;
        if (glm::dot(rDir, rDir) < 1e-8)
        {
            // The two vectors are effectively pointing in opposite directions.

            // Find some vector that's orthogonal to one of them (via
            // "Building and Orthonormal Basis, Revisited" in jcgt.)
            const float sign = aDir.z > 0 ? 1.f : -1.f;
            const float a = -1.f / (sign + aDir.z);
            const float b = aDir.x * aDir.y * a;
            dir = float3(1.f + sign * aDir.x * aDir.x * a, sign * b, -sign * aDir.x);
            // The spread angle needs to be pi/2 to encompass aDir and
            // bDir, then aTheta / 2 more on top of that. (Recall that
            // aTheta > bTheta, so we don't need to worry about bTheta).
            // Note that we could rotate dir around the vector cross(dir,
            // aTheta) and then be able to use the tighter spread angle
            // oTheta computed before, but it probably doesn't matter much
            // in this (rare) case.
            oTheta = std::min(glm::pi<float>(), glm::half_pi<float>() + aTheta);
            cosResult = std::cos(oTheta);
        }
        else
        {
            // Rotate aDir by an angle of rTheta around the axis rDir.
            const glm::mat4 rotationMatrix = glm::rotate(glm::mat4(), rTheta, rDir);
            dir = rotationMatrix * float4(aDir, 0);
            cosResult = std::cos(oTheta);
        }

        // TODO: write a unit test.
        // Testing code: make sure both a and b are inside the result.
        auto checkInside = [&](float3 d, float theta) {
                               // Make sure that sum of the angle between
                               // the two cone vectors and the spread angle
                               // of the given cone is still within the
                               // extent of the result cone.
                               float cosDelta = glm::dot(d, dir);
                               float delta = safeACos(cosDelta);
                               bool dInCone = (delta + theta <= oTheta * 1.01f ||
                                               delta + theta <= oTheta + 1e-3f);
                               if (!dInCone)
                               {
                                   logError("coneUnion error! angle diff " + std::to_string(delta + theta) +
                                       " > spread " + std::to_string(oTheta));
                                   assert(dInCone);
                               }
                           };
        checkInside(aDir, aTheta);
        checkInside(bDir, bTheta);

        return dir;
    }

    const Gui::DropdownList kSplitHeuristicList =
    {
        { (uint32_t)LightBVHBuilder::SplitHeuristic::Equal, "Equal" },
        { (uint32_t)LightBVHBuilder::SplitHeuristic::BinnedSAH, "Binned SAH" },
        { (uint32_t)LightBVHBuilder::SplitHeuristic::BinnedSAOH, "Binned SAOH" }
    };
}

namespace Falcor
{
    static_assert(sizeof(PackedNode) % 16 == 0, "PackedNode size should be a multiple of 16");

    LightBVHBuilder::SharedPtr LightBVHBuilder::create(const Options& options)
    {
        return SharedPtr(new LightBVHBuilder(options));
    }

    void LightBVHBuilder::build(LightBVH& bvh)
    {
        PROFILE("LightBVHBuilder::build()");

        bvh.clear();
        assert(!bvh.isValid() && bvh.mNodes.empty());

        // Get global list of emissive triangles.
        assert(bvh.mpLightCollection);
        const auto& triangles = bvh.mpLightCollection->getMeshLightTriangles();
        if (triangles.empty()) return;

        // Create list of triangles that should be included in BVH.
        // For each triangle, precompute data we need for the build.
        BuildingData data(bvh.mNodes);
        data.trianglesData.reserve(triangles.size());

        for (size_t i = 0; i < triangles.size(); i++)
        {
            if (!mOptions.usePreintegration || triangles[i].flux > 0.f)
            {
                LightBVHBuilder::TriangleSortData tri;
                for (uint32_t j = 0; j < 3; j++)
                {
                    tri.bounds |= triangles[i].vtx[j].pos;
                }
                tri.center = triangles[i].getCenter();
                tri.coneDirection = triangles[i].normal;
                tri.cosConeAngle = 1.f; // Single flat emitter => normal bounding cone angle is zero.
                tri.flux = triangles[i].flux;
                tri.triangleIndex = static_cast<uint32_t>(i);

                data.trianglesData.push_back(tri);
            }
        }

        // If there are no non-culled triangles, we're done.
        if (data.trianglesData.empty()) return;

        // Validate options.
        if (mOptions.maxTriangleCountPerLeaf > kMaxLeafTriangleCount)
        {
            throw std::exception(("Max triangle count per leaf exceeds the maximum supported (" + std::to_string(kMaxLeafTriangleCount) + ")").c_str());
        }
        if (data.trianglesData.size() > kMaxLeafTriangleOffset + kMaxLeafTriangleCount)
        {
            throw std::exception(("Emissive triangle count exceeds the maximum supported (" + std::to_string(kMaxLeafTriangleOffset + kMaxLeafTriangleCount) + ")").c_str());
        }

        // Allocate temporary memory for the BVH build.
        // To be grossly conservative, assume each triangle requires two nodes.
        // This is only system RAM and shouldn't be that much, so it's not worth being more careful about it.
        // TODO: Better estimate of how many nodes we will need.
        data.nodes.clear();
        data.nodes.reserve(2 * data.trianglesData.size());
        data.triangleIndices.reserve(data.trianglesData.size());

        const uint64_t invalidBitmask = std::numeric_limits<uint64_t>::max();
        data.triangleBitmasks.resize(triangles.size(), invalidBitmask); // This is sized based on input triangle count, as it's indexed by global triangle index.

        // Build the tree.
        SplitHeuristicFunction splitFunc = getSplitFunction(mOptions.splitHeuristicSelection);
        buildInternal(mOptions, splitFunc, 0ull, 0, Range(0, static_cast<uint32_t>(data.trianglesData.size())), data);
        assert(!data.nodes.empty());

        size_t numValid = 0;
        for (auto mask : data.triangleBitmasks)
            if (mask != invalidBitmask) numValid++;
        assert(numValid == data.trianglesData.size());

        // Compute per-node light bounding cones.
        float cosConeAngle;
        computeLightingConesInternal(0, data, cosConeAngle);

        // The BVH is ready, mark it as valid and upload the data.
        bvh.mIsValid = true;
        bvh.mMaxTriangleCountPerLeaf = mOptions.maxTriangleCountPerLeaf;
        bvh.uploadCPUBuffers(data.triangleIndices, data.triangleBitmasks);

        // Computate metadata.
        bvh.finalize();
    }

    bool LightBVHBuilder::renderUI(Gui::Widgets& widget)
    {
        // Render the build options.
        return renderOptions(widget, mOptions);
    }

    bool LightBVHBuilder::renderOptions(Gui::Widgets& widget, Options& options) const
    {
        bool optionsChanged = false;

        optionsChanged |= widget.checkbox("Allow refitting", options.allowRefitting);
        optionsChanged |= widget.var("Max triangle count per leaf", options.maxTriangleCountPerLeaf, 1u, kMaxLeafTriangleCount);
        optionsChanged |= widget.dropdown("Split heuristic", kSplitHeuristicList, (uint32_t&)options.splitHeuristicSelection);

        Gui::Group splitGroup(widget, "Split Options", true);
        if (splitGroup.open())
        {
            optionsChanged |= splitGroup.var("Bin count", options.binCount);
            optionsChanged |= splitGroup.checkbox("Create leaves ASAP", options.createLeavesASAP);
            if (!options.createLeavesASAP)
            {
                optionsChanged |= splitGroup.var("Bin count", options.binCount);
            }
            optionsChanged |= splitGroup.checkbox("Split along largest dimension", options.splitAlongLargest);
            optionsChanged |= splitGroup.checkbox("Use volume instead of surface area", options.useVolumeOverSA);
            if (options.useVolumeOverSA)
            {
                // TODO: Not sure we need volumeEpsilon as a configurable parameter. A small fixed epsilon should suffice.
                optionsChanged |= splitGroup.var("Dimension epsilon for volume", options.volumeEpsilon, 0.0f, 1.0f);
            }

            if (options.splitHeuristicSelection == SplitHeuristic::BinnedSAOH)
            {
                optionsChanged |= splitGroup.checkbox("Use leaf creation cost", options.useLeafCreationCost);
                optionsChanged |= splitGroup.checkbox("Use pre-integration", options.usePreintegration);
                optionsChanged |= splitGroup.checkbox("Use lighting cones", options.useLightingCones);
            }

            splitGroup.release();
        }

        return optionsChanged;
    }

    LightBVHBuilder::LightBVHBuilder(const Options& options) : mOptions(options)
    {
    }

    uint32_t LightBVHBuilder::buildInternal(const Options& options, const SplitHeuristicFunction& splitHeuristic, uint64_t bitmask, uint32_t depth, const Range& triangleRange, BuildingData& data)
    {
        assert(triangleRange.begin < triangleRange.end);

        // Compute the AABB and total flux of the node.
        float nodeFlux = 0.f;
        BBox nodeBounds;
        for (uint32_t dataIndex = triangleRange.begin; dataIndex < triangleRange.end; ++dataIndex)
        {
            nodeBounds |= data.trianglesData[dataIndex].bounds;
            nodeFlux += data.trianglesData[dataIndex].flux;
        }
        assert(nodeBounds.valid());

        data.currentNodeFlux = nodeFlux;

        bool trySplitting = triangleRange.length() > (options.createLeavesASAP ? options.maxTriangleCountPerLeaf : 1);
        const SplitResult splitResult = trySplitting ? splitHeuristic(data, triangleRange, nodeBounds, options) : SplitResult();

        // If we should split, then create an internal node and split.
        if (splitResult.isValid())
        {
            assert(triangleRange.begin < splitResult.triangleIndex && splitResult.triangleIndex < triangleRange.end);

            // Sort the centroids and update the lists accordingly.
            auto comp = [dim = splitResult.axis](const TriangleSortData& d1, const TriangleSortData& d2) { return d1.bounds.centroid()[dim] < d2.bounds.centroid()[dim]; };
            std::nth_element(std::begin(data.trianglesData) + triangleRange.begin, std::begin(data.trianglesData) + splitResult.triangleIndex, std::begin(data.trianglesData) + triangleRange.end, comp);

            // Allocate internal node.
            assert(data.nodes.size() < std::numeric_limits<uint32_t>::max());
            const uint32_t nodeIndex = (uint32_t)data.nodes.size();
            data.nodes.push_back({});

            InternalNode node = {};
            node.attribs.setAABB(nodeBounds.minPoint, nodeBounds.maxPoint);
            node.attribs.flux = nodeFlux;
            // The lighting normal bounding cone will be computed later when all leaf nodes have been created.

            if (depth >= kMaxBVHDepth)
            {
                // This is an unrecoverable error since we use bit masks to represent the traversal path from
                // the root node to each leaf node in the tree, which is necessary for pdf computation with MIS.
                throw std::exception(("BVH depth of " + std::to_string(depth + 1) + " reached; maximum of " + std::to_string(kMaxBVHDepth) + " allowed.").c_str());
            }

            uint32_t leftIndex = buildInternal(options, splitHeuristic, bitmask | (0ull << depth), depth + 1, Range(triangleRange.begin, splitResult.triangleIndex), data);
            uint32_t rightIndex = buildInternal(options, splitHeuristic, bitmask | (1ull << depth), depth + 1, Range(splitResult.triangleIndex, triangleRange.end), data);

            assert(leftIndex == nodeIndex + 1); // The left node should always be placed immediately after the current node.
            node.rightChildIdx = rightIndex;

            data.nodes[nodeIndex].setInternalNode(node);
            return nodeIndex;
        }
        else // No split => create leaf node
        {
            assert(triangleRange.length() <= options.maxTriangleCountPerLeaf);

            // Allocate leaf node.
            assert(data.nodes.size() < std::numeric_limits<uint32_t>::max());
            const uint32_t nodeIndex = (uint32_t)data.nodes.size();
            data.nodes.push_back({});

            LeafNode node = {};
            node.attribs.setAABB(nodeBounds.minPoint, nodeBounds.maxPoint);
            node.attribs.flux = nodeFlux;
            float cosTheta;
            node.attribs.coneDirection = computeLightingCone(triangleRange, data, cosTheta);
            node.attribs.cosConeAngle = cosTheta;

            node.triangleCount = triangleRange.length();
            node.triangleOffset = (uint32_t)data.triangleIndices.size();
            assert(node.triangleCount < kMaxLeafTriangleCount);
            assert(node.triangleOffset < kMaxLeafTriangleOffset);

            for (uint32_t triangleIdx = triangleRange.begin, index = 0; triangleIdx < triangleRange.end; ++triangleIdx, ++index)
            {
                uint32_t globalTriangleIndex = data.trianglesData[triangleIdx].triangleIndex;
                data.triangleIndices.push_back(globalTriangleIndex);
                data.triangleBitmasks[globalTriangleIndex] = bitmask;
            }
            assert(data.triangleIndices.size() == node.triangleOffset + node.triangleCount);

            data.nodes[nodeIndex].setLeafNode(node);
            return nodeIndex;
        }
    }

    float3 LightBVHBuilder::computeLightingConesInternal(const uint32_t nodeIndex, BuildingData& data, float& cosConeAngle)
    {
        if (!data.nodes[nodeIndex].isLeaf())
        {
            auto node = data.nodes[nodeIndex].getInternalNode();

            uint32_t leftIndex = nodeIndex + 1;
            uint32_t rightIndex = node.rightChildIdx;

            float leftNodeCosConeAngle = kInvalidCosConeAngle;
            float3 leftNodeConeDirection = computeLightingConesInternal(leftIndex, data, leftNodeCosConeAngle);
            float rightNodeCosConeAngle = kInvalidCosConeAngle;
            float3 rightNodeConeDirection = computeLightingConesInternal(rightIndex, data, rightNodeCosConeAngle);

            // TODO: Asserts in coneUnion
            //float3 coneDirection = coneUnion(leftNodeConeDirection, leftNodeCosConeAngle,
            float3 coneDirection = coneUnionOld(leftNodeConeDirection, leftNodeCosConeAngle,
                rightNodeConeDirection, rightNodeCosConeAngle, cosConeAngle);

            // Update bounding cone.
            node.attribs.cosConeAngle = cosConeAngle;
            node.attribs.coneDirection = coneDirection;
            data.nodes[nodeIndex].setNodeAttributes(node.attribs);

            return coneDirection;
        }
        else
        {
            // Load bounding cone.
            auto attribs = data.nodes[nodeIndex].getNodeAttributes();
            cosConeAngle = attribs.cosConeAngle;
            return attribs.coneDirection;
        }
    }

    float3 LightBVHBuilder::computeLightingCone(const Range& triangleRange, const BuildingData& data, float& cosTheta)
    {
        float3 coneDirection = float3(0.0f);
        cosTheta = kInvalidCosConeAngle;

        // We use the average normal as cone direction and grow the cone to include all light normals.
        // TODO: Switch to a more sophisticated algorithm to compute tighter bounding cones.
        float3 coneDirectionSum = float3(0.0f);
        for (uint32_t triangleIdx = triangleRange.begin; triangleIdx < triangleRange.end; ++triangleIdx)
        {
            coneDirectionSum += data.trianglesData[triangleIdx].coneDirection;
        }
        if (glm::length(coneDirectionSum) >= FLT_MIN)
        {
            coneDirection = glm::normalize(coneDirectionSum);
            cosTheta = 1.f;
            for (uint32_t triangleIdx = triangleRange.begin; triangleIdx < triangleRange.end; ++triangleIdx)
            {
                const TriangleSortData& td = data.trianglesData[triangleIdx];
                cosTheta = computeCosConeAngle(coneDirection, cosTheta, td.coneDirection, td.cosConeAngle);
            }
        }
        return coneDirection;
    }

    LightBVHBuilder::SplitResult LightBVHBuilder::computeSplitWithEqual(const BuildingData& /*data*/, const Range& triangleRange, const BBox& nodeBounds, const Options& /*parameters*/)
    {
        // Find the largest dimension.
        float3 dimensions = nodeBounds.dimensions();
        uint32_t dimension = dimensions[2] >= dimensions[0] && dimensions[2] >= dimensions[1] ?
            2 : (dimensions[1] >= dimensions[0] ? 1 : 0);

        // Split the triangle range half-way.
        SplitResult result;
        result.axis = dimension;
        result.triangleIndex = triangleRange.middle();
        assert(triangleRange.begin < result.triangleIndex && result.triangleIndex < triangleRange.end);
        return result;
    }

    /** Evaluates the SAH cost metric for a node.
        If the node is empty (invalid bounds), the cost evaluates to zero.
        See Eqn 15 in Moreau and Clarberg, "Importance Sampling of Many Lights on the GPU", Ray Tracing Gems, Ch. 18, 2019.
    */
    static float evalSAH(const BBox& bounds, const uint32_t triangleCount, const LightBVHBuilder::Options& parameters)
    {
        float aabbCost = bounds.valid() ? (parameters.useVolumeOverSA ? bounds.volume(parameters.volumeEpsilon) : bounds.surfaceArea()) : 0.f;
        float cost = aabbCost * (float)triangleCount;
        assert(cost >= 0.f && !std::isnan(cost) && !std::isinf(cost));
        return cost;
    }

    LightBVHBuilder::SplitResult LightBVHBuilder::computeSplitWithBinnedSAH(const BuildingData& data, const Range& triangleRange, const BBox& nodeBounds, const Options& parameters)
    {
        std::pair<float, SplitResult> overallBestSplit = std::make_pair(std::numeric_limits<float>::infinity(), SplitResult());
        assert(!overallBestSplit.second.isValid());

        struct Bin
        {
            BBox bounds = BBox();
            uint32_t triangleCount = 0;

            Bin() = default;
            Bin(const TriangleSortData& tri) : bounds(tri.bounds), triangleCount(1) {}
            Bin& operator|= (const Bin& rhs)
            {
                bounds |= rhs.bounds;
                triangleCount += rhs.triangleCount;
                return *this;
            }
        };

        assert(parameters.binCount > 1);
        std::vector<Bin> bins(parameters.binCount);
        std::vector<float> costs(parameters.binCount - 1);

        /** Helper function that computes the best split along the given dimension using the SAH metric.
            The triangles are binned to n bins, storing only the aggregate parameters (triangle count and bounds).
            Then the cost metric is evaluated for each of the n-1 potential splits.
        */
        const auto binAlongDimension = [&bins, &costs, &triangleRange, &data, &parameters, &overallBestSplit, &nodeBounds](uint32_t dimension)
        {
            // Helper to compute the bin id for a given triangle.
            auto getBinId = [&](const TriangleSortData& td)
            {
                float bmin = nodeBounds.minPoint[dimension], bmax = nodeBounds.maxPoint[dimension];
                assert(bmin < bmax);
                float scale = (float)parameters.binCount / (bmax - bmin);
                float p = td.bounds.centroid()[dimension];
                assert(bmin <= p && p <= bmax);
                return std::min((uint32_t)((p - bmin) * scale), parameters.binCount - 1);
            };

            // Reset the bins.
            for (Bin& bin : bins) bin = Bin();

            // Fill the bins with all triangles.
            for (uint32_t i = triangleRange.begin; i < triangleRange.end; ++i)
            {
                const auto& td = data.trianglesData[i];
                bins[getBinId(td)] |= td;
            }

            // First, compute A_j(L) * N_j(L) by sweeping over the bins from left to right.
            // Note that the costs vector has n-1 elements when there are n bins; the i:th elements represents the split between bin i and i+1.
            Bin total = Bin();
            for (std::size_t i = 0; i < costs.size(); ++i)
            {
                total |= bins[i];
                costs[i] = evalSAH(total.bounds, total.triangleCount, parameters);
            }

            // Then, compute A_j(R) * N_j(R) by sweeping over the bins from right to left.
            total = Bin();
            for (std::size_t i = costs.size(); i > 0; --i)
            {
                total |= bins[i];
                costs[i - 1] += evalSAH(total.bounds, total.triangleCount, parameters);
            }

            // Compute the cheapest split along the current dimension.
            std::pair<float, SplitResult> axisBestSplit = std::make_pair(std::numeric_limits<float>::infinity(), SplitResult{ dimension, 0 });
            for (uint32_t i = 0, triIdx = triangleRange.begin; i < costs.size(); ++i)
            {
                triIdx += bins[i].triangleCount;
                if (costs[i] < axisBestSplit.first)
                {
                    axisBestSplit = std::make_pair(costs[i], SplitResult{ dimension, triIdx });
                }
            }
            assert(triangleRange.begin <= axisBestSplit.second.triangleIndex && axisBestSplit.second.triangleIndex <= triangleRange.end);

            // Early out if all lights fall on either side of the split.
            if (axisBestSplit.second.triangleIndex == triangleRange.begin ||
                axisBestSplit.second.triangleIndex == triangleRange.end) return;

            if (axisBestSplit.first < overallBestSplit.first)
            {
                overallBestSplit = axisBestSplit;
                assert(triangleRange.begin < overallBestSplit.second.triangleIndex && overallBestSplit.second.triangleIndex < triangleRange.end);
            }
        };

        if (parameters.splitAlongLargest)
        {
            // Find the largest dimension.
            float3 dimensions = nodeBounds.dimensions();
            uint32_t largestDimension = dimensions[2] >= dimensions[0] && dimensions[2] >= dimensions[1] ?
                2 : (dimensions[1] >= dimensions[0] && dimensions[1] >= dimensions[2] ? 1 : 0);

            binAlongDimension(largestDimension);
        }
        else
        {
            for (uint32_t dimension = 0; dimension < 3; ++dimension)
            {
                binAlongDimension(dimension);
            }
        }

        // If we couldn't find a valid split, create leaf node immediately if possible or revert to equal splitting.
        if (!overallBestSplit.second.isValid())
        {
            if (triangleRange.length() <= parameters.maxTriangleCountPerLeaf) return SplitResult();
            logWarning("LightBVHBuilder::computeSplitWithBinnedSAH() was not able to compute a proper split: reverting to LightBVHBuilder::computeSplitWithEqual()");
            return computeSplitWithEqual(data, triangleRange, nodeBounds, parameters);
        }

        // If the best split we found is more expensive than the cost of a leaf node (and we can create one), then create a leaf node.
        assert(overallBestSplit.second.isValid());
        if (parameters.useLeafCreationCost && triangleRange.length() <= parameters.maxTriangleCountPerLeaf)
        {
            float leafCost = evalSAH(nodeBounds, triangleRange.length(), parameters);
            if (leafCost <= overallBestSplit.first) return SplitResult();
        }

        return overallBestSplit.second;
    }

    /** Utility function that implements the orientation cost heuristic according to Equation 1
        in Conty & Kulla, "Importance Sampling of Many Lights with Adaptive Tree Splitting", 2018.
        We're assuming flat diffuse emitters (theta_e = pi/2). For this case the orientation cost
        varies smoothly between pi (flat emitter) to 4pi (full sphere).
    */
    static float computeOrientationCost(const float theta_o)
    {
        float theta_w = std::min(theta_o + glm::half_pi<float>(), glm::pi<float>());
        float sin_theta_o = std::sin(theta_o);
        float cos_theta_o = std::cos(theta_o);
        return glm::two_pi<float>() * (1.0f - cos_theta_o) + glm::half_pi<float>() * (2.0f * theta_w * sin_theta_o - std::cos(theta_o - 2.0f * theta_w) - 2.0f * theta_o * sin_theta_o + cos_theta_o);
    };

    /** Evaluates the SAOH cost metric for a node.
        If the node is empty (invalid bounds), the cost evaluates to zero.
        See Eqn 16 in Moreau and Clarberg, "Importance Sampling of Many Lights on the GPU", Ray Tracing Gems, Ch. 18, 2019.
    */
    static float evalSAOH(const BBox& bounds, const float flux, const float cosTheta, const LightBVHBuilder::Options& parameters)
    {
        float fluxCost = parameters.usePreintegration ? flux : 1.0f;
        float aabbCost = bounds.valid() ? (parameters.useVolumeOverSA ? bounds.volume(parameters.volumeEpsilon) : bounds.surfaceArea()) : 0.f;
        float theta = cosTheta != kInvalidCosConeAngle ? safeACos(cosTheta) : glm::pi<float>();
        float orientationCost = parameters.useLightingCones ? computeOrientationCost(theta) : 1.0f;
        float cost = fluxCost * aabbCost * orientationCost;
        assert(cost >= 0.f && !std::isnan(cost) && !std::isinf(cost));
        return cost;
    }

    LightBVHBuilder::SplitResult LightBVHBuilder::computeSplitWithBinnedSAOH(const BuildingData& data, const Range& triangleRange, const BBox& nodeBounds, const Options& parameters)
    {
        std::pair<float, SplitResult> overallBestSplit = std::make_pair(std::numeric_limits<float>::infinity(), SplitResult());
        assert(!overallBestSplit.second.isValid());

        // Find the largest dimension.
        float3 dimensions = nodeBounds.dimensions();
        uint32_t largestDimension = dimensions[2] >= dimensions[0] && dimensions[2] >= dimensions[1] ?
            2 : (dimensions[1] >= dimensions[0] && dimensions[1] >= dimensions[2] ? 1 : 0);

        struct Bin
        {
            BBox bounds = BBox();
            uint32_t triangleCount = 0;
            float flux = 0.0f;
            float3 coneDirection = float3(0.0f);
            float cosConeAngle = 1.0f;

            Bin() = default;
            Bin(const TriangleSortData& tri) : bounds(tri.bounds), triangleCount(1), flux(tri.flux), coneDirection(tri.coneDirection), cosConeAngle(tri.cosConeAngle) {}
            Bin& operator|= (const Bin& rhs)
            {
                bounds |= rhs.bounds;
                triangleCount += rhs.triangleCount;
                flux += rhs.flux;
                coneDirection += rhs.coneDirection;
                // Note: cosConeAngle should be computed separately after the final cone direction is known
                return *this;
            }
        };

        assert(parameters.binCount > 1);
        std::vector<Bin> bins(parameters.binCount);
        std::vector<float> costs(parameters.binCount - 1);

        /** Helper function that computes the best split along the given dimension using the SAOH metric.
            The triangles are binned to n bins, storing only the aggregate parameters (triangle count, bounds, flux, and cone direction).
            Then the cost metric is evaluated for each of the n-1 potential splits.
            Note that while the bounds and flux are accurately represented by the aggregated parameters,
            the bounding cones are approximates based on the bins' bounding cones. This is less expensive,
            but also less precise than computing them directly from the triangles.
        */
        const auto binAlongDimension = [&bins, &costs, &triangleRange, &data, &parameters, &overallBestSplit, &nodeBounds, largestDimension, dimensions](uint32_t dimension)
        {
            // Helper to compute the bin id for a given triangle.
            auto getBinId = [&](const TriangleSortData& td)
            {
                float bmin = nodeBounds.minPoint[dimension], bmax = nodeBounds.maxPoint[dimension];
                float w = bmax - bmin;
                assert(w >= 0.f); // The node bounds can be zero if all primitives are axis-aligned and coplanar
                float scale = w > FLT_MIN ? (float)parameters.binCount / w : 0.f;
                float p = td.bounds.centroid()[dimension];
                assert(bmin <= p && p <= bmax);
                return std::min((uint32_t)((p - bmin) * scale), parameters.binCount - 1);
            };

            // Reset the bins.
            for (Bin& bin : bins) bin = Bin();

            // Fill the bins with all triangles.
            for (uint32_t i = triangleRange.begin; i < triangleRange.end; ++i)
            {
                const auto& td = data.trianglesData[i];
                bins[getBinId(td)] |= td;
            }

            // Compute the lighting cones for each bin.
            // The cone direction is the average direction over all lights in the bin and the cone angle is grown to include all.
            // If the vector is zero length (no lights or if all directions cancelled out), the cone is marked as invalid.
            // TODO: Switch to a more sophisticated algorithm to get narrower cones.
            for (Bin& bin : bins)
            {
                bin.cosConeAngle = glm::length(bin.coneDirection) < FLT_MIN ? kInvalidCosConeAngle : 1.0f;
                bin.coneDirection = glm::normalize(bin.coneDirection);
            }
            for (uint32_t i = triangleRange.begin; i < triangleRange.end; ++i)
            {
                const auto& td = data.trianglesData[i];
                Bin& bin = bins[getBinId(td)];
                bin.cosConeAngle = computeCosConeAngle(bin.coneDirection, bin.cosConeAngle, td.coneDirection, td.cosConeAngle);
            }

            // First, compute A_j(L) * N_j(L) by sweeping over the bins from left to right.
            // Note that the costs vector has n-1 elements when there are n bins; the i:th elements represents the split between bin i and i+1.
            Bin total = Bin();
            for (std::size_t i = 0; i < costs.size(); ++i)
            {
                total |= bins[i];

                // Compute the bounding cone angle for the union of bins 0..i.
                float cosTheta = kInvalidCosConeAngle;
                if (glm::length(total.coneDirection) >= FLT_MIN)
                {
                    cosTheta = 1.f;
                    float3 coneDir = glm::normalize(total.coneDirection);
                    for (std::size_t j = 0; j <= i; ++j)
                    {
                        cosTheta = computeCosConeAngle(coneDir, cosTheta, bins[j].coneDirection, bins[j].cosConeAngle);
                    }
                }

                costs[i] = evalSAOH(total.bounds, total.flux, cosTheta, parameters);
            }

            // Then, compute A_j(R) * N_j(R) by sweeping over the bins from right to left.
            total = Bin();
            for (std::size_t i = costs.size(); i > 0; --i)
            {
                total |= bins[i];

                // Compute the bounding cone angle for the union of bins i..n-1.
                float cosTheta = kInvalidCosConeAngle;
                if (glm::length(total.coneDirection) >= FLT_MIN)
                {
                    cosTheta = 1.f;
                    float3 coneDir = glm::normalize(total.coneDirection);
                    for (std::size_t j = i; j <= costs.size(); ++j)
                    {
                        cosTheta = computeCosConeAngle(coneDir, cosTheta, bins[j].coneDirection, bins[j].cosConeAngle);
                    }
                }

                costs[i - 1] += evalSAOH(total.bounds, total.flux, cosTheta, parameters);
            }

            // Compute the cheapest split along the current dimension.
            std::pair<float, SplitResult> axisBestSplit = std::make_pair(std::numeric_limits<float>::infinity(), SplitResult{ dimension, 0 });
            for (uint32_t i = 0, triIdx = triangleRange.begin; i < costs.size(); ++i)
            {
                triIdx += bins[i].triangleCount;
                if (costs[i] < axisBestSplit.first)
                {
                    axisBestSplit = std::make_pair(costs[i], SplitResult{ dimension, triIdx });
                }
            }
            assert(triangleRange.begin <= axisBestSplit.second.triangleIndex && axisBestSplit.second.triangleIndex <= triangleRange.end);

            // Scale the cost by the ratio of the node's extent to discourage long skinny nodes.
            axisBestSplit.first *= static_cast<float>(dimensions[largestDimension]) / static_cast<float>(dimensions[dimension]);

            // Early out if all lights fall on either side of the split.
            if (axisBestSplit.second.triangleIndex == triangleRange.begin ||
                axisBestSplit.second.triangleIndex == triangleRange.end) return;

            if (axisBestSplit.first < overallBestSplit.first)
            {
                overallBestSplit = axisBestSplit;
                assert(triangleRange.begin < overallBestSplit.second.triangleIndex && overallBestSplit.second.triangleIndex < triangleRange.end);
            }
        };

        // Compute the best split.
        if (parameters.splitAlongLargest)
        {
            binAlongDimension(largestDimension);
        }
        else
        {
            for (uint32_t dimension = 0; dimension < 3; ++dimension)
            {
                binAlongDimension(dimension);
            }
        }

        // If we couldn't find a valid split, create leaf node immediately if possible or revert to equal splitting.
        if (!overallBestSplit.second.isValid())
        {
            if (triangleRange.length() <= parameters.maxTriangleCountPerLeaf) return SplitResult();
            logWarning("LightBVHBuilder::computeSplitWithBinnedSAOH() was not able to compute a proper split: reverting to LightBVHBuilder::computeSplitWithEqual()");
            return computeSplitWithEqual(data, triangleRange, nodeBounds, parameters);
        }

        // If the best split we found is more expensive than the cost of a leaf node (and we can create one), then create a leaf node.
        assert(overallBestSplit.second.isValid());
        if (parameters.useLeafCreationCost && triangleRange.length() <= parameters.maxTriangleCountPerLeaf)
        {
            // Evaluate the cost metric for the node. This requires us to first compute the cone angle.
            float cosTheta = kInvalidCosConeAngle;
            computeLightingCone(triangleRange, data, cosTheta);
            float leafCost = evalSAOH(nodeBounds, data.currentNodeFlux, cosTheta, parameters);
            if (leafCost <= overallBestSplit.first) return SplitResult();
        }

        return overallBestSplit.second;
    }

    LightBVHBuilder::SplitHeuristicFunction LightBVHBuilder::getSplitFunction(SplitHeuristic heuristic)
    {
        switch (heuristic)
        {
        case SplitHeuristic::Equal:
            return computeSplitWithEqual;
        case SplitHeuristic::BinnedSAH:
            return computeSplitWithBinnedSAH;
        case SplitHeuristic::BinnedSAOH:
            return computeSplitWithBinnedSAOH;
        default:
            logError("Unsupported SplitHeuristic: " + std::to_string(static_cast<uint32_t>(heuristic)));
            return nullptr;
        }
    }

    SCRIPT_BINDING(LightBVHBuilder)
    {
        pybind11::enum_<LightBVHBuilder::SplitHeuristic> splitHeuristic(m, "SplitHeuristic");
        splitHeuristic.value("Equal", LightBVHBuilder::SplitHeuristic::Equal);
        splitHeuristic.value("BinnedSAH", LightBVHBuilder::SplitHeuristic::BinnedSAH);
        splitHeuristic.value("BinnedSAOH", LightBVHBuilder::SplitHeuristic::BinnedSAOH);

        // TODO use a nested class in the bindings when supported.
        ScriptBindings::SerializableStruct<LightBVHBuilder::Options> options(m, "LightBVHBuilderOptions");
#define field(f_) field(#f_, &LightBVHBuilder::Options::f_)
        options.field(splitHeuristicSelection);
        options.field(maxTriangleCountPerLeaf);
        options.field(binCount);
        options.field(volumeEpsilon);
        options.field(splitAlongLargest);
        options.field(useVolumeOverSA);
        options.field(useLeafCreationCost);
        options.field(createLeavesASAP);
        options.field(allowRefitting);
        options.field(usePreintegration);
        options.field(useLightingCones);
#undef field
    }
}
