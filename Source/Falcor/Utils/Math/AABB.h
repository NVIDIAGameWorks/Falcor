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
#pragma once
#include "Core/Macros.h"
#include "Core/API/Raytracing.h"
#include "Utils/Math/Matrix.h"
#include "Utils/Math/Vector.h"
#include <limits>

namespace Falcor
{
/**
 * Axis-aligned bounding box (AABB) stored by its min/max points.
 *
 * The user is responsible for checking the validity of returned AABBs.
 * There is an equivalent GPU-side implementation in the AABB.slang module.
 */
struct AABB
{
    float3 minPoint = float3(std::numeric_limits<float>::infinity());  ///< Minimum point.
    float3 maxPoint = float3(-std::numeric_limits<float>::infinity()); ///< Maximum point. If any minPoint > maxPoint the box is invalid.

    /// Construct bounding box initialized to +/-inf.
    AABB() = default;

    /// Construct bounding box initialized to single point.
    AABB(const float3& p) : minPoint(p), maxPoint(p) {}

    /// Construct bounding box initialized to min/max point.
    AABB(const float3& pmin, const float3& pmax) : minPoint(pmin), maxPoint(pmax) {}

    /// Set box to single point.
    void set(const float3& p) { minPoint = maxPoint = p; }

    /// Set the box corners explicitly.
    void set(const float3& pmin, const float3& pmax)
    {
        minPoint = pmin;
        maxPoint = pmax;
    }

    /// Invalidates the box.
    void invalidate()
    {
        minPoint = float3(std::numeric_limits<float>::infinity());
        maxPoint = float3(-std::numeric_limits<float>::infinity());
    }

    /// Returns true if bounding box is valid (all dimensions zero or larger).
    bool valid() const { return maxPoint.x >= minPoint.x && maxPoint.y >= minPoint.y && maxPoint.z >= minPoint.z; }

    /// Grows the box to include the point p.
    AABB& include(const float3& p)
    {
        minPoint = min(minPoint, p);
        maxPoint = max(maxPoint, p);
        return *this;
    }

    /// Grows the box to include another box.
    AABB& include(const AABB& b)
    {
        minPoint = min(minPoint, b.minPoint);
        maxPoint = max(maxPoint, b.maxPoint);
        return *this;
    }

    /// Make the box be the intersection between this and another box.
    AABB& intersection(const AABB& b)
    {
        minPoint = max(minPoint, b.minPoint);
        maxPoint = min(maxPoint, b.maxPoint);
        return *this;
    }

    /// Returns true if the two AABBs have any overlap.
    bool overlaps(AABB b)
    {
        b.intersection(*this);
        return b.valid() && b.volume() > 0.f;
    }

    /// Returns true if the AABB `b` is fully contained within this AABB.
    bool contains(const AABB& b)
    {
        AABB temp = *this;
        return temp.include(b) == *this;
    }

    /**
     * Returns the box center.
     * @return Center of the box if valid, undefined otherwise.
     */
    float3 center() const { return (minPoint + maxPoint) * 0.5f; }

    /**
     * Returns the box extent.
     * @return Size of the box if valid, undefined otherwise.
     */
    float3 extent() const { return maxPoint - minPoint; }

    /**
     * Returns the surface area of the box.
     * @return Surface area if box is valid, undefined otherwise.
     */
    float area() const
    {
        float3 e = extent();
        return (e.x * e.y + e.x * e.z + e.y * e.z) * 2.f;
    }

    /**
     * Return the volume of the box.
     * @return Volume if the box is valid, undefined otherwise.
     */
    float volume() const
    {
        float3 e = extent();
        return e.x * e.y * e.z;
    }

    /**
     * Returns the radius of the minimal sphere that encloses the box.
     * @return Radius of minimal bounding sphere, or undefined if box is invalid.
     */
    float radius() const { return 0.5f * length(extent()); }

    /**
     * Calculates the bounding box transformed by a matrix.
     * @param[in] mat Transform matrix
     * @return Bounding box after transformation.
     */
    AABB transform(const float4x4& mat) const
    {
        if (!valid())
            return {};

        float3 xa = mat.getCol(0).xyz() * minPoint.x;
        float3 xb = mat.getCol(0).xyz() * maxPoint.x;
        float3 xMin = min(xa, xb);
        float3 xMax = max(xa, xb);

        float3 ya = mat.getCol(1).xyz() * minPoint.y;
        float3 yb = mat.getCol(1).xyz() * maxPoint.y;
        float3 yMin = min(ya, yb);
        float3 yMax = max(ya, yb);

        float3 za = mat.getCol(2).xyz() * minPoint.z;
        float3 zb = mat.getCol(2).xyz() * maxPoint.z;
        float3 zMin = min(za, zb);
        float3 zMax = max(za, zb);

        float3 newMin = xMin + yMin + zMin + mat.getCol(3).xyz();
        float3 newMax = xMax + yMax + zMax + mat.getCol(3).xyz();

        return AABB(newMin, newMax);
    }

    /// Checks whether two bounding boxes are equal.
    bool operator==(const AABB& rhs) const { return all(minPoint == rhs.minPoint) && all(maxPoint == rhs.maxPoint); }

    /// Checks whether two bounding boxes are not equal.
    bool operator!=(const AABB& rhs) const { return any(minPoint != rhs.minPoint) || any(maxPoint != rhs.maxPoint); }

    /// Union of two boxes.
    AABB& operator|=(const AABB& rhs) { return include(rhs); }

    /// Union of two boxes.
    AABB operator|(const AABB& rhs) const
    {
        AABB bb = *this;
        return bb |= rhs;
    }

    /// Intersection of two boxes.
    AABB& operator&=(const AABB& rhs) { return intersection(rhs); }

    /// Intersection of two boxes.
    AABB operator&(const AABB& rhs) const
    {
        AABB bb = *this;
        return bb &= rhs;
    }

    /// Conversion to RtAABB.
    explicit operator RtAABB() const { return {minPoint, maxPoint}; }
};
} // namespace Falcor
