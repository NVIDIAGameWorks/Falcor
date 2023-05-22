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
#include "Utils/Math/Vector.h"
#include <limits>

namespace Falcor
{
/**
 * Axis-aligned UV tile stored by its min/max points.
 */
struct Rectangle
{
    float2 minPoint = float2(std::numeric_limits<float>::infinity());  ///< Minimum point.
    float2 maxPoint = float2(-std::numeric_limits<float>::infinity()); ///< Maximum point. If any minPoint > maxPoint the tile is invalid.

    /// Construct bounding tile initialized to +/-inf.
    Rectangle() = default;

    /// Construct bounding tile initialized to single point.
    Rectangle(const float2& p) : minPoint(p), maxPoint(p) {}

    /// Construct bounding tile initialized to min/max point.
    Rectangle(const float2& pmin, const float2& pmax) : minPoint(pmin), maxPoint(pmax) {}

    /// Set tile to single point.
    void set(const float2& p) { minPoint = maxPoint = p; }

    /// Set the tile corners explicitly.
    void set(const float2& pmin, const float2& pmax)
    {
        minPoint = pmin;
        maxPoint = pmax;
    }

    /// Invalidates the tile.
    void invalidate()
    {
        minPoint = float2(std::numeric_limits<float>::infinity());
        maxPoint = float2(-std::numeric_limits<float>::infinity());
    }

    /// Returns true if bounding tile is valid (all dimensions zero or larger).
    bool valid() const { return maxPoint.x >= minPoint.x && maxPoint.y >= minPoint.y; }

    /// Grows the tile to include the point p.
    Rectangle& include(const float2& p)
    {
        minPoint = min(minPoint, p);
        maxPoint = max(maxPoint, p);
        return *this;
    }

    /// Grows the tile to include another tile.
    Rectangle& include(const Rectangle& b)
    {
        minPoint = min(minPoint, b.minPoint);
        maxPoint = max(maxPoint, b.maxPoint);
        return *this;
    }

    /// Make the tile be the intersection between this and another tile.
    Rectangle& intersection(const Rectangle& b)
    {
        minPoint = max(minPoint, b.minPoint);
        maxPoint = min(maxPoint, b.maxPoint);
        return *this;
    }

    /// Returns true if the two rectangles have any overlap.
    bool overlaps(Rectangle b)
    {
        b.intersection(*this);
        return b.valid() && b.area() > 0.f;
    }

    /// Returns true if the Rectangle `b` is fully contained within this Rectangle.
    bool contains(const Rectangle& b)
    {
        Rectangle temp = *this;
        return temp.include(b) == *this;
    }

    /**
     * Returns the tile center.
     * @return Center of the tile if valid, undefined otherwise.
     */
    float2 center() const { return (minPoint + maxPoint) * 0.5f; }

    /**
     * Returns the tile extent.
     * @return Size of the tile if valid, undefined otherwise.
     */
    float2 extent() const { return maxPoint - minPoint; }

    /**
     * Returns the surface area of the tile.
     * @return Surface area if tile is valid, undefined otherwise.
     */
    float area() const
    {
        float2 e = extent();
        return e.x * e.y;
    }

    /**
     * Returns the radius of the minimal sphere that encloses the tile.
     * @return Radius of minimal bounding sphere, or undefined if tile is invalid.
     */
    float radius() const { return 0.5f * length(extent()); }

    /// Checks whether two bounding tilees are equal.
    bool operator==(const Rectangle& rhs) const { return all(minPoint == rhs.minPoint) && all(maxPoint == rhs.maxPoint); }

    /// Checks whether two bounding tilees are not equal.
    bool operator!=(const Rectangle& rhs) const { return any(minPoint != rhs.minPoint) || any(maxPoint != rhs.maxPoint); }

    /// Union of two tilees.
    Rectangle& operator|=(const Rectangle& rhs) { return include(rhs); }

    /// Union of two tilees.
    Rectangle operator|(const Rectangle& rhs) const
    {
        Rectangle bb = *this;
        return bb |= rhs;
    }

    /// Intersection of two tilees.
    Rectangle& operator&=(const Rectangle& rhs) { return intersection(rhs); }

    /// Intersection of two tilees.
    Rectangle operator&(const Rectangle& rhs) const
    {
        Rectangle bb = *this;
        return bb &= rhs;
    }
};
} // namespace Falcor
