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
#include "Utils/Math/MathConstants.slangh"

/** Axis-aligned bounding box (AABB).
*/
struct AABB
{
    float3 minPoint;    ///< Minimum point.
    float3 maxPoint;    ///< Maximum point. If any minPoint > maxPoint the box is invalid.

    /** Initializes an AABB.
        Note if minPoint > maxPoint in any component the box is invalid.
    */
    __init(float3 minPoint, float3 maxPoint)
    {
        this.minPoint = minPoint;
        this.maxPoint = maxPoint;
    }

    /** Set box to single point.
    */
    [mutating] void set(float3 p)
    {
        minPoint = p;
        maxPoint = p;
    }

    /** Set the box corners explicitly.
        Note if min > max in any component the box is invalid.
    */
    [mutating] void set(float3 _min, float3 _max)
    {
        minPoint = _min;
        maxPoint = _max;
    }

    /** Invalidates the box.
    */
    [mutating] void invalidate()
    {
        minPoint = FLT_MAX;
        maxPoint = -FLT_MAX;
    }

    /** Returns true if the box is valid.
    */
    bool valid()
    {
        return all(minPoint <= maxPoint);
    }

    /** Grows the box to include the point p.
    */
    [mutating] void include(float3 p)
    {
        minPoint = min(minPoint, p);
        maxPoint = max(maxPoint, p);
    }

    /** Grows the box to include another box.
    */
    [mutating] void include(AABB b)
    {
        minPoint = min(minPoint, b.minPoint);
        maxPoint = max(maxPoint, b.maxPoint);
    }

    /** Check if point is included in the box.
        \return True if p is in the box (inclusive test), false if outside or box invalid.
    */
    bool contains(float3 p)
    {
        return valid() && all(p >= minPoint && p <= maxPoint);
    }

    /** Returns the box center.
        \return Center of the box if valid, undefined otherwise.
    */
    float3 center()
    {
        return (minPoint + maxPoint) * 0.5f;
    }

    /** Returns the box extent.
        \return Size of the box if valid, undefined otherwise.
    */
    float3 extent()
    {
        return maxPoint - minPoint;
    }

    /** Returns the surface area of the box.
        \return Surface area if box is valid, undefined otherwise.
    */
    float area()
    {
        float3 e = extent();
        return (e.x * e.y + e.x * e.z + e.y * e.z) * 2.f;
    }

    /** Return the volume of the box.
        \return Volume if the box is valid, undefined otherwise.
    */
    float volume()
    {
        float3 e = extent();
        return e.x * e.y * e.z;
    }

    /** Returns the radius of the minimal sphere that encloses the box.
        \return Radius of minimal bounding sphere, or undefined if box is invalid.
    */
    float radius()
    {
        return 0.5f * length(extent());
    }

    /** Check if two boxes intersect. The test is inclusive on both sides.
        \param[in] other The other box.
        \return True if the two boxes intersect. The result is undefined if either box is invalid.
    */
    bool intersects(const AABB other)
    {
        return all(maxPoint >= other.minPoint && minPoint <= other.maxPoint);
    }

    /** Returns the minimum distance between the box and a point.
        The distance is the minimum distance between any point in the box and the other point.
        \param[in] p The point to which to compute the distance.
        \return Minimum distance between between point and box, or 0 if the point lies inside. The result is undefined if the box is invalid.
    */
    float minDistance(float3 p)
    {
        float3 d1 = minPoint - p;
        float3 d2 = p - maxPoint;
        float3 d = max(max(d1, d2), 0.f);
        return length(d);
    }

    /** Returns the minimum distance between two boxes.
        The distance is the minimum distance between any two points in the boxes.
        \param[in] other The other box.
        \return Minimum distance between boxes, or 0 if they intersect. The result is undefined if either box is invalid.
    */
    float minDistance(const AABB other)
    {
        float3 d1 = minPoint - other.maxPoint;
        float3 d2 = other.minPoint - maxPoint;
        float3 d = max(max(d1, d2), 0.f);
        return length(d);
    }
};
