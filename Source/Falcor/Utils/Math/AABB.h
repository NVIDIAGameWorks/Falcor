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
#include "Core/Macros.h"
#include "Core/API/Raytracing.h"
#include "Utils/Math/Matrix.h"
#include "Utils/Math/Vector.h"
#include <limits>

namespace Falcor
{
    /** Axis-aligned bounding box (AABB) stored by its min/max points.

        The user is responsible for checking the validity of returned AABBs.
        There is an equivalent GPU-side implementation in the AABB.slang module.
    */
    struct AABB
    {
        float3 minPoint = float3(std::numeric_limits<float>::infinity());   ///< Minimum point.
        float3 maxPoint = float3(-std::numeric_limits<float>::infinity());  ///< Maximum point. If any minPoint > maxPoint the box is invalid.

        /** Construct bounding box initialized to +/-inf.
        */
        AABB() = default;

        /** Construct bounding box initialized to single point.
        */
        AABB(const float3& p) : minPoint(p), maxPoint(p) {}

        /** Construct bounding box initialized to min/max point.
        */
        AABB(const float3& pmin, const float3& pmax) : minPoint(pmin), maxPoint(pmax) {}

        /** Set box to single point.
        */
        void set(const float3& p) { minPoint = maxPoint = p; }

        /** Set the box corners explicitly.
        */
        void set(const float3& pmin, const float3& pmax)
        {
            minPoint = pmin;
            maxPoint = pmax;
        }

        /** Invalidates the box.
        */
        void invalidate()
        {
            minPoint = float3(std::numeric_limits<float>::infinity());
            maxPoint = float3(-std::numeric_limits<float>::infinity());
        }

        /** Returns true if bounding box is valid (all dimensions zero or larger).
        */
        bool valid() const
        {
            return maxPoint.x >= minPoint.x && maxPoint.y >= minPoint.y && maxPoint.z >= minPoint.z;
        }

        /** Grows the box to include the point p.
        */
        AABB& include(const float3& p)
        {
            minPoint = min(minPoint, p);
            maxPoint = max(maxPoint, p);
            return *this;
        }

        /** Grows the box to include another box.
        */
        AABB& include(const AABB& b)
        {
            minPoint = min(minPoint, b.minPoint);
            maxPoint = max(maxPoint, b.maxPoint);
            return *this;
        }

        /** Make the box be the intersection between this and another box.
        */
        AABB& intersection(const AABB& b)
        {
            minPoint = glm::max(minPoint, b.minPoint);
            maxPoint = glm::min(maxPoint, b.maxPoint);
            return *this;
        }

        /** Returns the box center.
            \return Center of the box if valid, undefined otherwise.
        */
        float3 center() const
        {
            return (minPoint + maxPoint) * 0.5f;
        }

        /** Returns the box extent.
            \return Size of the box if valid, undefined otherwise.
        */
        float3 extent() const
        {
            return maxPoint - minPoint;
        }

        /** Returns the surface area of the box.
            \return Surface area if box is valid, undefined otherwise.
        */
        float area() const
        {
            float3 e = extent();
            return (e.x * e.y + e.x * e.z + e.y * e.z) * 2.f;
        }

        /** Return the volume of the box.
            \return Volume if the box is valid, undefined otherwise.
        */
        float volume() const
        {
            float3 e = extent();
            return e.x * e.y * e.z;
        }

        /** Returns the radius of the minimal sphere that encloses the box.
            \return Radius of minimal bounding sphere, or undefined if box is invalid.
        */
        float radius() const
        {
            return 0.5f * glm::length(extent());
        }

        /** Calculates the bounding box transformed by a matrix.
            \param[in] mat Transform matrix
            \return Bounding box after transformation.
        */
        AABB transform(const rmcv::mat4& rmcv_mat) const
        {
            if (!valid()) return {};

            auto mat = rmcv::toGLM(rmcv_mat);

            float3 xa = float3(mat[0] * minPoint.x);
            float3 xb = float3(mat[0] * maxPoint.x);
            float3 xMin = glm::min(xa, xb);
            float3 xMax = glm::max(xa, xb);

            float3 ya = float3(mat[1] * minPoint.y);
            float3 yb = float3(mat[1] * maxPoint.y);
            float3 yMin = glm::min(ya, yb);
            float3 yMax = glm::max(ya, yb);

            float3 za = float3(mat[2] * minPoint.z);
            float3 zb = float3(mat[2] * maxPoint.z);
            float3 zMin = glm::min(za, zb);
            float3 zMax = glm::max(za, zb);

            float3 newMin = xMin + yMin + zMin + float3(mat[3]);
            float3 newMax = xMax + yMax + zMax + float3(mat[3]);

            return AABB(newMin, newMax);
        }

        /** Checks whether two bounding boxes are equal.
        */
        bool operator== (const AABB& rhs) const
        {
            return minPoint == rhs.minPoint && maxPoint == rhs.maxPoint;
        }

        /** Checks whether two bounding boxes are not equal.
        */
        bool operator!= (const AABB& rhs) const
        {
            return minPoint != rhs.minPoint || maxPoint != rhs.maxPoint;
        }

        /** Union of two boxes.
        */
        AABB& operator|= (const AABB& rhs)
        {
            return include(rhs);
        }

        /** Union of two boxes.
        */
        AABB operator| (const AABB& rhs) const
        {
            AABB bb = *this;
            return bb |= rhs;
        }

        /** Intersection of two boxes.
        */
        AABB& operator&= (const AABB& rhs)
        {
            return intersection(rhs);
        }

        /** Intersection of two boxes.
        */
        AABB operator& (const AABB& rhs) const
        {
            AABB bb = *this;
            return bb &= rhs;
        }

        /** Conversion to RtAABB.
        */
        explicit operator RtAABB() const
        {
            return { minPoint, maxPoint };
        }
    };
}
