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
#include "Utils/Math/Vector.h"
#include <limits>

namespace Falcor
{
    /** An axis-aligned bounding box stored by its min/max points.
        The user is responsible for checking the validity of returned bounding-boxes before using them.
        Note: Falcor already has an AABB class that works differently, hence the name.
    */
    struct BBox
    {
        float3 minPoint = float3(std::numeric_limits<float>::infinity());     // +inf
        float3 maxPoint = float3(-std::numeric_limits<float>::infinity());    // -inf

        BBox() {}
        BBox(const glm::float3& p) : minPoint(p), maxPoint(p) {}

        /** Returns true if bounding box is valid (all dimensions zero or larger). */
        bool valid() const { return maxPoint.x >= minPoint.x && maxPoint.y >= minPoint.y && maxPoint.z >= minPoint.z; }

        /** Returns the dimensions of the bounding box. */
        float3 dimensions() const { return maxPoint - minPoint; }

        /** Returns the centroid of the bounding box. */
        float3 centroid() const { return (minPoint + maxPoint) * 0.5f; }

        /** Returns the surface area of the bounding box. */
        float surfaceArea() const
        {
            const float3 dims = dimensions();
            return 2.0f * (dims.x * dims.y + dims.y * dims.z + dims.x * dims.z);
        }

        /** Returns the volume of the bounding box.
            \param[in] epsilon Replace dimensions that are zero by this value.
            \return the volume of the bounding box if it is valid, -inf otherwise.
        */
        float volume(float epsilon = 0.0f) const
        {
            if (valid() == false)
            {
                return -std::numeric_limits<float>::infinity();
            }

            const float3 dims = glm::max(float3(epsilon), dimensions());
            return dims.x * dims.y * dims.z;
        }

        /** Union of two boxes. */
        BBox& operator|= (const BBox& rhs)
        {
            minPoint = glm::min(minPoint, rhs.minPoint);
            maxPoint = glm::max(maxPoint, rhs.maxPoint);
            return *this;
        }

        BBox operator| (const BBox& rhs) const { BBox bb = *this; bb |= rhs; return bb; }

        /** Intersection of two boxes. */
        BBox& operator&= (const BBox& rhs)
        {
            minPoint = glm::max(minPoint, rhs.minPoint);
            maxPoint = glm::min(maxPoint, rhs.maxPoint);
            return *this;
        }

        BBox operator& (const BBox& rhs) const { BBox bb = *this; bb &= rhs; return bb; }
    };
}
