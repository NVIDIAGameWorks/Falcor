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
#include "Vector.h"

namespace Falcor
{
    /** An Axis-Aligned Bounding Box
    */
    struct BoundingBox
    {
        float3 center; ///< Center position of the bounding box
        float3 extent; ///< Half length of each side. Essentially the coordinates to the max corner relative to the center.

        /** Checks whether two bounding boxes are equivalent in position and size
        */
        bool operator==(const BoundingBox& other)
        {
            return (other.center == center) && (other.extent == extent);
        }

        /** Calculates the bounding box transformed by a matrix
            \param[in] mat Transform matrix
            \return Bounding box after transformation
        */
        BoundingBox transform(const glm::mat4& mat) const
        {
            float3 min = center - extent;
            float3 max = center + extent;

            float3 xa = float3(mat[0] * min.x);
            float3 xb = float3(mat[0] * max.x);
            float3 xMin = glm::min(xa, xb);
            float3 xMax = glm::max(xa, xb);

            float3 ya = float3(mat[1] * min.y);
            float3 yb = float3(mat[1] * max.y);
            float3 yMin = glm::min(ya, yb);
            float3 yMax = glm::max(ya, yb);

            float3 za = float3(mat[2] * min.z);
            float3 zb = float3(mat[2] * max.z);
            float3 zMin = glm::min(za, zb);
            float3 zMax = glm::max(za, zb);


            float3 newMin = xMin + yMin + zMin + float3(mat[3]);
            float3 newMax = xMax + yMax + zMax + float3(mat[3]);

            return BoundingBox::fromMinMax(newMin, newMax);
        }

        /** Gets the minimum position of the bounding box
            \return Minimum position
        */
        float3 getMinPos() const
        {
            return center - extent;
        }

        /** Gets the maximum position of the bounding box
            \return Maximum position
        */
        float3 getMaxPos() const
        {
            return center + extent;
        }

        /** Gets the size of each dimension of the bounding box.
            \return X,Y and Z lengths of the bounding box
        */
        float3 getSize() const
        {
            return extent * 2.0f;
        }

        /** Construct a bounding box from a minimum and maximum point.
            \param[in] min Minimum point
            \param[in] max Maximum point
            \return A bounding box
        */
        static BoundingBox fromMinMax(const float3& min, const float3& max)
        {
            BoundingBox box;
            box.center = (max + min) * float3(0.5f);
            box.extent = (max - min) * float3(0.5f);
            return box;
        }

        /** Constructs a bounding box from the union of two other bounding boxes.
            \param[in] bb0 First bounding box
            \param[in] bb1 Second bounding box
            \return A bounding box
        */
        static BoundingBox fromUnion(const BoundingBox& bb0, const BoundingBox& bb1)
        {
            return BoundingBox::fromMinMax(min(bb0.getMinPos(), bb1.getMinPos()), max(bb0.getMaxPos(), bb1.getMaxPos()));
        }
    };
}
