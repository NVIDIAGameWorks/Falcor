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

#include "Falcor.h"
#include "Marker2DSet.h"

namespace Falcor
{
    class SelectionWheel
    {
    public:
        using SharedPtr = std::shared_ptr<SelectionWheel>;
        static const uint32_t kInvalidIndex = -1;

        struct Desc
        {
            std::vector<uint32_t> sectorGroups;     ///< Describes how many sectors each group should have in the selection wheel.
            float2 position;                        ///< Center position of the selection wheel.
            float minRadius;                        ///< The minimum radius of the selection wheel.
            float maxRadius;                        ///< The maximum radius of the selection wheel.
            float4 baseColor;                       ///< The base color of the selection wheel.
            float4 highlightColor;                  ///< The highlight color for the selected sector.
            float4 lineColor;                       ///< The color of the lines that separate sectors and groups.
            float borderWidth;                      ///< Thickness of the border in pixels.
        };

        static SharedPtr create(Marker2DSet::SharedPtr pMarker2DSet);

        void update(const float2& mousePos, const Desc& description);

        bool isMouseOnSector(const float2& mousePos, uint32_t groupIndex, uint32_t sectorIndex);
        bool isMouseOnGroup(const float2& mousePos, uint32_t groupIndex, uint32_t& sectorIndex);

        float2 getCenterPositionOfSector(uint32_t groupIndex, uint32_t sectorIndex);

        float getAngleOfSectorInGroup(uint32_t groupIndex);
        float getRotationOfSector(uint32_t groupIndex, uint32_t sectorIndex);
        float getGroupAngle();

    private:
        SelectionWheel(Marker2DSet::SharedPtr pMarker2DSet) : mpMarker2DSet(pMarker2DSet) {}

        void computeMouseAngleAndDirLength(const float2& mousePos, float& mouseAngle, float& dirLength);
        void computeGroupAndSectorIndexFromAngle(float mouseAngle, uint32_t& groupIndex, uint32_t& sectorIndex);

        /** Specialized add function for this class. When adding a circle sector this will allow for cutting off some parts of the sector's sides.
            \param[in] rotation The starting angle of the sector. It is the left side of the sector.
            \param[in] angle The angle that describes the circle sector.
            \param[in] color The color of the circle sector.
            \param[in] margin The cutoff of one of the sides. If positive, it will cut off from the left side, if negative, from the right, and if zero, no cut off.
            \param[in] marginOnBothSides If true, does the cut off on both sides instead of one. The margin variable need to be positive if this is true.
            \param[in] excludeBorderFlags Flags for which borders should be excluded from rendering.
        */
        void addCircleSector(float rotation, float angle, const float4& color, const float4& borderColor, float margin = 0.f, bool marginOnBothSides = false, ExcludeBorderFlags excludeBorderFlags = ExcludeBorderFlags::None);

    private:
        Desc                    mDescription;
        Marker2DSet::SharedPtr  mpMarker2DSet;
    };
}
