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
#include "SelectionWheel.h"

using namespace Falcor;

Falcor::SelectionWheel::SharedPtr Falcor::SelectionWheel::create(Marker2DSet::SharedPtr pMarker2DSet)
{
    return SharedPtr(new SelectionWheel(pMarker2DSet));
}

void Falcor::SelectionWheel::update(const float2& mousePos, const Desc& description)
{
    const float kPI = (float)M_PI;
    const float kPI2 = kPI * 2.f;
    const float kStartOffest = kPI / 2.f;
    mDescription = description;

    const uint32_t groupCount = (uint32_t)mDescription.sectorGroups.size();
    float minSectorAngle = FLT_MAX;
    for (uint32_t groupIndex = 0; groupIndex < groupCount; groupIndex++)
    {
        float sectorAngle = getAngleOfSectorInGroup(groupIndex);
        minSectorAngle = std::fminf(sectorAngle, minSectorAngle);
    }

    const float groupAngle = getGroupAngle();
    float halfGroupSpacingAngle = minSectorAngle * 0.05f;

    uint32_t excludeGroup = UINT32_MAX;

    float dirLength;
    float mouseAngle;
    computeMouseAngleAndDirLength(mousePos, mouseAngle, dirLength);

    float4 borderColor = mDescription.lineColor;
    borderColor.w = mDescription.borderWidth;

    // Check if mouse is hovering over the selection wheel
    if (dirLength >= mDescription.minRadius && dirLength <= mDescription.maxRadius)
    {
        // Highlight the sector that the mouse is over.
        uint32_t groupIndex, sectorIndex;
        computeGroupAndSectorIndexFromAngle(mouseAngle, groupIndex, sectorIndex);
        excludeGroup = groupIndex;

        float rotation = getRotationOfSector(groupIndex, sectorIndex);
        float sectorAngle = getAngleOfSectorInGroup(groupIndex);

        const uint32_t kSectorCount = mDescription.sectorGroups[groupIndex];
        uint32_t lastSectorIndex = kSectorCount - 1;
        // If the selected sector is on the sides, render two circle sectors, one for the selected and one for the others.
        if (sectorIndex == 0 || sectorIndex == lastSectorIndex)
        {
            // The other sectors.
            float groupSpacing = sectorIndex == 0 ? halfGroupSpacingAngle : -halfGroupSpacingAngle;
            float offset = sectorIndex == 0 ? sectorAngle : 0.0f;
            addCircleSector(groupAngle * groupIndex + offset, groupAngle - sectorAngle, mDescription.baseColor, borderColor, -groupSpacing, false, sectorIndex == 0 ? ExcludeBorderFlags::Left : ExcludeBorderFlags::Right);

            // Highlighted sector.
            addCircleSector(rotation, sectorAngle, mDescription.highlightColor, borderColor, groupSpacing);
        }
        else // If the selected sector is in the middle, render two circle sectors for the surrounding sectors and one for the selected sector.
        {
            // Highlighted sector.
            addCircleSector(rotation, sectorAngle, mDescription.highlightColor, borderColor);

            // Left sectors.
            addCircleSector(groupAngle * groupIndex, sectorAngle * sectorIndex, mDescription.baseColor, borderColor, halfGroupSpacingAngle, false, ExcludeBorderFlags::Right);

            // Right sectors.
            addCircleSector(groupAngle * groupIndex + sectorAngle * (sectorIndex + 1), sectorAngle * (lastSectorIndex - sectorIndex), mDescription.baseColor, borderColor, -halfGroupSpacingAngle, false, ExcludeBorderFlags::Left);
        }
    }

    // Draw the groups that were not hovered over.
    for (uint32_t groupIndex = 0; groupIndex < groupCount; groupIndex++)
    {
        if (excludeGroup != groupIndex)
        {
            addCircleSector(groupAngle * groupIndex, groupAngle, mDescription.baseColor, borderColor, halfGroupSpacingAngle, true);
        }
    }
}

bool Falcor::SelectionWheel::isMouseOnSector(const float2& mousePos, uint32_t groupIndex, uint32_t sectorIndex)
{
    checkArgument(groupIndex < (uint32_t)mDescription.sectorGroups.size(), "'groupIndex' ({}) is out of bounds.", groupIndex);
    checkArgument(sectorIndex < mDescription.sectorGroups[groupIndex], "'sectorIndex' ({}) is out of bounds.", sectorIndex);

    float dirLength, mouseAngle;
    computeMouseAngleAndDirLength(mousePos, mouseAngle, dirLength);

    float rotation = getRotationOfSector(groupIndex, sectorIndex);
    float sectorAngle = getAngleOfSectorInGroup(groupIndex);
    return rotation <= mouseAngle && mouseAngle <= (rotation + sectorAngle) && dirLength >= mDescription.minRadius && dirLength <= mDescription.maxRadius;
}

bool Falcor::SelectionWheel::isMouseOnGroup(const float2& mousePos, uint32_t groupIndex, uint32_t& sectorIndex)
{
    checkArgument(groupIndex < (uint32_t)mDescription.sectorGroups.size(), "'groupIndex' ({}) is out of bounds.", groupIndex);

    float dirLength, mouseAngle;
    computeMouseAngleAndDirLength(mousePos, mouseAngle, dirLength);

    float minRotation = getRotationOfSector(groupIndex, 0);
    float sectorAngle = getAngleOfSectorInGroup(groupIndex);
    float maxRotation = minRotation + sectorAngle * mDescription.sectorGroups[groupIndex];

    bool isInGroup = minRotation <= mouseAngle && mouseAngle <= maxRotation && dirLength >= mDescription.minRadius && dirLength <= mDescription.maxRadius;

    if (isInGroup)
    {
        sectorIndex = (uint32_t)std::floor((mouseAngle - minRotation) / sectorAngle);
    }
    else
    {
        sectorIndex = kInvalidIndex;
    }

    return isInGroup;
}

float2 Falcor::SelectionWheel::getCenterPositionOfSector(uint32_t groupIndex, uint32_t sectorIndex)
{
    checkArgument(groupIndex < (uint32_t)mDescription.sectorGroups.size(), "'groupIndex' ({}) is out of bounds.", groupIndex);
    checkArgument(sectorIndex < mDescription.sectorGroups[groupIndex], "'sectorIndex' ({}) is out of bounds.", sectorIndex);
    float rotation = getRotationOfSector(groupIndex, sectorIndex);
    float sectorAngle = getAngleOfSectorInGroup(groupIndex);
    float angle = rotation + sectorAngle * 0.5f;
    float2 dir(glm::cos(angle), glm::sin(angle));
    return mDescription.position + dir * (mDescription.minRadius + mDescription.maxRadius) * 0.5f;
}

float Falcor::SelectionWheel::getAngleOfSectorInGroup(uint32_t groupIndex)
{
    checkArgument(groupIndex < (uint32_t)mDescription.sectorGroups.size(), "'groupIndex' ({}) is out of bounds.", groupIndex);
    const uint32_t kSectorCount = mDescription.sectorGroups[groupIndex];
    float groupAngle = getGroupAngle();
    return groupAngle / (float)kSectorCount;
}

float Falcor::SelectionWheel::getRotationOfSector(uint32_t groupIndex, uint32_t sectorIndex)
{
    checkArgument(groupIndex < (uint32_t)mDescription.sectorGroups.size(), "'groupIndex' ({}) is out of bounds.", groupIndex);
    checkArgument(sectorIndex < mDescription.sectorGroups[groupIndex], "'sectorIndex' ({}) is out of bounds.", sectorIndex);
    float groupAngle = getGroupAngle();
    float sectorAngle = getAngleOfSectorInGroup(groupIndex);
    return groupAngle * groupIndex + sectorAngle * sectorIndex;
}

float Falcor::SelectionWheel::getGroupAngle()
{
    const size_t kGroupCount = (uint32_t)mDescription.sectorGroups.size();
    return 2.f * (float)M_PI / (float)kGroupCount;
}

void Falcor::SelectionWheel::computeMouseAngleAndDirLength(const float2& mousePos, float& mouseAngle, float& dirLength)
{
    float2 dir = mousePos - mDescription.position;
    dirLength = glm::length(dir);
    mouseAngle = std::atan2(dir.y, dir.x);
    if (mouseAngle < 0.f)
        mouseAngle = (float)M_PI * 2.f + mouseAngle;
}

void Falcor::SelectionWheel::computeGroupAndSectorIndexFromAngle(float angle, uint32_t& groupIndex, uint32_t& sectorIndex)
{
    float groupAngle = getGroupAngle();
    groupIndex = (uint32_t)std::floor(angle / groupAngle);
    float sectorAngle = getAngleOfSectorInGroup(groupIndex);
    float minRotation = getRotationOfSector(groupIndex, 0);
    sectorIndex = (uint32_t)std::floor((angle - minRotation) / sectorAngle);
}

void Falcor::SelectionWheel::addCircleSector(float rotation, float angle, const float4& color, const float4& borderColor, float margin, bool marginOnBothSides, ExcludeBorderFlags excludeBorderFlags)
{
    constexpr float kStartOffset = (float)M_PI / 2.f;
    float rotaion = kStartOffset - rotation - angle*0.5f - (marginOnBothSides ? 0.f : margin*0.5f);
    mpMarker2DSet->addCircleSector(mDescription.position, rotaion, angle - std::fabs(marginOnBothSides ? 2.f * margin : margin), mDescription.minRadius, mDescription.maxRadius, color, borderColor, excludeBorderFlags);
}
