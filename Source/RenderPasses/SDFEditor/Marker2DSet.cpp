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
#include "Marker2DSet.h"
#include "Falcor.h"

void Falcor::Marker2DSet::addMarker(const Marker2DDataBlob& newMarker)
{
    if (mMarkers.size() >= mMaxMarkerCount)
    {
        throw RuntimeError("Number of markers exceeds the maximum number allowed!");
    }

    mMarkers.push_back(newMarker);
    mDirtyBuffer = true;
}

Falcor::Marker2DSet::SharedPtr Falcor::Marker2DSet::create(uint32_t maxMarkerCount)
{
    return SharedPtr(new Marker2DSet(maxMarkerCount));
}

void Falcor::Marker2DSet::clear()
{
    mMarkers.clear();
    mDirtyBuffer = true;
}

void Falcor::Marker2DSet::addSimpleMarker(const SDF2DShapeType markerType, const float size, const float2& pos, const float rotation, const float4& color)
{
    Marker2DDataBlob markerBlob;;
    markerBlob.type = markerType;
    SimpleMarker2DData* pSimpleMarker = reinterpret_cast<SimpleMarker2DData*>(markerBlob.payload.data);
    pSimpleMarker->transform.scale = size;
    pSimpleMarker->transform.rotation = rotation;
    pSimpleMarker->transform.translation = pos;
    pSimpleMarker->color = color;
    addMarker(markerBlob);
}

void Falcor::Marker2DSet::addRoundedLine(const float2& posA, const float2& posB, const float lineWidth, const float4& color)
{
    Marker2DDataBlob markerBlob;
    markerBlob.type = SDF2DShapeType::RoundedLine;
    RoundedLineMarker2DData* pMarker = reinterpret_cast<RoundedLineMarker2DData*>(markerBlob.payload.data);
    pMarker->line.positionA = posA;
    pMarker->line.positionB = posB;
    pMarker->line.width = lineWidth;
    pMarker->color = color;
    addMarker(markerBlob);
}

void Falcor::Marker2DSet::addVector(const float2& posA, const float2& posB, const float lineWidth, const float arrowHeight, const float4& color)
{
    Marker2DDataBlob markerBlob;
    markerBlob.type = SDF2DShapeType::Vector;
    VectorMarker2DData* pMarker = reinterpret_cast<VectorMarker2DData*>(markerBlob.payload.data);
    pMarker->line.positionA = posA;
    pMarker->line.positionB = posB;
    pMarker->line.width = lineWidth;
    pMarker->arrowHeight = arrowHeight;
    pMarker->color = color;
    addMarker(markerBlob);
}

void Falcor::Marker2DSet::addTriangle(const float2& posA, const float2& posB, const float2& posC, const float4& color)
{
    Marker2DDataBlob markerBlob;
    markerBlob.type = SDF2DShapeType::Triangle;
    TriangleMarker2DData* pMarker = reinterpret_cast<TriangleMarker2DData*>((void*)markerBlob.payload.data);
    pMarker->positionA = posA;
    pMarker->positionB = posB;
    pMarker->positionC = posC;
    pMarker->color = color;
    addMarker(markerBlob);
}

void Falcor::Marker2DSet::addRoundedBox(const float2& pos, const float2& halfSides, const float radius, const float rotation, const float4& color)
{
    Marker2DDataBlob markerBlob;
    markerBlob.type = SDF2DShapeType::RoundedBox;
    RoundedBoxMarker2DData* pMarker = reinterpret_cast<RoundedBoxMarker2DData*>(markerBlob.payload.data);
    pMarker->transform.translation = pos;
    pMarker->transform.scale = radius;
    pMarker->transform.rotation = rotation;
    pMarker->halfSides = halfSides;
    pMarker->color = color;
    addMarker(markerBlob);
}

void Falcor::Marker2DSet::addMarkerOpMarker(const SDFOperationType op, const SDF2DShapeType typeA, const float2& posA, const float markerSizeA, const SDF2DShapeType typeB, const float2& posB, const float markerSizeB, const float4& color, const float4 dimmedColor)
{
    Marker2DDataBlob markerBlob;
    markerBlob.type = SDF2DShapeType::MarkerOpMarker;
    MarkerOpMarker2DData* pMarker = reinterpret_cast<MarkerOpMarker2DData*>(markerBlob.payload.data);
    pMarker->operation = op;
    pMarker->markerA.position = posA;
    pMarker->markerA.size = markerSizeA;
    pMarker->markerA.type = typeA;
    pMarker->markerB.position = posB;
    pMarker->markerB.size = markerSizeB;
    pMarker->markerB.type = typeB;
    pMarker->color = color;
    pMarker->dimmedColor = dimmedColor;
    addMarker(markerBlob);
}

void Falcor::Marker2DSet::addArrowFromTwoTris(const float2& startPos, const float2& endPos, const float headLength, const float headWidth, const float shaftWidth, const float4& color)
{
    Marker2DDataBlob markerBlob;
    markerBlob.type = SDF2DShapeType::ArrowFromTwoTris;
    ArrowFromTwoTrisMarker2DData* pMarker = reinterpret_cast<ArrowFromTwoTrisMarker2DData*>(markerBlob.payload.data);
    pMarker->line.positionA = startPos;
    pMarker->line.positionB = endPos;
    pMarker->line.width = shaftWidth;
    pMarker->headLength = headLength;
    pMarker->headWidth = headWidth;
    pMarker->color = color;
    addMarker(markerBlob);
}

void Falcor::Marker2DSet::addCircleSector(const float2& pos, const float rotation, const float angle, const float minRadius, const float maxRadius, const float4& color, const float4& borderColorXYZThicknessW, ExcludeBorderFlags excludeBorderFlags)
{
    Marker2DDataBlob markerBlob;
    markerBlob.type = SDF2DShapeType::CircleSector;
    CircleSectorMarker2DData* pMarker = reinterpret_cast<CircleSectorMarker2DData*>(markerBlob.payload.data);
    pMarker->position = pos;
    pMarker->rotation = rotation;
    pMarker->angle = angle * 0.5f;
    pMarker->maxRadius = maxRadius;
    pMarker->minRadius = minRadius;
    pMarker->color = color;
    pMarker->borderColor = borderColorXYZThicknessW;
    pMarker->excludeBorders = uint32_t(excludeBorderFlags);
    addMarker(markerBlob);
}

void Falcor::Marker2DSet::setShaderData(const ShaderVar& var)
{
    updateBuffer();
    
    var["markers"] = mpMarkerBuffer;
    var["markerCount"] = (uint32_t)mMarkers.size();
}

void Falcor::Marker2DSet::updateBuffer()
{
    if (!mpMarkerBuffer || mDirtyBuffer)
    {
        mDirtyBuffer = false;

        // Invalidate buffer if it is empty.
        if (mMarkers.empty())
        {
            mpMarkerBuffer = nullptr;
        }
        // Create a new buffer if it does not exist or if the size is too small for the markers.
        else if (!mpMarkerBuffer || mpMarkerBuffer->getElementCount() < (uint32_t)mMarkers.size())
        {
            mpMarkerBuffer = Buffer::createStructured(sizeof(Marker2DDataBlob), (uint32_t)mMarkers.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, mMarkers.data(), false);
            mpMarkerBuffer->setName("Marker2DSet::mpMarkerBuffer");
        }
        // Else update the existing buffer.
        else  
        {
            mpMarkerBuffer->setBlob(mMarkers.data(), 0, mMarkers.size() * sizeof(Marker2DDataBlob));
        }
    }
}
