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
#include "Marker2DTypes.slang"

#include <vector>

namespace Falcor
{
    /** Struct holding a set of Marker2DDataBlob. This is a helper struct that has functions to fill the markers depending on the shape type.
    */
    class Marker2DSet
    {
    public:
        using SharedPtr = std::shared_ptr<Marker2DSet>;

        static SharedPtr create(uint32_t maxMarkerCount);

        /** Resets the marker index to the first position. This will allow the next add-calls to add markers from the beginning again.
        */
        void clear();

        /** Add a simple marker with the specified shape, color, and transform.
            \param[in] markerType The type of marker to add.
            \param[in] size The size of the marker.
            \param[in] pos The position of the marker.
            \param[in] rotation The rotation of the marker in radians.
            \param[in] color The color of the marker.
        */
        void addSimpleMarker(const SDF2DShapeType markerType, const float size, const float2& pos, const float rotation, const float4& color);

        /** Add a rouned line as a marker.
            \param[in] posA Start position of the line.
            \param[in] posB End position of the line.
            \param[in] lineWidth The with of the line.
            \param[in] color The color of the line.
        */
        void addRoundedLine(const float2& posA, const float2& posB, const float lineWidth, const float4& color);

        /** Add a triangle as a marker.
            \param[in] posA First corner of the triangle.
            \param[in] posB Second corner of the triangle.
            \param[in] posC Third corner of the triangle.
            \param[in] color The color of the line.
        */
        void addTriangle(const float2& posA, const float2& posB, const float2& posC, const float4& color);

        /** Add a rounded box as a marker.
            \param[in] pos The position of the rounded box.
            \param[in] halfSides Half the length of the rounded box.
            \param[in] radius The radius of the corners.
            \param[in] rotation Rotation in radians.
            \param[in] color The color of the line.
        */
        void addRoundedBox(const float2& pos, const float2& halfSides, const float radius, const float rotation, const float4& color);

        /** Add two markers that have some part of them overlap each other and a operation that should be applied between the two.
            \param[in] op The operation to use.
            \param[in] typeA The shape of the first marker.
            \param[in] posA Position of the first marker.
            \param[in] markerSizeA Size of the first marker.
            \param[in] typeB The shape of the second marker.
            \param[in] posB Position of the second marker.
            \param[in] markerSizeB Size of the second marker.
            \param[in] color Color of the two markers.
            \param[in] dimmedColor The dimmed color of the markers.
        */
        void addMarkerOpMarker(const SDFOperationType op, const SDF2DShapeType typeA, const float2& posA, const float markerSizeA, const SDF2DShapeType typeB, const float2& posB, const float markerSizeB, const float4& color, const float4 dimmedColor);

        /** Add an arrow marker by using two SDF triangles.
            \param[in] startPos Start position of the arrow.
            \param[in] endPos End position of the arrow.
            \param[in] headLength The length of the arrow head.
            \param[in] headWidth The width of the arrow head.
            \param[in] shaftWidth The width of the arrow shaft.
            \param[in] color Color of the arrow.
        */
        void addArrowFromTwoTris(const float2& startPos, const float2& endPos, const float headLength, const float headWidth, const float shaftWidth, const float4& color);

        /** Add a vector represented as an arrow.
            \param[in] posA Start position of the arrow.
            \param[in] posB End position of the arrow.
            \param[in] lineWidth The width of the line.
            \param[in] arrowHeight The height of the arrow head.
            \param[in] color Color of the vector.
        */
        void addVector(const float2& posA, const float2& posB, const float lineWidth, const float arrowHeight, const float4& color);

        /** Add a circle sector. Can also be cut by setting the minRadius to a value larger than zero.
            \param[in] pos The center position of the circle sector as if it was a full circle.
            \param[in] rotation The rotation of the circle sector.
            \param[in] angle The angle of the sircle sector.
            \param[in] minRadius The minimum radius where it si going to cut at. Everything with a lower radius than this will not be seen.
            \param[in] maxRadius The maximum radius of the circle sector.
            \param[in] color Color of the circle sector.
            \param[in] borderColorXYZThicknessW Border color in the x, y, and z components and its thickness in the w component.
            \param[in] excludeBorderFlags Flags for which borders should be excluded from rendering.
        */
        void addCircleSector(const float2& pos, const float rotation, const float angle, const float minRadius, const float maxRadius, const float4& color, const float4& borderColorXYZThicknessW, ExcludeBorderFlags excludeBorderFlags);

        /** Retrive the list of marker objects.
            \return The list of markers.
        */
        const std::vector<Marker2DDataBlob>& getMarkers() const { return mMarkers; }

        /** Get the buffer that holds all markers in this set.
        */
        Buffer::SharedPtr getBuffer() const { return mpMarkerBuffer; }

        /** Set shader data.
        */
        void setShaderData(const ShaderVar& var);

    protected:

        Marker2DSet(uint32_t maxMarkerCount) : mMaxMarkerCount(maxMarkerCount) {}

        /** Adds a Marker2D object to the buffer. Throws a runtime error when marker count exceeds the maximum marker count.
        */
        void addMarker(const Marker2DDataBlob& newMarker);

        /** Update the GPU buffer with marker data if it has changed.
        */
        void updateBuffer();

    private:
        uint32_t                        mMaxMarkerCount;
        std::vector<Marker2DDataBlob>   mMarkers;
        Buffer::SharedPtr               mpMarkerBuffer;
        bool                            mDirtyBuffer = false;
    };
}

