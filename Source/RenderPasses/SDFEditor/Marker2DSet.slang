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

/** Structs for markers (2D GUI) for the SDF editor.
*/
__exported import Marker2DTypes;
__exported import Utils.SDF.SDF2DDraw;
__exported import Utils.SDF.SDF2DPrimitives;
__exported import Utils.SDF.SDFOperations;
import Utils.Math.MathHelpers;

/** Struct for a 2D marker that is being set on the CPU side, and sent to the shader for rendering.
    It works by holding a blob of data and a type. The blob of data can be set manually or cast to a specific marker type.
*/
extension Marker2DDataBlob
{
    /** Internal helper function to calculate the signed distance from a basic marker object and also the blend factor.
        \param[in] pixelPos The pixel position.
        \param[in] marker The basic marker object.
        \param[in,out] smoothFactor The smoothing factor used to calculate the blend factor.
        \param[out] distance The signed distance to the marker.
        \param[out] blendFactor The blend radius factor.
    */
    void calcDistanceandAndBlendFactorFromBasicMarker(const uint2 pixelPos, BasicMarker2D marker, inout float smoothFactor, out float distance, out float blendRadiusFactor)
    {
        float2 newPos;
        switch (marker.type)
        {
        case SDF2DShapeType::Circle:
            blendRadiusFactor = 1.0f;
            newPos = sdfTransform(float2(pixelPos), float2(1.0f), 0.0f, marker.position);
            distance = sdfCircle(newPos, marker.size * 0.5f);
            break;
        case SDF2DShapeType::Square:
            blendRadiusFactor = 1.0f / smoothFactor;
            smoothFactor = 1.0f;
            newPos = sdfTransform(float2(pixelPos), float2(marker.size), 0.0f, marker.position);
            distance = sdfSquare(newPos);
            break;
        case SDF2DShapeType::RoundedBox:
            blendRadiusFactor = 1.0f;
            newPos = sdfTransform(float2(pixelPos), float2(1.0f), 0.0f, marker.position);
            distance = sdfRoundedBox(newPos, float2(marker.size) * 0.5f, marker.size * 0.1f);
            break;
        default:
            break;
        }
    }

    /** Render this marker twice so a part of the markers overlap. The interaction is decided by the operation type passed as the first float in the list.
        \param[in] pixelPos The pixel position.
        \param[in] inColor  The color of the shapes.
        \param[in] blendRadius The antialising width.
        \return The color at the pixel position.
    */
    float4 renderMarkerOpMarker(const uint2 pixelPos, const float4 inColor, const float blendRadius = 1.0f)
    {
        MarkerOpMarker2DData marker = reinterpret<MarkerOpMarker2DData, Marker2DPayload>(this.payload);

        float distanceA = 0.0f;
        float distanceB = 0.0f;
        float blendRadiusFactor = 0.0f;
        float smoothFactor = 1.0f;
        calcDistanceandAndBlendFactorFromBasicMarker(pixelPos, marker.markerA, smoothFactor, distanceA, blendRadiusFactor);
        smoothFactor = (marker.markerA.size + marker.markerB.size) * 0.5f;
        calcDistanceandAndBlendFactorFromBasicMarker(pixelPos, marker.markerB, smoothFactor, distanceB, blendRadiusFactor);

        // Draw union of markers first, and then the operation on top.
        float distanceUnion = sdfUnion(distanceA, distanceB);
        float4 newColor = sdfFilled(distanceUnion, 0.0f,  blendRadius * blendRadiusFactor, marker.dimmedColor);
        float4 color = (1.0f - newColor.a) * inColor + newColor.a * newColor;

        // Draw the operation.
        float distance = distanceUnion;
        switch (marker.operation)
        {
        case SDFOperationType::Intersection:
            distance = sdfIntersection(distanceA, distanceB);
            break;
        case SDFOperationType::Subtraction:
            distance = sdfSubtraction(distanceA, distanceB);
            break;
        case SDFOperationType::SmoothUnion:
            distance = sdfSmoothUnion(distanceA, distanceB, smoothFactor * 0.6f);
            break;
        case SDFOperationType::SmoothSubtraction:
            distance = sdfSmoothSubtraction(distanceA, distanceB, smoothFactor * 0.4f);
            break;
        default:
            break;
        }
        newColor = sdfFilled(distance, 0.0f, blendRadius * blendRadiusFactor, marker.color);
        return  (1.0f - newColor.a) * color + newColor.a * newColor;
    }

    /** Render an arrow by using two SDF triangles.
        \param[in] pixelPos The pixel position.
        \param[in] inColor  The color of the shapes.
        \param[in] blendRadius The antialising width.
        \return The color at the pixel position.
    */
    float4 renderArrowFromTwoTris(const uint2 pixelPos, const float4 inColor, const float blendRadius = 1.0f)
    {
        ArrowFromTwoTrisMarker2DData marker = reinterpret<ArrowFromTwoTrisMarker2DData, Marker2DPayload>(this.payload);

        float2 dir = marker.line.positionB - marker.line.positionA;
        float len = length(dir);
        float2 dirN = dir * (1.0 / len);                // Normalized.
        float2 dirNOrtho = float2(-dirN.y, dirN.x);

        float2 shaftMeetsHeadPoint = marker.line.positionB - dirN * marker.headLength;

        // Get distance from shaft triangle.
        float2 shaftRightPoint = shaftMeetsHeadPoint + marker.line.width * 0.5 * dirNOrtho;
        float2 shaftLeftPoint = shaftMeetsHeadPoint - marker.line.width * 0.5 * dirNOrtho;
        float distanceToShaft = sdfTriangle(pixelPos, marker.line.positionA, shaftRightPoint, shaftLeftPoint);

        // Get distance from the head trinagle.
        float2 headRightPoint = shaftMeetsHeadPoint + marker.headWidth * 0.5 * dirNOrtho;
        float2 headLeftPoint = shaftMeetsHeadPoint - marker.headWidth * 0.5 * dirNOrtho;
        float distanceToHead = sdfTriangle(pixelPos, headLeftPoint, headRightPoint, marker.line.positionB);

        // Combine the triangles, fill them, and return the color at the pixel position.
        float distance = sdfUnion(distanceToHead, distanceToShaft);
        float4 newcolor = sdfFilled(distance, 0.0f, blendRadius, marker.color);
        return (1 - newcolor.a) * inColor + newcolor.a * newcolor;
    }

    /** Render a circle sector.
        \param[in] pixelPos The pixel position.
        \param[in] inColor  The color of the shapes.
        \param[in] blendRadius The antialising width.
        \return The color at the pixel position.
    */
    float4 renderCircleSector(const uint2 pixelPos, const float4 inColor, const float blendRadius = 1.0f)
    {
        CircleSectorMarker2DData marker = reinterpret<CircleSectorMarker2DData, Marker2DPayload>(this.payload);

        // Inspiered from sdf functions for pie and line segment of https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
        float2 newPos = float2(pixelPos);
        newPos = sdfTransform(newPos, float2(1.0f), marker.rotation, marker.position);
        float2 oldPos = newPos;
        newPos.x = abs(newPos.x);
        float2 c = float2(sin(marker.angle), cos(marker.angle));
        float circleMin = marker.minRadius - length(newPos);
        float circleMax = length(newPos) - marker.maxRadius;
        float m = length(newPos - c*clamp(dot(c,newPos), marker.minRadius, marker.maxRadius));
        float s = sign(c.y * newPos.x - c.x * newPos.y);
        float distance = max(m*s, max(circleMin, circleMax));
        float4 newColor = sdfFilled(distance, 0.0f, blendRadius, marker.color);
        distance = marker.borderColor.w+10.0f;

        // Calc distance to borders
        float4 borderColor = float4(marker.borderColor.xyz, newColor.a);

        // Top
        if ((marker.excludeBorders & uint(ExcludeBorderFlags::Top)) == 0)
        {
            float m = length(newPos - c*marker.maxRadius);
            distance = max(abs(circleMax), m*s);
        }

        // Bottom
        if ((marker.excludeBorders & uint(ExcludeBorderFlags::Bottom)) == 0)
        {
            float m = length(newPos - c*marker.minRadius);
            distance = min(distance, max(abs(circleMin), m*s));
        }

        // Sides
        bool isLeft = (marker.excludeBorders & uint(ExcludeBorderFlags::Left)) == 0;
        bool isRight = (marker.excludeBorders & uint(ExcludeBorderFlags::Right)) == 0;
        if (isLeft || isRight)
        {
            if (!isLeft) oldPos.x = -oldPos.x;
            if (isLeft && isRight) oldPos.x = abs(oldPos.x);
            float2 a = c*marker.minRadius;
            float2 b = c*marker.maxRadius;
            float2 pa = oldPos - a, ba = b - a;
            float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
            distance = min(distance, length(pa - h*ba));
        }

        // Apply border color
        newColor = lerp(newColor, borderColor, smoothstep(marker.borderColor.w, marker.borderColor.w-2.0f, abs(distance)));

        return lerp(inColor, newColor, newColor.a);
    }

    /** Internal helper function to render a simple marker2D object.
    */
    float4 sdfDrawSimpleMarker<SDF : ISDF2D>(const SDF sdf, const uint2 pixelPos, const SimpleMarker2DData marker, const float4 backgroundColor, const float blendRadius)
    {
        return sdfDraw(sdf, pixelPos, float2(marker.transform.scale), marker.transform.rotation, marker.transform.translation, marker.color, backgroundColor, blendRadius);
    }

    /** Render the marker.
        \param[in] pixelPos The pixel position.
        \param[in] inColor  The color of the shapes.
        \param[in] blendRadius The antialising width.
        \return The color at the pixel position.
    */
    float4 render(const uint2 pixelPos, const float4 inColor, const float blendRadius = 1.0f)
    {

        float4 finalColor = float4(0.f, 0.f, 0.f, 1.f);
        if (uint(this.type) >= uint(SDF2DShapeType::Circle) && uint(this.type) <= uint(SDF2DShapeType::Arrow))     // The simple markers which share a lot.
        {
            SimpleMarker2DData marker = reinterpret<SimpleMarker2DData, Marker2DPayload>(this.payload);

            // Get the color at the pixel position depending on the shape type.
            switch (this.type)
            {
            case SDF2DShapeType::Circle:
                finalColor = sdfDraw(SDF2DCircle(marker.transform.scale), pixelPos, float2(1.0), marker.transform.rotation, marker.transform.translation, marker.color, inColor, blendRadius);
                break;
            case SDF2DShapeType::Square:
                finalColor = sdfDrawSimpleMarker(SDF2DSquare(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Diamond:
                finalColor = sdfDrawSimpleMarker(SDF2DDiamond(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Heart:
                finalColor = sdfDrawSimpleMarker(SDF2DHeart(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Chevron:
                finalColor = sdfDrawSimpleMarker(SDF2DChevron(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Ring:
                finalColor = sdfDrawSimpleMarker(SDF2DRing(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Tag:
                finalColor = sdfDrawSimpleMarker(SDF2DTag(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Cross:
                finalColor = sdfDrawSimpleMarker(SDF2DCross(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Asterisk:
                finalColor = sdfDrawSimpleMarker(SDF2DAsterisk(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Infinity:
                finalColor = sdfDrawSimpleMarker(SDF2DInfinity(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Pin:
                finalColor = sdfDrawSimpleMarker(SDF2DPin(), pixelPos, marker, inColor, blendRadius);
                break;
            case SDF2DShapeType::Arrow:
                finalColor = sdfDrawSimpleMarker(SDF2DArrow(), pixelPos, marker, inColor, blendRadius);
                break;
            default:
                break;
            }
        }
        else if (this.type == SDF2DShapeType::RoundedBox)
        {
            RoundedBoxMarker2DData marker = reinterpret<RoundedBoxMarker2DData, Marker2DPayload>(this.payload);
            finalColor = sdfDraw(SDF2DRoundedBox(marker.halfSides, marker.transform.scale), pixelPos, float2(1.0), marker.transform.rotation, marker.transform.translation, marker.color, inColor, blendRadius);
        }
        else if (this.type == SDF2DShapeType::Triangle)
        {
            TriangleMarker2DData marker = reinterpret<TriangleMarker2DData, Marker2DPayload>(this.payload);
            finalColor = sdfDraw(SDF2DTriangle(marker.positionA, marker.positionB, marker.positionC), pixelPos, marker.color, inColor, blendRadius);
        }
        else if (this.type == SDF2DShapeType::RoundedLine)
        {
            RoundedLineMarker2DData marker = reinterpret<RoundedLineMarker2DData, Marker2DPayload>(this.payload);
            finalColor = sdfDraw(SDF2DRoundedLine(marker.line.positionA, marker.line.positionB, marker.line.width), pixelPos, marker.color, inColor, blendRadius);
        }
        else if (this.type == SDF2DShapeType::Vector)
        {
            VectorMarker2DData marker = reinterpret<VectorMarker2DData, Marker2DPayload>(this.payload);
            finalColor = sdfDraw(SDF2DVector(marker.line.positionA, marker.line.positionB, marker.line.width, marker.arrowHeight), pixelPos, marker.color, inColor, blendRadius);
        }
        else if (this.type == SDF2DShapeType::MarkerOpMarker)
        {
            finalColor = renderMarkerOpMarker(pixelPos, inColor, blendRadius);
        }
        else if (this.type == SDF2DShapeType::ArrowFromTwoTris)
        {
            finalColor = renderArrowFromTwoTris(pixelPos, inColor, blendRadius);
        }
        else if (this.type == SDF2DShapeType::CircleSector)
        {
            finalColor = renderCircleSector(pixelPos, inColor, blendRadius);
        }
        
        return finalColor;
    }
};

struct Marker2DSet
{
    StructuredBuffer<Marker2DDataBlob> markers;
    uint markerCount;

    float4 render(uint2 pixel, float4 inColor)
    {
        float4 color = inColor;
        for (uint index = 0; index < markerCount; index++)
        {
            color = markers[index].render(pixel, color, 1.0f);
        }
        return color;
    }
};
