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
#include "Utils/Math/MathConstants.slangh"

import Utils.Math.MathHelpers;

// Some of the code below is based on the JCGT article "Antialiased 2D Grid, Marker, and Arrow Shaders" by Rougier.
// However, the code has been bug fixed in several places and extended in some ways + addition of new primitives.
// http://jcgt.org/published/0003/04/01/

/** Signed distance to unit circle located at the origin.
*/
float sdfCircle(float2 p, float radius)
{
    return length(p) - radius;
}

/** Signed distance to unit square.
*/
float sdfSquare(float2 p)
{
    return max(abs(p.x), abs(p.y)) - 1.0f / 2.0f;
}

/** Signed distance to diamond.
*/
float  sdfDiamond(float2 p)
{
    float x = M_SQRT2 / 2.0f * (p.x - p.y);
    float y = M_SQRT2 / 2.0f * (p.x + p.y);
    return max(abs(x), abs(y)) - 1.0f / (2.0f * M_SQRT2);
}

/** Signed distance to heart.
*/
float sdfHeart(float2 p)
{
    float x = M_SQRT2 / 2.0f * (p.x - p.y);
    float y = M_SQRT2 / 2.0f * (p.x + p.y);
    float r1 = max(abs(x), abs(y)) - 1.0f / 3.5f;
    float r2 = length(p - M_SQRT2 / 2.0f * float2(1.0f, -1.0f) / 3.5f) - 1.0f / 3.5f;
    float r3 = length(p - M_SQRT2 / 2.0f * float2(-1.0f, -1.0f) / 3.5f) - 1.0f / 3.5f;
    return min(min(r1, r2), r3);
}

/** Signed distance to chevron.
*/
float sdfChevron(float2 p)
{
    float x = 1.0f / M_SQRT2 * (p.x - p.y);
    float y = 1.0f / M_SQRT2 * (p.x + p.y);
    float r1 = max(abs(x), abs(y)) - 1.0f / 3.0f;
    float r2 = max(abs(x - 1.0f / 3.0f), abs(y - 1.0f / 3.0f)) - 1.0f / 3.0f;
    return max(r1, -r2);
}

/** Signed distance to ring.
*/
float sdfRing(float2 p)
{
    float r1 = length(p) - 1.0f / 2.0f;
    float r2 = length(p) - 1.0f / 4.0f;
    return max(r1, -r2);
}

/** Signed distance to tag.
*/
float sdfTag(float2 p)
{
    float r1 = max(abs(p.x) - 1.0f / 2.0f, abs(p.y) - 1.0f / 6.0f);
    float r2 = abs(p.x - 1.0f / 1.5f) + abs(p.y) - 1.0f;
    return max(r1, 0.75f * r2);
}

/** Signed distance to cross.
*/
float sdfCross(float2 p)
{
    float x = M_SQRT2 / 2.0f * (p.x - p.y);
    float y = M_SQRT2 / 2.0f * (p.x + p.y);
    float r1 = max(abs(x - 1.0f / 3.0f), abs(x + 1.0f / 3.0f));
    float r2 = max(abs(y - 1.0f / 3.0f), abs(y + 1.0f / 3.0f));
    float r3 = max(abs(x), abs(y));
    return max(min(r1, r2), r3) - 1.0f / 2.0f;
}

/** Signed distance to asterisk.
*/
float sdfAsterisk(float2 p)
{
    float x = M_SQRT2 / 2.0f * (p.x - p.y);
    float y = M_SQRT2 / 2.0f * (p.x + p.y);
    float r1 = max(abs(x) - 1.0f / 2.0f, abs(y) - 1.0f / 10.0f);
    float r2 = max(abs(y) - 1.0f / 2.0f, abs(x) - 1.0f / 10.0f);
    float r3 = max(abs(p.x) - 1.0f / 2.0f, abs(p.y) - 1.0f / 10.0f);
    float r4 = max(abs(p.y) - 1.0f / 2.0f, abs(p.x) - 1.0f / 10.0f);
    return min(min(r1, r2), min(r3, r4));
}

/** Signed distance to infinity.
*/
float sdfInfinity(float2 p)
{
    const float2 c1 = float2(0.2125f, 0.0f);
    const float2 c2 = float2(-0.2125f, 0.0f);
    float r1 = length(p - c1) - 1.0f / 3.5f;
    float r2 = length(p - c1) - 1.0f / 7.5f;
    float r3 = length(p - c2) - 1.0f / 3.5f;
    float r4 = length(p - c2) - 1.0f / 7.5f;
    return min(max(r1, -r2), max(r3, -r4));
}

/** Signed distance to pin.
*/
float sdfPin(float2 p)
{
    float2 c1 = float2(0.0f, -0.15f);
    float r1 = length(p - c1) - 1.0f / 2.675f;
    float2 c2 = float2(1.49f, -0.8f);
    float r2 = length(p - c2) - 2.0f;
    float2 c3 = float2(-1.49f, -0.8f);
    float r3 = length(p - c3) - 2.0f;
    float r4 = length(p - c1) - 1.0f / 5.0f;
    return max(min(r1, max(max(r2, r3), -p.y)), -r4);
}

/** Signed distance to arrow.
*/
float sdfArrow(float2 p)
{
    float r1 = abs(p.x) + abs(p.y) - 1.0f / 2.0f;
    float r2 = max(abs(p.x + 1.0f / 2.0f), abs(p.y)) - 1.0f / 2.0f;
    float r3 = max(abs(p.x - 1.0f / 6.0f) - 1.0f / 4.0f, abs(p.y) - 1.0f / 4.0f);
    return min(r3, max(0.75f * r1, r2));
}

/** Signed distance to rounded box.
*/
float sdfRoundedBox(float2 p, float2 halfBoxSides, float radius)
{
    halfBoxSides -= float2(radius);
    float2 d = abs(p) - halfBoxSides;
    float outsideDist = length(float2(max(d.x, 0.0f), max(d.y, 0.0f))) - radius;
    float insideDist = min(max(d.x, d.y), 0.0f);           // Smallest negative distance clamed to 0.0.
    return outsideDist + insideDist;
}

/** Compute the signed distance from an arbitrary triangle.
*/
float sdfTriangle(float2 p, float2 p0, float2 p1, float2 p2)   // Expects counterclockwise order of vertices.
{
    // Edge from p0 to p1.
    float2 N = normalize(p1 - p0);
    N = float2(-N.y, N.x);
    float distance = dot(N, p - p0);

    // Edge from p1 to p2.
    N = normalize(p2 - p1);
    N = float2(-N.y, N.x);
    distance = max(distance, dot(N, p - p1));

    // Edge from p2 to p0.
    N = normalize(p0 - p2);
    N = float2(-N.y, N.x);
    distance = max(distance, dot(N, p - p2));

    return distance;
}

/** Computes the signed distance from a line segment.
*/
float sdfSegmentDistance(float2 p, float2 p1, float2 p2)
{
    float2 center = (p1 + p2) * 0.5f;
    float len = length(p2 - p1);
    float2 dir = (p2 - p1) / len;
    float2 rel_p = p - center;
    float dist1 = abs(dot(rel_p, float2(dir.y, -dir.x)));
    float dist2 = abs(dot(rel_p, dir)) - 0.5f * len;
    return max(dist1, dist2);
}

/** Computes the signed distance from rounded line segment.
*/
float sdfRoundedLine(float2 p, float2 p1, float2 p2, float width)
{
    float2 dir = normalize(p2 - p1) * width * 0.5f;
    float d1 = sdfSegmentDistance(p, p1 + dir, p2 - dir) - width * 0.5f;
    float d2 = length(p - p1) - width * 0.5f;
    float d3 = length(p - p2) - width * 0.5f;
    return min(min(d1, d2), d3);
}

/** An arrow than can be used to render (normal) vectors, for example.
*/
float sdfVector(float2 p, float2 p1, float2 p2, float width, float arrowHeight)
{
    float2 dir = p2 - p1;
    float len = length(dir);
    float factor = (len - arrowHeight) / len;
    float2 p3 = p1 + dir * factor;
    float d1 = sdfRoundedLine(p, p1, p3, width);
    float2 ortho = normalize(float2(dir.y, -dir.x));
    float d2 = sdfTriangle(p, p2, p3 + ortho * arrowHeight * 0.3f, p3 - ortho * arrowHeight * 0.3f);
    return min(d1, d2);
}

/** Interface for 2D SDF primitives
*/
interface ISDF2D
{
    float eval(float2 p);
};

struct SDF2DCircle : ISDF2D
{
    float radius;
    __init(float radius = 1.0f) { this.radius = radius; }
    float eval(float2 p) { return sdfCircle(p, radius); }
};

struct SDF2DSquare : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfSquare(p); }
};

struct SDF2DDiamond : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfDiamond(p); }
};

struct SDF2DHeart : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfHeart(p); }
};

struct SDF2DChevron : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfChevron(p); }
};

struct SDF2DRing : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfRing(p); }
};

struct SDF2DTag : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfTag(p); }
};

struct SDF2DCross : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfCross(p); }
};

struct SDF2DAsterisk : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfAsterisk(p); }
};

struct SDF2DInfinity : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfInfinity(p); }
};

struct SDF2DPin : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfPin(p); }
};

struct SDF2DArrow : ISDF2D
{
    __init() { }
    float eval(float2 p) { return sdfArrow(p); }
};

struct SDF2DRoundedBox : ISDF2D
{
    float2 halfSides;
    float radius;
    __init(float2 halfSides, float radius)
    {
        this.halfSides = halfSides;
        this.radius = radius;
    }
    float eval(float2 p) { return sdfRoundedBox(p, halfSides, radius); }
};

struct SDF2DVector : ISDF2D
{
    float2 startPoint;
    float2 endPoint;    // Where the arrow is located.
    float width;
    float arrowHeight;
    __init(float2 startPoint, float2 endPoint, float width, float arrowHeight)
    {
        this.startPoint = startPoint;
        this.endPoint = endPoint;
        this.width = width;
        this.arrowHeight = arrowHeight;
    }
    float eval(float2 p) { return sdfVector(p, startPoint, endPoint, width, arrowHeight); }
};

struct SDF2DRoundedLine : ISDF2D
{
    float2 startPoint;
    float2 endPoint;
    float width;
    __init(float2 startPoint, float2 endPoint, float width)
    {
        this.startPoint = startPoint;
        this.endPoint = endPoint;
        this.width = width;
    }
    float eval(float2 p) { return sdfRoundedLine(p, startPoint, endPoint, width); }
};

struct SDF2DTriangle : ISDF2D
{
    float2 p0, p1, p2;
    __init(float2 p0, float2 p1, float2 p2)
    {
        this.p0 = p0;
        this.p1 = p1;
        this.p2 = p2;
    }
    float eval(float2 p) { return sdfTriangle(p, p0, p1, p2); }
};
