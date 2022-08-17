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
import Scene.Scene;

// More stable way of computing inclination of vector.
float acosStable(float3 w)
{
    return 2.0f * asin(0.5f * sqrt(w.x * w.x + w.y * w.y + (w.z - 1.0f) * (w.z - 1.0f)));
}

float2 toSpherical(float3 w)
{
    return float2(acosStable(w), atan2(w.y, w.x));
}

float3 fromSpherical(float2 s)
{
    return float3(cos(s.y) * sin(s.x), sin(s.y) * sin(s.x), cos(s.x));
}

float2 sphericalToUnit(float2 angles)
{
    return float2(sqrt(angles.x * 2.0f * M_1_PI), (angles.y + M_PI) * 0.5f * M_1_PI);
}

float2 unitToSpherical(float2 uv)
{
    return float2(uv.x * uv.x * 0.5f * M_PI, 2.0f * uv.y * M_PI - M_PI);
}

// Note: This must pack sizes with values up to RGLMaterialData::kMaxResolution.
uint2 packSize(uint4 unpacked)
{
    return uint2(unpacked.x | (unpacked.y << 16), unpacked.z | (unpacked.w << 16));
}

uint4 unpackSize(uint2 packed)
{
    return uint4(packed.x, packed.x >> 16, packed.y, packed.y >> 16) & 0xffff;
}

float loadf(uint buf, int idx)
{
    return asfloat(gScene.materials.getBuffer(buf).Load(idx * 4));
}

float loadf(ByteAddressBuffer buf, int idx)
{
    return asfloat(buf.Load(idx * 4));
}

/** This class represents a coordinate warp defined by 1D tables of (increasing) coordinates xs and ys.

    A trivial warping transforms from [0, 1]^2 to the space defined by xs and ys by linearly interpolating from xs and ys.
    This class implements the inverse of this: Given values interpolated from xs and ys, return the original values in [0, 1]^2.
*/
struct CoordinateWarp
{
    int2 size;
    ByteAddressBuffer xs;
    ByteAddressBuffer ys;

    static CoordinateWarp fromManagedBuffers(int2 size, uint xsBufferID, uint ysBufferID)
    {
        return CoordinateWarp(size, gScene.materials.getBuffer(xsBufferID), gScene.materials.getBuffer(ysBufferID));
    }

    __init(int2 size, ByteAddressBuffer xs, ByteAddressBuffer ys)
    {
        this.size = size;
        this.xs = xs;
        this.ys = ys;
    }

    int bisect(ByteAddressBuffer buf, int size, float x)
    {
        int a = 0, b = size - 1;
        while (b - a > 1)
        {
            int mid = (a + b) / 2;
            float val = loadf(buf, mid);
            if (val <= x)
            {
                a = mid;
            }
            else
            {
                b = mid;
            }
        }
        return a;
    }

    void warp(ByteAddressBuffer buf, int size, float x, out int idx, out float u)
    {
        idx = bisect(buf, size, x);
        if (idx < size - 1)
        {
            float a = loadf(buf, idx + 0);
            float b = loadf(buf, idx + 1);
            u = a == b ? 0.0f : (x - a) / (b - a);
        }
        else
        {
            u = 0.0f;
        }
    }

    float2 warp(float2 coords)
    {
        int ui, vi;
        float u, v;
        warp(xs, size.x, coords.x, ui, u);
        warp(ys, size.y, coords.y, vi, v);

        return float2(ui + u, vi + v);
    }
};

/** This class linearly interpolates floating point data from a 2D array.
    The data is given in the form of a row-major ByteAddressBuffer.
    Texel locations are at grid cell corners, and the input domain is on [0, 1]^2.
*/
struct Brick2D
{
    int2 size;
    ByteAddressBuffer data;

    static Brick2D fromManagedBuffers(int2 size, uint dataBufferID)
    {
        return Brick2D(size, gScene.materials.getBuffer(dataBufferID));
    }

    __init(int2 size, ByteAddressBuffer data)
    {
        this.size = size;
        this.data = data;
    }

    float eval(float2 uv)
    {
        uv *= size - 1;
        int2 pos = clamp(int2(uv), int2(0), size - 2);
        uv = saturate(uv - pos);
        int idx = pos.x + pos.y * size.x;
        float val00 = loadf(data, idx             );
        float val10 = loadf(data, idx + 1         );
        float val01 = loadf(data, idx     + size.x);
        float val11 = loadf(data, idx + 1 + size.x);
        return lerp(lerp(val00, val10, uv.x), lerp(val01, val11, uv.x), uv.y);
    }
};

/** This class linearly interpolates RGB floating point data from a 4D array, given as a ByteAddressBuffer.

    The data layout is slightly unconventional to fit with the format from the RGL dataset.
    The data is arranged in 2D slices of size size.z * size.w (with size.z being the major axis).
    Each slice represents 1 channel, and three adjacent slices form the RGB data.
    There are size.x * size.y * 3 slices, with size.y being the major axis.

    Evaluation is done by specifying the 2D slice in the 4D array (domain [0, size.x] x [0, size.y])
    and the UV coordinates in that slice (domain [0, 1]^2). The resulting data is then lineraly interpolated
    in all 4 dimensions.
*/
struct Brick4D
{
    int4 size;
    ByteAddressBuffer data;

    static Brick4D fromManagedBuffers(int4 size, uint dataBufferID)
    {
        return Brick4D(size, gScene.materials.getBuffer(dataBufferID));
    }

    __init(int4 size, ByteAddressBuffer data)
    {
        this.size = size;
        this.data = data;
    }

    float interpolate(int base, int2 stride, float2 uv)
    {
        float val00 = loadf(data, base                      );
        float val10 = loadf(data, base + stride.x           );
        float val01 = loadf(data, base            + stride.y);
        float val11 = loadf(data, base + stride.x + stride.y);
        return lerp(lerp(val00, val10, uv.x), lerp(val01, val11, uv.x), uv.y);
    }

    float interpolate(int base, int3 stride, float3 uvw)
    {
        return lerp(interpolate(base, stride.xy, uvw.xy), interpolate(base + stride.z, stride.xy, uvw.xy), uvw.z);
    }

    float interpolate(int base, int4 stride, float4 uvwx)
    {
        return lerp(interpolate(base, stride.xyz, uvwx.xyz), interpolate(base + stride.w, stride.xyz, uvwx.xyz), uvwx.w);
    }

    float3 eval(float2 slice, float2 xi)
    {
        float4 uv = float4(slice, xi * (size.zw - 1));
        int4 pos = int4(uv);
        pos.zw = clamp(pos.zw, 0, size.zw - 2);
        uv = saturate(uv - pos);
        int4 stride = int4(size.x == 1 ? 0 : size.y * 3 * size.z * size.w, size.y == 1 ? 0 : 3 * size.z * size.w, 1, size.w);
        int base = dot(stride, pos);

        return float3(
            interpolate(base + size.z * size.w * 0, stride, uv),
            interpolate(base + size.z * size.w * 1, stride, uv),
            interpolate(base + size.z * size.w * 2, stride, uv)
        );
    }
};

/** This class allows sampling a 2D PDF that is interpolated over two separate dimensions.

    pdf is a 4D table: The first two dimensions are interpolated, and the last two dimensions are sampled.
    An illustrative use case is sampling a direction wo given direction wi: The pdf then is a list of
    discrete 2D slices of the PDF for wo, where wi is fixed for each slice.
    We would like to interpolate these slices for any wi.
    Additionally, the PDF for wo is linearly interpolated; this means the CDF is quadratic in wo.

    This class allows for sampling (i.e. warping from [0,1]^2 to the output domain) and also inverting
    the sampling, and computing the PDF of a sample.
*/
struct InterpolatedDistribution2D
{
    int4 size;
    ByteAddressBuffer marginalCdf;
    ByteAddressBuffer conditionalCdf;
    ByteAddressBuffer pdf;

    static InterpolatedDistribution2D fromManagedBuffers(int4 size, uint marginalCdfBufID, uint conditionalCdfBufID, uint pdfBufID)
    {
        return InterpolatedDistribution2D(
            size,
            gScene.materials.getBuffer(marginalCdfBufID),
            gScene.materials.getBuffer(conditionalCdfBufID),
            gScene.materials.getBuffer(pdfBufID)
        );
    }

    __init(int4 size, ByteAddressBuffer marginalCdf, ByteAddressBuffer conditionalCdf, ByteAddressBuffer pdf)
    {
        this.size = size;
        this.marginalCdf = marginalCdf;
        this.conditionalCdf = conditionalCdf;
        this.pdf = pdf;
    }

    float interpolate(ByteAddressBuffer buf, int base, int2 stride, float2 uv)
    {
        float val00 = loadf(buf, base                      );
        float val10 = loadf(buf, base + stride.x           );
        float val01 = loadf(buf, base            + stride.y);
        float val11 = loadf(buf, base + stride.x + stride.y);
        return lerp(lerp(val00, val10, uv.x), lerp(val01, val11, uv.x), uv.y);
    }

    float4 gather(ByteAddressBuffer buf, int base, int4 stride, float2 uv)
    {
        return float4(
            interpolate(buf, base                      , stride.xy, uv),
            interpolate(buf, base            + stride.z, stride.xy, uv),
            interpolate(buf, base + stride.w           , stride.xy, uv),
            interpolate(buf, base + stride.w + stride.z, stride.xy, uv)
        );
    }

    float interpolate(ByteAddressBuffer buf, int base, int3 stride, float3 uvw)
    {
        return lerp(interpolate(buf, base, stride.xy, uvw.xy), interpolate(buf, base + stride.z, stride.xy, uvw.xy), uvw.z);
    }

    float interpolate(ByteAddressBuffer buf, int base, int4 stride, float4 uvwx)
    {
        return lerp(interpolate(buf, base, stride.xyz, uvwx.xyz), interpolate(buf, base + stride.w, stride.xyz, uvwx.xyz), uvwx.w);
    }

    int bisectInterpolated(ByteAddressBuffer buf, int base, int2 stride, int size, float2 uv, float x)
    {
        int a = base, b = base + size - 1;
        while (b - a > 1)
        {
            int mid = (a + b) / 2;
            float val = interpolate(buf, mid, stride, uv);
            if (val <= x)
            {
                a = mid;
            }
            else
            {
                b = mid;
            }
        }
        return a - base;
    }

    int bisectInterpolated(ByteAddressBuffer buf, int base, int3 stride, int size, float3 uvw, float x)
    {
        int a = base, b = base + size - 1;
        while (b - a > 1)
        {
            int mid = (a + b) / 2;
            float val = interpolate(buf, mid, stride, uvw);
            if (val <= x)
            {
                a = mid;
            }
            else
            {
                b = mid;
            }
        }
        return a - base;
    }

    float evalQuadratic(float a, float b, float u)
    {
        return a * u + (b - a) * u * u * 0.5f;
    }

    float solveQuadraticInterpolant(float a, float b, float x)
    {
        if (abs(a - b) < 1e-4f * (a + b))
        {
            return x * 2.0f / (a + b);
        }
        else
        {
            return (-a + sqrt(a * a + 2.0f * (b - a) * x)) / (b - a);
        }
    }

    float3 sample(float2 slice, float2 xi)
    {
        const float kOneMinusEpsilon = 1.0f - 1e-6f;
        xi = clamp(xi, 1.0f - kOneMinusEpsilon, kOneMinusEpsilon);

        int2 sliceI = int2(slice);
        float2 uv = frac(slice);

        int2 strideMarginal = int2(size.x == 1 ? 0 : size.y * size.w, size.y == 1 ? 0 : size.w);
        int3 strideConditional = int3(strideMarginal, 1) * size.z;
        int baseMarginal = dot(sliceI, strideMarginal);
        int row = bisectInterpolated(marginalCdf, baseMarginal, strideMarginal, size.z, uv, xi.y);
        int baseConditional = dot(int3(sliceI, row), strideConditional);

        float rowBase = interpolate(marginalCdf, baseMarginal + row, strideMarginal, uv);
        float rowA = interpolate(conditionalCdf, baseConditional + size.z     - 1, strideConditional.xy, uv);
        float rowB = interpolate(conditionalCdf, baseConditional + size.z * 2 - 1, strideConditional.xy, uv);
        float rowF = solveQuadraticInterpolant(rowA, rowB, xi.y - rowBase);

        xi.x *= lerp(rowA, rowB, rowF);
        int col = bisectInterpolated(conditionalCdf, baseConditional, strideConditional, size.w, float3(uv, rowF), xi.x);
        float colBase = interpolate(conditionalCdf, baseConditional + col, strideConditional, float3(uv, rowF));
        float4 corners = gather(this.pdf, baseConditional + col, int4(strideConditional, 1), uv);
        float colA = lerp(corners.x, corners.y, rowF);
        float colB = lerp(corners.z, corners.w, rowF);
        float colF = solveQuadraticInterpolant(colA, colB, xi.x - colBase);

        float2 warped = float2(col + colF, row + rowF) / (size.zw - 1);
        float pdf = lerp(colA, colB, colF) * (size.z - 1) * (size.w - 1);

        return float3(warped, pdf);
    }

    float3 invert(float2 slice, float2 rc)
    {
        int2 sliceI = int2(slice);
        float2 uv = frac(slice);

        rc *= (size.zw - 1);
        int col = int(rc.x);
        int row = int(rc.y);
        float colF = frac(rc.x);
        float rowF = frac(rc.y);

        int2 strideMarginal = int2(size.x == 1 ? 0 : size.y * size.w, size.y == 1 ? 0 : size.w);
        int3 strideConditional = int3(strideMarginal, 1) * size.z;
        int baseMarginal = dot(sliceI, strideMarginal);
        int baseConditional = dot(int3(sliceI, row), strideConditional);

        float rowBase = interpolate(marginalCdf, baseMarginal + row, strideMarginal, uv);
        float rowA = interpolate(conditionalCdf, baseConditional + size.z     - 1, strideConditional.xy, uv);
        float rowB = interpolate(conditionalCdf, baseConditional + size.z * 2 - 1, strideConditional.xy, uv);
        float xiY = rowBase + evalQuadratic(rowA, rowB, rowF);

        float colBase = interpolate(conditionalCdf, baseConditional + col, strideConditional, float3(uv, rowF));
        float4 corners = gather(this.pdf, baseConditional + col, int4(strideConditional, 1), uv);
        float colA = lerp(corners.x, corners.y, rowF);
        float colB = lerp(corners.z, corners.w, rowF);
        float xiX = (colBase + evalQuadratic(colA, colB, colF)) / lerp(rowA, rowB, rowF);
        float pdf = lerp(colA, colB, colF) * (size.z - 1) * (size.w - 1);;

        return float3(xiX, xiY, pdf);
    }

    float evalPdf(float2 slice, float2 xi)
    {
        float4 uv = float4(slice, xi * (size.zw - 1));
        int4 pos = int4(uv);
        pos.zw = clamp(pos.zw, 0, size.zw - 2);
        uv = saturate(uv - pos);
        int4 stride = int4(size.x == 1 ? 0 : size.y * size.z * size.w, size.y == 1 ? 0 : size.z * size.w, 1, size.w);
        int base = dot(stride, pos);

        return interpolate(pdf, base, stride, uv) * (size.z - 1) * (size.w - 1);
    }
};
