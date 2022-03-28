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

import Utils.SDF.SDF2DPrimitives;
import Utils.Math.MathHelpers;

/** Compute a 2x3 matrix to be used for scaling, rotating, and translating an SDF primitive. The scale is applied first, then rotation, and finally translation.
    The inverse is returned, which can then be applied directly to the querying point.
    \param[in] scale The scale of the transform.
    \param[in] rotationAngle Rotation angle of the transform.
    \param[in] translation Translation of the transform.
    \return Inverse 2x3 transform matrix.
*/
float2x3 sdfComputeTransform(float2 scale, float rotationAngle, float2 translation)
{
    float ca = cos(rotationAngle);
    float sa = sin(rotationAngle);
    float2x3 M = float2x3(scale.x * ca, -scale.y * sa, translation.x, scale.x * sa, scale.y * ca, translation.y);
    return inverse(M);
}

/** Applies a scale, rotation, and translation to a point. See sdfComputeTransform() above.
    \param[in] p The 2D point to be transformed.
    \param[in] scale The scale of the transform.
    \param[in] rotationAngle Rotation angle of the transform.
    \param[in] translation Translation of the transform.
    \return Transformed point.
*/
float2 sdfTransform(float2 p, float2 scale, float rotationAngle, float2 translation)
{
    float2x3 inverseTransform = sdfComputeTransform(scale, rotationAngle, translation);
    return mul(inverseTransform, float3(p, 1.0f)).xy;
}

/** Filler function that may be used by signed distance functions. Returns alpha so you can blend as you wish.
    Based on the JCGT article "Antialiased 2D Grid, Marker, and Arrow Shaders" by Rougier.
    http://jcgt.org/published/0003/04/01/
    \param[in] signedDistance The signed distance to the primitive.
    \param[in] lineWidth Line width.
    \param[in] antialiasWidth Antialias with.
    \return Alpha.
*/
float sdfFilledAlpha(float signedDistance, float lineWidth, float antialiasWidth)
{
    float borderDistance = abs(signedDistance) - antialiasWidth;
    float alpha = abs(borderDistance / antialiasWidth);    // 0.0 at the outer border and 1.0 close to full fill
    alpha = smoothstep(0.0f, 1.0f, alpha);

    if (signedDistance < 0.0f)
    {
        return 1.0f;
    }
    else if (borderDistance < 0.0f)
    {
        return alpha;
    }
    else
    {
        return 0.0f;
    }
}

/** Filler function that may be used by signed distance functions. Returns color with alpha.
    \param[in] signedDistance The signed distance to the primitive.
    \param[in] lineWidth Line width.
    \param[in] antialiasWidth Antialias with.
    \param[in] fillColor Color of the primitive.
    \return Color.
*/
float4 sdfFilled(float signedDistance, float lineWidth, float antialiasWidth, float4 fillColor)
{
    float alpha = sdfFilledAlpha(signedDistance, lineWidth, antialiasWidth);
    return float4(fillColor.rgb, alpha * fillColor.a);
}

/** Draw function that blends the computed SDF primitive color with the background color. Transforms the primitive prior to evaluation.
    \param[in] sdf The SDF primitive.
    \param[in] p The 2D point where the SDF is to be evaluated..
    \param[in] scale The scale of the transform.
    \param[in] rotationAngle Rotation angle of the transform.
    \param[in] translation Translation of the transform.
    \param[in] primitiveColor The color of the SDF primitive.
    \param[in] backgroundColor Incoming background color.
    \param[in] blendRadius Blend radius for antialiasing.
    \return Color.
*/
float4 sdfDraw<SDF : ISDF2D>(const SDF sdf, float2 p, float2 scale, float rotationAngle, float2 translation,
    float4 primitiveColor, float4 backgroundColor, float blendRadius = 1.f)
{
    p = sdfTransform(p, scale, rotationAngle, translation);
    float signedDistance = sdf.eval(p);
    float4 color = sdfFilled(signedDistance, 0.0f, blendRadius / (0.5f * (scale.x + scale.y)), primitiveColor);
    return lerp(backgroundColor, color, color.a);
}

/** Draw function that blends the computed SDF primitive color with the background color.
    \param[in] sdf The SDF primitive.
    \param[in] p The 2D point where the SDF is to be evaluated..
    \param[in] primitiveColor The color of the SDF primitive.
    \param[in] backgroundColor Incoming background color.
    \param[in] blendRadius Blend radius for antialiasing.
    \return Color.
*/
float4 sdfDraw<SDF : ISDF2D>(const SDF sdf, float2 p, float4 primitiveColor, float4 backgroundColor, float blendRadius = 1.f)
{
    return sdfDraw(sdf, p, float2(1.0f), 0.0f, float2(0.0f), primitiveColor, backgroundColor, blendRadius);
}
