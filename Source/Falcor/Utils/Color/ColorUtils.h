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
#include <glm/gtx/matrix_operation.hpp>

/** Color conversion utility functions.

    Falcor currently assumes all input/outputs are in sRGB, which uses the
    ITU-R Rec. BT.709 (Rec.709) color space.
    We have conversion functions to/from CIE XYZ to do certain operations like
    white point correction, color temperature conversion etc.

    Matlab matrices for convenience below (row major):

    RGB Rec.709 to CIE XYZ (derived from primaries and D65 whitepoint):

        M = [ 0.4123907992659595   0.3575843393838780   0.1804807884018343;
              0.2126390058715104   0.7151686787677559   0.0721923153607337;
              0.0193308187155918   0.1191947797946259   0.9505321522496608 ]

    CIE XYZ to LMS using the CAT02 transform (part of CIECAM02):

        M = [ 0.7328   0.4296  -0.1624;
             -0.7036   1.6975   0.0061;
              0.0030   0.0136   0.9834 ]

    CIE XYZ to LMS using the Bradford transform (part of the original CIECAM97 model):

        M = [ 0.8951   0.2664  -0.1614;
             -0.7502   1.7135   0.0367;
              0.0389  -0.0685   1.0296 ]

    Note: glm is column major, so the pre-defined matrices below are transposed.

*/

namespace Falcor
{
    // Transform from RGB color in Rec.709 to CIE XYZ.
    static const glm::float3x3 kColorTransform_RGBtoXYZ_Rec709 =
    {
        0.4123907992659595, 0.2126390058715104, 0.0193308187155918,
        0.3575843393838780, 0.7151686787677559, 0.1191947797946259,
        0.1804807884018343, 0.0721923153607337, 0.9505321522496608
    };

    // Transform from XYZ color to RGB in Rec.709.
    static const glm::float3x3 kColorTransform_XYZtoRGB_Rec709 =
    {
        3.2409699419045213, -0.9692436362808798, 0.0556300796969936,
        -1.5373831775700935, 1.8759675015077206, -0.2039769588889765,
        -0.4986107602930033, 0.0415550574071756, 1.0569715142428784
    };

    // Transform from CIE XYZ to LMS using the CAT02 transform.
    static const glm::float3x3 kColorTransform_XYZtoLMS_CAT02 =
    {
        0.7328, -0.7036, 0.0030,
        0.4296, 1.6975, 0.0136,
        -0.1624, 0.0061, 0.9834
    };

    // Transform from LMS to CIE XYZ using the inverse CAT02 transform.
    static const glm::float3x3 kColorTransform_LMStoXYZ_CAT02 =
    {
        1.096123820835514, 0.454369041975359, -0.009627608738429,
        -0.278869000218287, 0.473533154307412, -0.005698031216113,
        0.182745179382773, 0.072097803717229, 1.015325639954543
    };

    // Transform from CIE XYZ to LMS using the Bradford transform.
    static const glm::float3x3 kColorTransform_XYZtoLMS_Bradford =
    {
        0.8951, -0.7502, 0.0389,
        0.2664, 1.7135, -0.0685,
        -0.1614, 0.0367, 1.0296
    };

    // Transform from LMS to CIE XYZ using the inverse Bradford transform.
    static const glm::float3x3 kColorTransform_LMStoXYZ_Bradford =
    {
        0.98699290546671214, 0.43230526972339445, -0.00852866457517732,
        -0.14705425642099013, 0.51836027153677744, 0.04004282165408486,
        0.15996265166373122, 0.04929122821285559, 0.96848669578754998
    };

    /** Transforms an RGB color in Rec.709 to CIE XYZ.
    */
    static float3 RGBtoXYZ_Rec709(float3 c)
    {
        return kColorTransform_RGBtoXYZ_Rec709 * c;
    }

    /** Transforms an XYZ color to RGB in Rec.709.
    */
    static float3 XYZtoRGB_Rec709(float3 c)
    {
        return kColorTransform_XYZtoRGB_Rec709 * c;
    }

    /** Converts (chromaticities, luminance) to XYZ color.
    */
    static float3 xyYtoXYZ(float x, float y, float Y)
    {
        return float3(x * Y / y, Y, (1.f - x - y) * Y / y);
    }

    /** Transforms color temperature of a blackbody emitter to color in CIE XYZ.
        This function uses an approximation based on piecewise rational polynomials:
        Kang et al., Design of Advanced Color Temperature Control System for HDTV Applications, 2002.
        https://pdfs.semanticscholar.org/cc7f/c2e67601ccb1a8fec048c9b78a4224c34d26.pdf

        \param[in] T Color temperature in degrees Kelvin, supported range is 1667K to 25000K.
        \param[in] Y Luminance.
        \return CIE XYZ color.
    */
    static float3 colorTemperatureToXYZ(float T, float Y = 1.f)
    {
        if (T < 1667.f || T > 25000.f)
        {
            logError("colorTemperatureToXYZ() - T is out of range");
            return float3(0, 0, 0);
        }

        // We do the computations in double
        double t = T;
        double t2 = t * t;
        double t3 = t * t * t;

        double xc = 0.0;
        if (T < 4000.f)
        {
            xc = -0.2661239e9 / t3 - 0.2343580e6 / t2 + 0.8776956e3 / t + 0.179910;
        }
        else
        {
            xc = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390;
        }

        double x = xc;
        double x2 = x * x;
        double x3 = x * x * x;

        double yc = 0.0;
        if (T < 2222.f)
        {
            yc = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683;
        }
        else if (T < 4000.f)
        {
            yc = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
        }
        else
        {
            yc = +3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
        }

        // Return as XYZ color.
        return xyYtoXYZ((float)xc, (float)yc, Y);
    }

    /** Calculates the 3x3 matrix that performs white balancing in RGB Rec.709 space
        to a target color temperature.

        The function uses the von Kries transform, i.e. a diagonal scaling matrix in LMS space.
        The default LMS transform is CAT02 (part of CIECAM02).

        The transform is chosen so that the D65 white point is exactly preserved at T=6500K.
        Note that the transformed RGB can be out-of-gamut in Rec.709 (negative values
        are possible) depending on T, so it is advisable to gamut clamp the result.

        \param[in] T Target color temperature (K).
        \return 3x3 matrix M, which transforms linear RGB in Rec.709 using c' = M * c.
    */
    static glm::float3x3 calculateWhiteBalanceTransformRGB_Rec709(float T)
    {
        static const glm::float3x3 MA = kColorTransform_XYZtoLMS_CAT02 * kColorTransform_RGBtoXYZ_Rec709;    // RGB -> LMS
        static const glm::float3x3 invMA = kColorTransform_XYZtoRGB_Rec709 * kColorTransform_LMStoXYZ_CAT02; // LMS -> RGB

        // Compute destination reference white in LMS space.
        static const float3 wd = kColorTransform_XYZtoLMS_CAT02 * colorTemperatureToXYZ(6500.f);

        // Compute source reference white in LMS space.
        const float3 ws = kColorTransform_XYZtoLMS_CAT02 * colorTemperatureToXYZ(T);

        // Derive final 3x3 transform in RGB space.
        float3 scale = wd / ws;
        glm::float3x3 D = glm::diagonal3x3(scale);

        return invMA * D * MA;
    }
}
