/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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

/** FLIP GitHub repository,
    by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
    Pointer to the repository: https://github.com/NVlabs/flip.

    Visualizing and Communicating Errors in Rendered Images
    Ray Tracing Gems II, 2021,
    by Pontus Andersson, Jim Nilsson, and Tomas Akenine-Moller.
    Pointer to the chapter: https://research.nvidia.com/publication/2021-08_Visualizing-and-Communicating.

    Visualizing Errors in Rendered High Dynamic Range Images
    Eurographics 2021,
    by Pontus Andersson, Jim Nilsson, Peter Shirley, and Tomas Akenine-Moller.
    Pointer to the paper: https://research.nvidia.com/publication/2021-05_HDR-FLIP.

    FLIP: A Difference Evaluator for Alternating Images
    High Performance Graphics 2020,
    by Pontus Andersson, Jim Nilsson, Tomas Akenine-Moller,
    Magnus Oskarsson, Kalle Astrom, and Mark D. Fairchild.
    Pointer to the paper: https://research.nvidia.com/publication/2020-07_FLIP.
*/

#ifndef FLIP_SHADER_COMMON_HLSLI
#define FLIP_SHADER_COMMON_HLSLI


//  support functions

static const float3 D65ReferenceIlluminant = float3(0.950428545, 1.000000000, 1.088900371);
static const float3 InvD65ReferenceIlluminant = float3(1.052156925, 1.000000000, 0.918357670);

static const float Pi = 3.141592653;
static const float PiSquared = Pi * Pi;
static const float InvSqrt2 = 1.0 / sqrt(2.0);

float sRGB2LinearRGB(float sRGBColor)
{
    if (sRGBColor <= 0.04045)
        return sRGBColor / 12.92;
    else
        return pow((sRGBColor + 0.055) / 1.055, 2.4);
}

float3 sRGB2LinearRGB(float3 sRGBColor)
{
    return float3(sRGB2LinearRGB(sRGBColor.r), sRGB2LinearRGB(sRGBColor.g), sRGB2LinearRGB(sRGBColor.b));
}

float linear2sRGB(float linearColor)
{
    if (linearColor <= 0.0031308)
        return linearColor * 12.92;
    else
        return 1.055 * pow(linearColor, 1.0 / 2.4) - 0.055;
}

float3 linearRGB2sRGB(float3 linearColor)
{
    return float3(linear2sRGB(linearColor.r), linear2sRGB(linearColor.g), linear2sRGB(linearColor.b));
}

float3 linearRGB2XYZ(float3 linColor)
{
    // Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
    // Assumes D65 standard illuminant
    const float a11 = 10135552.0 / 24577794.0;
    const float a12 = 8788810.0 / 24577794.0;
    const float a13 = 4435075.0 / 24577794.0;
    const float a21 = 2613072.0 / 12288897.0;
    const float a22 = 8788810.0 / 12288897.0;
    const float a23 = 887015.0 / 12288897.0;
    const float a31 = 1425312.0 / 73733382.0;
    const float a32 = 8788810.0 / 73733382.0;
    const float a33 = 70074185.0 / 73733382.0;

    float3 xyzColor;
    xyzColor.r = a11 * linColor.r + a12 * linColor.g + a13 * linColor.b;
    xyzColor.g = a21 * linColor.r + a22 * linColor.g + a23 * linColor.b;
    xyzColor.b = a31 * linColor.r + a32 * linColor.g + a33 * linColor.b;

    return xyzColor;
}

float3 XYZ2LinearRGB(float3 xyzColor)
{
    // Return values in linear RGB, assuming D65 standard illuminant
    const float a11 = 3.241003275;
    const float a12 = -1.537398934;
    const float a13 = -0.498615861;
    const float a21 = -0.969224334;
    const float a22 = 1.875930071;
    const float a23 = 0.041554224;
    const float a31 = 0.055639423;
    const float a32 = -0.204011202;
    const float a33 = 1.057148933;

    float3 linColor;
    linColor.r = a11 * xyzColor.r + a12 * xyzColor.g + a13 * xyzColor.b;
    linColor.g = a21 * xyzColor.r + a22 * xyzColor.g + a23 * xyzColor.b;
    linColor.b = a31 * xyzColor.r + a32 * xyzColor.g + a33 * xyzColor.b;

    return linColor;
}

float3 XYZ2CIELab(float3 xyzColor, const float3 invReferenceIlluminant = InvD65ReferenceIlluminant)
{
    // the default illuminant is D65
    float3 tmpColor = xyzColor * invReferenceIlluminant;

    float delta = 6.0 / 29.0;
    float deltaSquare = delta * delta;
    float deltaCube = delta * deltaSquare;
    float factor = 1.0 / (3.0 * deltaSquare);
    float term = 4.0 / 29.0;

    tmpColor.r = (tmpColor.r > deltaCube ? pow(tmpColor.r, 1.0 / 3.0) : factor * tmpColor.r + term);
    tmpColor.g = (tmpColor.g > deltaCube ? pow(tmpColor.g, 1.0 / 3.0) : factor * tmpColor.g + term);
    tmpColor.b = (tmpColor.b > deltaCube ? pow(tmpColor.b, 1.0 / 3.0) : factor * tmpColor.b + term);

    float3 labColor;
    labColor.r = 116.0 * tmpColor.g - 16.0;
    labColor.g = 500.0 * (tmpColor.r - tmpColor.g);
    labColor.b = 200.0 * (tmpColor.g - tmpColor.b);

    return labColor;
}

float3 CIELab2XYZ(float3 labColor, const float3 referenceIlluminant = D65ReferenceIlluminant)
{
    // the default illuminant is D65
    float Y = (labColor.r + 16.0) / 116.0;
    float X = labColor.g / 500.0 + Y;
    float Z = Y - labColor.b / 200.0;

    float delta = 6.0 / 29.0;
    float factor = 3.0 * delta * delta;
    float term = 4.0 / 29.0;
    X = ((X > delta) ? X * X * X : (X - term) * factor);
    Y = ((Y > delta) ? Y * Y * Y : (Y - term) * factor);
    Z = ((Z > delta) ? Z * Z * Z : (Z - term) * factor);

    return float3(X, Y, Z) * referenceIlluminant;
}

float3 XYZ2YCxCz(float3 xyzColor, const float3 invReferenceIlluminant = InvD65ReferenceIlluminant)
{
    // the default illuminant is D65
    float3 tmpColor = xyzColor * invReferenceIlluminant;

    float3 ycxczColor;
    ycxczColor.x = 116.0 * tmpColor.g - 16.0;
    ycxczColor.y = 500.0 * (tmpColor.r - tmpColor.g);
    ycxczColor.z = 200.0 * (tmpColor.g - tmpColor.b);

    return ycxczColor;
}

float3 YCxCz2XYZ(float3 ycxczColor, const float3 referenceIlluminant = D65ReferenceIlluminant)
{
    // the default illuminant is D65
    float Y = (ycxczColor.r + 16.0) / 116.0;
    float X = ycxczColor.g / 500.0 + Y;
    float Z = Y - ycxczColor.b / 200.0;

    return float3(X, Y, Z) * referenceIlluminant;
}

float3 CIELab2sRGB(float3 labColor)
{
    return linearRGB2sRGB(XYZ2LinearRGB(CIELab2XYZ(labColor)));
}

float3 sRGB2CIELab(float3 srgbColor)
{
    return XYZ2CIELab(linearRGB2XYZ(sRGB2LinearRGB(srgbColor)));
}

float3 linearRGB2CIELab(float3 lColor)
{
    return XYZ2CIELab(linearRGB2XYZ(lColor));
}

float3 sRGB2YCxCz(float3 srgbColor)
{
    return XYZ2YCxCz(linearRGB2XYZ(sRGB2LinearRGB(srgbColor)));
}

float3 YCxCz2LinearRGB(float3 ycxczColor)
{
    return XYZ2LinearRGB(YCxCz2XYZ(ycxczColor));
}

float3 linearRGB2YCxCz(float3 lColor)
{
    return XYZ2YCxCz(linearRGB2XYZ(lColor));
}

float linearRGB2Luminance(float3 linearRGB)
{
    const float3 luminanceConstants = float3(0.2126f, 0.7152f, 0.0722f);
    return dot(linearRGB, luminanceConstants);
}

static const float3 MagmaMap[] = {
    float3(0.001462, 0.000466, 0.013866),
    float3(0.002258, 0.001295, 0.018331),
    float3(0.003279, 0.002305, 0.023708),
    float3(0.004512, 0.003490, 0.029965),
    float3(0.005950, 0.004843, 0.037130),
    float3(0.007588, 0.006356, 0.044973),
    float3(0.009426, 0.008022, 0.052844),
    float3(0.011465, 0.009828, 0.060750),
    float3(0.013708, 0.011771, 0.068667),
    float3(0.016156, 0.013840, 0.076603),
    float3(0.018815, 0.016026, 0.084584),
    float3(0.021692, 0.018320, 0.092610),
    float3(0.024792, 0.020715, 0.100676),
    float3(0.028123, 0.023201, 0.108787),
    float3(0.031696, 0.025765, 0.116965),
    float3(0.035520, 0.028397, 0.125209),
    float3(0.039608, 0.031090, 0.133515),
    float3(0.043830, 0.033830, 0.141886),
    float3(0.048062, 0.036607, 0.150327),
    float3(0.052320, 0.039407, 0.158841),
    float3(0.056615, 0.042160, 0.167446),
    float3(0.060949, 0.044794, 0.176129),
    float3(0.065330, 0.047318, 0.184892),
    float3(0.069764, 0.049726, 0.193735),
    float3(0.074257, 0.052017, 0.202660),
    float3(0.078815, 0.054184, 0.211667),
    float3(0.083446, 0.056225, 0.220755),
    float3(0.088155, 0.058133, 0.229922),
    float3(0.092949, 0.059904, 0.239164),
    float3(0.097833, 0.061531, 0.248477),
    float3(0.102815, 0.063010, 0.257854),
    float3(0.107899, 0.064335, 0.267289),
    float3(0.113094, 0.065492, 0.276784),
    float3(0.118405, 0.066479, 0.286321),
    float3(0.123833, 0.067295, 0.295879),
    float3(0.129380, 0.067935, 0.305443),
    float3(0.135053, 0.068391, 0.315000),
    float3(0.140858, 0.068654, 0.324538),
    float3(0.146785, 0.068738, 0.334011),
    float3(0.152839, 0.068637, 0.343404),
    float3(0.159018, 0.068354, 0.352688),
    float3(0.165308, 0.067911, 0.361816),
    float3(0.171713, 0.067305, 0.370771),
    float3(0.178212, 0.066576, 0.379497),
    float3(0.184801, 0.065732, 0.387973),
    float3(0.191460, 0.064818, 0.396152),
    float3(0.198177, 0.063862, 0.404009),
    float3(0.204935, 0.062907, 0.411514),
    float3(0.211718, 0.061992, 0.418647),
    float3(0.218512, 0.061158, 0.425392),
    float3(0.225302, 0.060445, 0.431742),
    float3(0.232077, 0.059889, 0.437695),
    float3(0.238826, 0.059517, 0.443256),
    float3(0.245543, 0.059352, 0.448436),
    float3(0.252220, 0.059415, 0.453248),
    float3(0.258857, 0.059706, 0.457710),
    float3(0.265447, 0.060237, 0.461840),
    float3(0.271994, 0.060994, 0.465660),
    float3(0.278493, 0.061978, 0.469190),
    float3(0.284951, 0.063168, 0.472451),
    float3(0.291366, 0.064553, 0.475462),
    float3(0.297740, 0.066117, 0.478243),
    float3(0.304081, 0.067835, 0.480812),
    float3(0.310382, 0.069702, 0.483186),
    float3(0.316654, 0.071690, 0.485380),
    float3(0.322899, 0.073782, 0.487408),
    float3(0.329114, 0.075972, 0.489287),
    float3(0.335308, 0.078236, 0.491024),
    float3(0.341482, 0.080564, 0.492631),
    float3(0.347636, 0.082946, 0.494121),
    float3(0.353773, 0.085373, 0.495501),
    float3(0.359898, 0.087831, 0.496778),
    float3(0.366012, 0.090314, 0.497960),
    float3(0.372116, 0.092816, 0.499053),
    float3(0.378211, 0.095332, 0.500067),
    float3(0.384299, 0.097855, 0.501002),
    float3(0.390384, 0.100379, 0.501864),
    float3(0.396467, 0.102902, 0.502658),
    float3(0.402548, 0.105420, 0.503386),
    float3(0.408629, 0.107930, 0.504052),
    float3(0.414709, 0.110431, 0.504662),
    float3(0.420791, 0.112920, 0.505215),
    float3(0.426877, 0.115395, 0.505714),
    float3(0.432967, 0.117855, 0.506160),
    float3(0.439062, 0.120298, 0.506555),
    float3(0.445163, 0.122724, 0.506901),
    float3(0.451271, 0.125132, 0.507198),
    float3(0.457386, 0.127522, 0.507448),
    float3(0.463508, 0.129893, 0.507652),
    float3(0.469640, 0.132245, 0.507809),
    float3(0.475780, 0.134577, 0.507921),
    float3(0.481929, 0.136891, 0.507989),
    float3(0.488088, 0.139186, 0.508011),
    float3(0.494258, 0.141462, 0.507988),
    float3(0.500438, 0.143719, 0.507920),
    float3(0.506629, 0.145958, 0.507806),
    float3(0.512831, 0.148179, 0.507648),
    float3(0.519045, 0.150383, 0.507443),
    float3(0.525270, 0.152569, 0.507192),
    float3(0.531507, 0.154739, 0.506895),
    float3(0.537755, 0.156894, 0.506551),
    float3(0.544015, 0.159033, 0.506159),
    float3(0.550287, 0.161158, 0.505719),
    float3(0.556571, 0.163269, 0.505230),
    float3(0.562866, 0.165368, 0.504692),
    float3(0.569172, 0.167454, 0.504105),
    float3(0.575490, 0.169530, 0.503466),
    float3(0.581819, 0.171596, 0.502777),
    float3(0.588158, 0.173652, 0.502035),
    float3(0.594508, 0.175701, 0.501241),
    float3(0.600868, 0.177743, 0.500394),
    float3(0.607238, 0.179779, 0.499492),
    float3(0.613617, 0.181811, 0.498536),
    float3(0.620005, 0.183840, 0.497524),
    float3(0.626401, 0.185867, 0.496456),
    float3(0.632805, 0.187893, 0.495332),
    float3(0.639216, 0.189921, 0.494150),
    float3(0.645633, 0.191952, 0.492910),
    float3(0.652056, 0.193986, 0.491611),
    float3(0.658483, 0.196027, 0.490253),
    float3(0.664915, 0.198075, 0.488836),
    float3(0.671349, 0.200133, 0.487358),
    float3(0.677786, 0.202203, 0.485819),
    float3(0.684224, 0.204286, 0.484219),
    float3(0.690661, 0.206384, 0.482558),
    float3(0.697098, 0.208501, 0.480835),
    float3(0.703532, 0.210638, 0.479049),
    float3(0.709962, 0.212797, 0.477201),
    float3(0.716387, 0.214982, 0.475290),
    float3(0.722805, 0.217194, 0.473316),
    float3(0.729216, 0.219437, 0.471279),
    float3(0.735616, 0.221713, 0.469180),
    float3(0.742004, 0.224025, 0.467018),
    float3(0.748378, 0.226377, 0.464794),
    float3(0.754737, 0.228772, 0.462509),
    float3(0.761077, 0.231214, 0.460162),
    float3(0.767398, 0.233705, 0.457755),
    float3(0.773695, 0.236249, 0.455289),
    float3(0.779968, 0.238851, 0.452765),
    float3(0.786212, 0.241514, 0.450184),
    float3(0.792427, 0.244242, 0.447543),
    float3(0.798608, 0.247040, 0.444848),
    float3(0.804752, 0.249911, 0.442102),
    float3(0.810855, 0.252861, 0.439305),
    float3(0.816914, 0.255895, 0.436461),
    float3(0.822926, 0.259016, 0.433573),
    float3(0.828886, 0.262229, 0.430644),
    float3(0.834791, 0.265540, 0.427671),
    float3(0.840636, 0.268953, 0.424666),
    float3(0.846416, 0.272473, 0.421631),
    float3(0.852126, 0.276106, 0.418573),
    float3(0.857763, 0.279857, 0.415496),
    float3(0.863320, 0.283729, 0.412403),
    float3(0.868793, 0.287728, 0.409303),
    float3(0.874176, 0.291859, 0.406205),
    float3(0.879464, 0.296125, 0.403118),
    float3(0.884651, 0.300530, 0.400047),
    float3(0.889731, 0.305079, 0.397002),
    float3(0.894700, 0.309773, 0.393995),
    float3(0.899552, 0.314616, 0.391037),
    float3(0.904281, 0.319610, 0.388137),
    float3(0.908884, 0.324755, 0.385308),
    float3(0.913354, 0.330052, 0.382563),
    float3(0.917689, 0.335500, 0.379915),
    float3(0.921884, 0.341098, 0.377376),
    float3(0.925937, 0.346844, 0.374959),
    float3(0.929845, 0.352734, 0.372677),
    float3(0.933606, 0.358764, 0.370541),
    float3(0.937221, 0.364929, 0.368567),
    float3(0.940687, 0.371224, 0.366762),
    float3(0.944006, 0.377643, 0.365136),
    float3(0.947180, 0.384178, 0.363701),
    float3(0.950210, 0.390820, 0.362468),
    float3(0.953099, 0.397563, 0.361438),
    float3(0.955849, 0.404400, 0.360619),
    float3(0.958464, 0.411324, 0.360014),
    float3(0.960949, 0.418323, 0.359630),
    float3(0.963310, 0.425390, 0.359469),
    float3(0.965549, 0.432519, 0.359529),
    float3(0.967671, 0.439703, 0.359810),
    float3(0.969680, 0.446936, 0.360311),
    float3(0.971582, 0.454210, 0.361030),
    float3(0.973381, 0.461520, 0.361965),
    float3(0.975082, 0.468861, 0.363111),
    float3(0.976690, 0.476226, 0.364466),
    float3(0.978210, 0.483612, 0.366025),
    float3(0.979645, 0.491014, 0.367783),
    float3(0.981000, 0.498428, 0.369734),
    float3(0.982279, 0.505851, 0.371874),
    float3(0.983485, 0.513280, 0.374198),
    float3(0.984622, 0.520713, 0.376698),
    float3(0.985693, 0.528148, 0.379371),
    float3(0.986700, 0.535582, 0.382210),
    float3(0.987646, 0.543015, 0.385210),
    float3(0.988533, 0.550446, 0.388365),
    float3(0.989363, 0.557873, 0.391671),
    float3(0.990138, 0.565296, 0.395122),
    float3(0.990871, 0.572706, 0.398714),
    float3(0.991558, 0.580107, 0.402441),
    float3(0.992196, 0.587502, 0.406299),
    float3(0.992785, 0.594891, 0.410283),
    float3(0.993326, 0.602275, 0.414390),
    float3(0.993834, 0.609644, 0.418613),
    float3(0.994309, 0.616999, 0.422950),
    float3(0.994738, 0.624350, 0.427397),
    float3(0.995122, 0.631696, 0.431951),
    float3(0.995480, 0.639027, 0.436607),
    float3(0.995810, 0.646344, 0.441361),
    float3(0.996096, 0.653659, 0.446213),
    float3(0.996341, 0.660969, 0.451160),
    float3(0.996580, 0.668256, 0.456192),
    float3(0.996775, 0.675541, 0.461314),
    float3(0.996925, 0.682828, 0.466526),
    float3(0.997077, 0.690088, 0.471811),
    float3(0.997186, 0.697349, 0.477182),
    float3(0.997254, 0.704611, 0.482635),
    float3(0.997325, 0.711848, 0.488154),
    float3(0.997351, 0.719089, 0.493755),
    float3(0.997351, 0.726324, 0.499428),
    float3(0.997341, 0.733545, 0.505167),
    float3(0.997285, 0.740772, 0.510983),
    float3(0.997228, 0.747981, 0.516859),
    float3(0.997138, 0.755190, 0.522806),
    float3(0.997019, 0.762398, 0.528821),
    float3(0.996898, 0.769591, 0.534892),
    float3(0.996727, 0.776795, 0.541039),
    float3(0.996571, 0.783977, 0.547233),
    float3(0.996369, 0.791167, 0.553499),
    float3(0.996162, 0.798348, 0.559820),
    float3(0.995932, 0.805527, 0.566202),
    float3(0.995680, 0.812706, 0.572645),
    float3(0.995424, 0.819875, 0.579140),
    float3(0.995131, 0.827052, 0.585701),
    float3(0.994851, 0.834213, 0.592307),
    float3(0.994524, 0.841387, 0.598983),
    float3(0.994222, 0.848540, 0.605696),
    float3(0.993866, 0.855711, 0.612482),
    float3(0.993545, 0.862859, 0.619299),
    float3(0.993170, 0.870024, 0.626189),
    float3(0.992831, 0.877168, 0.633109),
    float3(0.992440, 0.884330, 0.640099),
    float3(0.992089, 0.891470, 0.647116),
    float3(0.991688, 0.898627, 0.654202),
    float3(0.991332, 0.905763, 0.661309),
    float3(0.990930, 0.912915, 0.668481),
    float3(0.990570, 0.920049, 0.675675),
    float3(0.990175, 0.927196, 0.682926),
    float3(0.989815, 0.934329, 0.690198),
    float3(0.989434, 0.941470, 0.697519),
    float3(0.989077, 0.948604, 0.704863),
    float3(0.988717, 0.955742, 0.712242),
    float3(0.988367, 0.962878, 0.719649),
    float3(0.988033, 0.970012, 0.727077),
    float3(0.987691, 0.977154, 0.734536),
    float3(0.987387, 0.984288, 0.742002),
    float3(0.987053, 0.991438, 0.749504)
};

static const float3 ViridisMap[] = {
    float3(0.267004, 0.004874, 0.329415),
    float3(0.268510, 0.009605, 0.335427),
    float3(0.269944, 0.014625, 0.341379),
    float3(0.271305, 0.019942, 0.347269),
    float3(0.272594, 0.025563, 0.353093),
    float3(0.273809, 0.031497, 0.358853),
    float3(0.274952, 0.037752, 0.364543),
    float3(0.276022, 0.044167, 0.370164),
    float3(0.277018, 0.050344, 0.375715),
    float3(0.277941, 0.056324, 0.381191),
    float3(0.278791, 0.062145, 0.386592),
    float3(0.279566, 0.067836, 0.391917),
    float3(0.280267, 0.073417, 0.397163),
    float3(0.280894, 0.078907, 0.402329),
    float3(0.281446, 0.084320, 0.407414),
    float3(0.281924, 0.089666, 0.412415),
    float3(0.282327, 0.094955, 0.417331),
    float3(0.282656, 0.100196, 0.422160),
    float3(0.282910, 0.105393, 0.426902),
    float3(0.283091, 0.110553, 0.431554),
    float3(0.283197, 0.115680, 0.436115),
    float3(0.283229, 0.120777, 0.440584),
    float3(0.283187, 0.125848, 0.444960),
    float3(0.283072, 0.130895, 0.449241),
    float3(0.282884, 0.135920, 0.453427),
    float3(0.282623, 0.140926, 0.457517),
    float3(0.282290, 0.145912, 0.461510),
    float3(0.281887, 0.150881, 0.465405),
    float3(0.281412, 0.155834, 0.469201),
    float3(0.280868, 0.160771, 0.472899),
    float3(0.280255, 0.165693, 0.476498),
    float3(0.279574, 0.170599, 0.479997),
    float3(0.278826, 0.175490, 0.483397),
    float3(0.278012, 0.180367, 0.486697),
    float3(0.277134, 0.185228, 0.489898),
    float3(0.276194, 0.190074, 0.493001),
    float3(0.275191, 0.194905, 0.496005),
    float3(0.274128, 0.199721, 0.498911),
    float3(0.273006, 0.204520, 0.501721),
    float3(0.271828, 0.209303, 0.504434),
    float3(0.270595, 0.214069, 0.507052),
    float3(0.269308, 0.218818, 0.509577),
    float3(0.267968, 0.223549, 0.512008),
    float3(0.266580, 0.228262, 0.514349),
    float3(0.265145, 0.232956, 0.516599),
    float3(0.263663, 0.237631, 0.518762),
    float3(0.262138, 0.242286, 0.520837),
    float3(0.260571, 0.246922, 0.522828),
    float3(0.258965, 0.251537, 0.524736),
    float3(0.257322, 0.256130, 0.526563),
    float3(0.255645, 0.260703, 0.528312),
    float3(0.253935, 0.265254, 0.529983),
    float3(0.252194, 0.269783, 0.531579),
    float3(0.250425, 0.274290, 0.533103),
    float3(0.248629, 0.278775, 0.534556),
    float3(0.246811, 0.283237, 0.535941),
    float3(0.244972, 0.287675, 0.537260),
    float3(0.243113, 0.292092, 0.538516),
    float3(0.241237, 0.296485, 0.539709),
    float3(0.239346, 0.300855, 0.540844),
    float3(0.237441, 0.305202, 0.541921),
    float3(0.235526, 0.309527, 0.542944),
    float3(0.233603, 0.313828, 0.543914),
    float3(0.231674, 0.318106, 0.544834),
    float3(0.229739, 0.322361, 0.545706),
    float3(0.227802, 0.326594, 0.546532),
    float3(0.225863, 0.330805, 0.547314),
    float3(0.223925, 0.334994, 0.548053),
    float3(0.221989, 0.339161, 0.548752),
    float3(0.220057, 0.343307, 0.549413),
    float3(0.218130, 0.347432, 0.550038),
    float3(0.216210, 0.351535, 0.550627),
    float3(0.214298, 0.355619, 0.551184),
    float3(0.212395, 0.359683, 0.551710),
    float3(0.210503, 0.363727, 0.552206),
    float3(0.208623, 0.367752, 0.552675),
    float3(0.206756, 0.371758, 0.553117),
    float3(0.204903, 0.375746, 0.553533),
    float3(0.203063, 0.379716, 0.553925),
    float3(0.201239, 0.383670, 0.554294),
    float3(0.199430, 0.387607, 0.554642),
    float3(0.197636, 0.391528, 0.554969),
    float3(0.195860, 0.395433, 0.555276),
    float3(0.194100, 0.399323, 0.555565),
    float3(0.192357, 0.403199, 0.555836),
    float3(0.190631, 0.407061, 0.556089),
    float3(0.188923, 0.410910, 0.556326),
    float3(0.187231, 0.414746, 0.556547),
    float3(0.185556, 0.418570, 0.556753),
    float3(0.183898, 0.422383, 0.556944),
    float3(0.182256, 0.426184, 0.557120),
    float3(0.180629, 0.429975, 0.557282),
    float3(0.179019, 0.433756, 0.557430),
    float3(0.177423, 0.437527, 0.557565),
    float3(0.175841, 0.441290, 0.557685),
    float3(0.174274, 0.445044, 0.557792),
    float3(0.172719, 0.448791, 0.557885),
    float3(0.171176, 0.452530, 0.557965),
    float3(0.169646, 0.456262, 0.558030),
    float3(0.168126, 0.459988, 0.558082),
    float3(0.166617, 0.463708, 0.558119),
    float3(0.165117, 0.467423, 0.558141),
    float3(0.163625, 0.471133, 0.558148),
    float3(0.162142, 0.474838, 0.558140),
    float3(0.160665, 0.478540, 0.558115),
    float3(0.159194, 0.482237, 0.558073),
    float3(0.157729, 0.485932, 0.558013),
    float3(0.156270, 0.489624, 0.557936),
    float3(0.154815, 0.493313, 0.557840),
    float3(0.153364, 0.497000, 0.557724),
    float3(0.151918, 0.500685, 0.557587),
    float3(0.150476, 0.504369, 0.557430),
    float3(0.149039, 0.508051, 0.557250),
    float3(0.147607, 0.511733, 0.557049),
    float3(0.146180, 0.515413, 0.556823),
    float3(0.144759, 0.519093, 0.556572),
    float3(0.143343, 0.522773, 0.556295),
    float3(0.141935, 0.526453, 0.555991),
    float3(0.140536, 0.530132, 0.555659),
    float3(0.139147, 0.533812, 0.555298),
    float3(0.137770, 0.537492, 0.554906),
    float3(0.136408, 0.541173, 0.554483),
    float3(0.135066, 0.544853, 0.554029),
    float3(0.133743, 0.548535, 0.553541),
    float3(0.132444, 0.552216, 0.553018),
    float3(0.131172, 0.555899, 0.552459),
    float3(0.129933, 0.559582, 0.551864),
    float3(0.128729, 0.563265, 0.551229),
    float3(0.127568, 0.566949, 0.550556),
    float3(0.126453, 0.570633, 0.549841),
    float3(0.125394, 0.574318, 0.549086),
    float3(0.124395, 0.578002, 0.548287),
    float3(0.123463, 0.581687, 0.547445),
    float3(0.122606, 0.585371, 0.546557),
    float3(0.121831, 0.589055, 0.545623),
    float3(0.121148, 0.592739, 0.544641),
    float3(0.120565, 0.596422, 0.543611),
    float3(0.120092, 0.600104, 0.542530),
    float3(0.119738, 0.603785, 0.541400),
    float3(0.119512, 0.607464, 0.540218),
    float3(0.119423, 0.611141, 0.538982),
    float3(0.119483, 0.614817, 0.537692),
    float3(0.119699, 0.618490, 0.536347),
    float3(0.120081, 0.622161, 0.534946),
    float3(0.120638, 0.625828, 0.533488),
    float3(0.121380, 0.629492, 0.531973),
    float3(0.122312, 0.633153, 0.530398),
    float3(0.123444, 0.636809, 0.528763),
    float3(0.124780, 0.640461, 0.527068),
    float3(0.126326, 0.644107, 0.525311),
    float3(0.128087, 0.647749, 0.523491),
    float3(0.130067, 0.651384, 0.521608),
    float3(0.132268, 0.655014, 0.519661),
    float3(0.134692, 0.658636, 0.517649),
    float3(0.137339, 0.662252, 0.515571),
    float3(0.140210, 0.665859, 0.513427),
    float3(0.143303, 0.669459, 0.511215),
    float3(0.146616, 0.673050, 0.508936),
    float3(0.150148, 0.676631, 0.506589),
    float3(0.153894, 0.680203, 0.504172),
    float3(0.157851, 0.683765, 0.501686),
    float3(0.162016, 0.687316, 0.499129),
    float3(0.166383, 0.690856, 0.496502),
    float3(0.170948, 0.694384, 0.493803),
    float3(0.175707, 0.697900, 0.491033),
    float3(0.180653, 0.701402, 0.488189),
    float3(0.185783, 0.704891, 0.485273),
    float3(0.191090, 0.708366, 0.482284),
    float3(0.196571, 0.711827, 0.479221),
    float3(0.202219, 0.715272, 0.476084),
    float3(0.208030, 0.718701, 0.472873),
    float3(0.214000, 0.722114, 0.469588),
    float3(0.220124, 0.725509, 0.466226),
    float3(0.226397, 0.728888, 0.462789),
    float3(0.232815, 0.732247, 0.459277),
    float3(0.239374, 0.735588, 0.455688),
    float3(0.246070, 0.738910, 0.452024),
    float3(0.252899, 0.742211, 0.448284),
    float3(0.259857, 0.745492, 0.444467),
    float3(0.266941, 0.748751, 0.440573),
    float3(0.274149, 0.751988, 0.436601),
    float3(0.281477, 0.755203, 0.432552),
    float3(0.288921, 0.758394, 0.428426),
    float3(0.296479, 0.761561, 0.424223),
    float3(0.304148, 0.764704, 0.419943),
    float3(0.311925, 0.767822, 0.415586),
    float3(0.319809, 0.770914, 0.411152),
    float3(0.327796, 0.773980, 0.406640),
    float3(0.335885, 0.777018, 0.402049),
    float3(0.344074, 0.780029, 0.397381),
    float3(0.352360, 0.783011, 0.392636),
    float3(0.360741, 0.785964, 0.387814),
    float3(0.369214, 0.788888, 0.382914),
    float3(0.377779, 0.791781, 0.377939),
    float3(0.386433, 0.794644, 0.372886),
    float3(0.395174, 0.797475, 0.367757),
    float3(0.404001, 0.800275, 0.362552),
    float3(0.412913, 0.803041, 0.357269),
    float3(0.421908, 0.805774, 0.351910),
    float3(0.430983, 0.808473, 0.346476),
    float3(0.440137, 0.811138, 0.340967),
    float3(0.449368, 0.813768, 0.335384),
    float3(0.458674, 0.816363, 0.329727),
    float3(0.468053, 0.818921, 0.323998),
    float3(0.477504, 0.821444, 0.318195),
    float3(0.487026, 0.823929, 0.312321),
    float3(0.496615, 0.826376, 0.306377),
    float3(0.506271, 0.828786, 0.300362),
    float3(0.515992, 0.831158, 0.294279),
    float3(0.525776, 0.833491, 0.288127),
    float3(0.535621, 0.835785, 0.281908),
    float3(0.545524, 0.838039, 0.275626),
    float3(0.555484, 0.840254, 0.269281),
    float3(0.565498, 0.842430, 0.262877),
    float3(0.575563, 0.844566, 0.256415),
    float3(0.585678, 0.846661, 0.249897),
    float3(0.595839, 0.848717, 0.243329),
    float3(0.606045, 0.850733, 0.236712),
    float3(0.616293, 0.852709, 0.230052),
    float3(0.626579, 0.854645, 0.223353),
    float3(0.636902, 0.856542, 0.216620),
    float3(0.647257, 0.858400, 0.209861),
    float3(0.657642, 0.860219, 0.203082),
    float3(0.668054, 0.861999, 0.196293),
    float3(0.678489, 0.863742, 0.189503),
    float3(0.688944, 0.865448, 0.182725),
    float3(0.699415, 0.867117, 0.175971),
    float3(0.709898, 0.868751, 0.169257),
    float3(0.720391, 0.870350, 0.162603),
    float3(0.730889, 0.871916, 0.156029),
    float3(0.741388, 0.873449, 0.149561),
    float3(0.751884, 0.874951, 0.143228),
    float3(0.762373, 0.876424, 0.137064),
    float3(0.772852, 0.877868, 0.131109),
    float3(0.783315, 0.879285, 0.125405),
    float3(0.793760, 0.880678, 0.120005),
    float3(0.804182, 0.882046, 0.114965),
    float3(0.814576, 0.883393, 0.110347),
    float3(0.824940, 0.884720, 0.106217),
    float3(0.835270, 0.886029, 0.102646),
    float3(0.845561, 0.887322, 0.099702),
    float3(0.855810, 0.888601, 0.097452),
    float3(0.866013, 0.889868, 0.095953),
    float3(0.876168, 0.891125, 0.095250),
    float3(0.886271, 0.892374, 0.095374),
    float3(0.896320, 0.893616, 0.096335),
    float3(0.906311, 0.894855, 0.098125),
    float3(0.916242, 0.896091, 0.100717),
    float3(0.926106, 0.897330, 0.104071),
    float3(0.935904, 0.898570, 0.108131),
    float3(0.945636, 0.899815, 0.112838),
    float3(0.955300, 0.901065, 0.118128),
    float3(0.964894, 0.902323, 0.123941),
    float3(0.974417, 0.903590, 0.130215),
    float3(0.983868, 0.904867, 0.136897),
    float3(0.993248, 0.906157, 0.143936)
};

#endif // #define FLIP_SHADER_COMMON_HLSLI
