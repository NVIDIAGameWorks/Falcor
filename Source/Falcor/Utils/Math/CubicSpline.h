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

namespace Falcor
{
    template<typename T>
    class CubicSpline
    {
    public:
        /** Creates a position-based cubic spline.
            \param[in] controlPoints Array of control points
            \param[in] pointCount Number of control points
        */
        CubicSpline(const T* controlPoints, uint32_t pointCount)
        {
            // The following code is based on the article from http://graphicsrunner.blogspot.co.uk/2008/05/camera-animation-part-ii.html
            static const T kHalf  = T(0.5f);
            static const T kOne = T(1);
            static const T kTwo = T(2);
            static const T kThree = T(3);
            static const T kFour = T(4);

            // Calculate Gamma
            std::vector<T> gamma(pointCount);
            gamma[0] = kHalf;
            for(uint32_t i = 1; i < pointCount - 1; i++)
            {
                gamma[i] = kOne / (kFour - gamma[i - 1]);
            }
            gamma[pointCount - 1] = kOne / (kTwo - gamma[pointCount - 2]);

            // Calculate Delta
            std::vector<T> delta(pointCount);
            delta[0] = kThree * (controlPoints[1] - controlPoints[0]) * gamma[0];

            for(uint32_t i = 1; i < pointCount; i++)
            {
                uint32_t index = (i == (pointCount - 1)) ? i : i + 1;
                delta[i] = (kThree * (controlPoints[index] - controlPoints[i - 1]) - delta[i - 1]) * gamma[i];
            }

            // Calculate D
            std::vector<T> D(pointCount);
            D[pointCount - 1] = delta[pointCount - 1];

            for(int32_t i = int32_t(pointCount - 2); i >= 0; i--)
            {
                D[i] = delta[i] - gamma[i] * D[i + 1];
            }

            // Calculate the coefficients
            mCoefficient.resize(pointCount - 1);

            for(uint32_t i = 0; i < pointCount - 1; i++)
            {
                mCoefficient[i].a = controlPoints[i];
                mCoefficient[i].b = D[i];
                mCoefficient[i].c = kThree * (controlPoints[i + 1] - controlPoints[i]) - kTwo * D[i] - D[i + 1];
                mCoefficient[i].d = kTwo * (controlPoints[i] - controlPoints[i + 1]) + D[i] + D[i + 1];
            }
        }

        /** Create a position and time-based cubic spline
            \param[in] controlPoints Array of control points
            \param[in] pointCount Number of control points
            \param[in] durations Array containing durations/intervals for each control point
        */
        CubicSpline(const T* points, uint32_t pointCount, float const* durations)
        {
            // The following code is based on the article from http://graphicsrunner.blogspot.co.uk/2008/05/camera-animation-part-ii.html
            // http://math.stackexchange.com/questions/62360/natural-cubic-splines-vs-piecewise-hermite-splines
            // https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
            if (pointCount >= 2)
            {
                mCoefficient.resize(pointCount);

                // Calculate Gamma =: mCoefficient.a
                mCoefficient[0].a = T(.5f);
                for (size_t i = 1; i < pointCount - 1; i++) {
                    mCoefficient[i].a = durations[i] / (T(2.f) * (durations[i - 1] + durations[i]) - durations[i - 1] * mCoefficient[i - 1].a);
                }
                mCoefficient[pointCount - 1].a = 1.0f / (T(2) - mCoefficient[pointCount - 2].a);

                // Calculate Delta =: mCoefficient.b
                mCoefficient[0].b = T(3) / durations[0] * (points[1] - points[0]) * mCoefficient[0].a;
                for (size_t i = 1; i < pointCount - 1; i++) {
                    mCoefficient[i].b = (T(3) / (durations[i - 1] * durations[i]) * (
                        durations[i - 1] * durations[i - 1] * (points[i + 1] - points[i])
                        + durations[i] * durations[i] * (points[i] - points[i - 1])
                        )
                        - durations[i - 1] * mCoefficient[i - 1].b) * mCoefficient[i].a / durations[i];
                }
                mCoefficient[pointCount - 1].b = (T(3) / durations[pointCount - 2] * (points[pointCount - 1] - points[pointCount - 2]) - mCoefficient[pointCount - 2].b) * mCoefficient[pointCount - 1].a;

                // Calculate D := mCoefficient.d
                mCoefficient[pointCount - 1].d = mCoefficient[pointCount - 1].b;
                for (size_t i = pointCount - 1; i-- > 0;) {
                    mCoefficient[i].d = mCoefficient[i].b - mCoefficient[i].a * mCoefficient[i + 1].d;
                }

                // Calculate actual spline
                for (size_t i = 0; i < pointCount - 1; i++) {
                    mCoefficient[i].a = points[i];
                    mCoefficient[i].b = mCoefficient[i].d * durations[i];
                    mCoefficient[i].c = T(3) * (points[i + 1] - points[i]) - T(2) * mCoefficient[i].d * durations[i] - mCoefficient[i + 1].d * durations[i];
                    mCoefficient[i].d = T(2) * (points[i] - points[i + 1]) + mCoefficient[i].d * durations[i] + mCoefficient[i + 1].d * durations[i];
                }

                mCoefficient.resize(pointCount - 1);
            }
        }

        T interpolate(uint32_t section, float point) const
        {
            const CubicCoeff& coeff = mCoefficient[section];
            T result = (((coeff.d * point) + coeff.c) * point + coeff.b) * point + coeff.a;
            return result;
        }

    private:
        struct CubicCoeff
        {
            T a, b, c, d;
        };
        std::vector<CubicCoeff> mCoefficient;
    };
}