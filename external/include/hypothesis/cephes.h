/*
    cephes.h: A subset of cephes math routines used by hypothesis.h

    Redistributed under the BSD license with permission of the author, see
    https://github.com/deepmind/torch-cephes/blob/master/LICENSE.txt

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
          notice, this list of conditions and the following disclaimer in the
          documentation and/or other materials provided with the distribution.
        * Neither the name of the <organization> nor the
          names of its contributors may be used to endorse or promote products
          derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <cmath>
#include <stdexcept>

namespace cephes {
    static const double biginv =  2.22044604925031308085e-16;
    static const double big = 4.503599627370496e15;
    static const double MAXGAM = 171.624376956302725;
    static const double MACHEP = 1.11022302462515654042E-16;
    static const double MAXLOG = 7.09782712893383996843E2;
    static const double MINLOG = -7.08396418532264106224E2;

    /* Forward declarations */
    static double pseries(double a, double b, double x);
    static double incbd(double a, double b, double x);
    static double incbcf(double a, double b, double x);

    inline double incbet(double aa, double bb, double xx) {
        double a, b, t, x, xc, w, y;
        int flag;

        if (aa <= 0.0 || bb <= 0.0)
            goto domerr;

        if ((xx <= 0.0) || (xx >= 1.0)) {
            if (xx == 0.0)
                return 0.0;
            if (xx == 1.0)
                return 1.0;
            domerr:
            throw std::runtime_error("incbet: domain error!");
        }

        flag = 0;
        if ((bb * xx) <= 1.0 && xx <= 0.95) {
            t = pseries(aa, bb, xx);
            goto done;
        }

        w = 1.0 - xx;

        /* Reverse a and b if x is greater than the mean. */
        if (xx > (aa / (aa + bb))) {
            flag = 1;
            a = bb;
            b = aa;
            xc = xx;
            x = w;
        } else {
            a = aa;
            b = bb;
            xc = w;
            x = xx;
        }

        if (flag == 1 && (b * x) <= 1.0 && x <= 0.95) {
            t = pseries(a, b, x);
            goto done;
        }

        /* Choose expansion for better convergence. */
        y = x * (a + b - 2.0) - (a - 1.0);
        if (y < 0.0)
            w = incbcf(a, b, x);
        else
            w = incbd(a, b, x) / xc;

        /* Multiply w by the factor
             a      b   _             _     _
            x  (1-x)   | (a+b) / ( a | (a) | (b) ) .   */

        y = a * std::log(x);
        t = b * std::log(xc);
        if ((a + b) < MAXGAM && std::abs(y) < MAXLOG && std::abs(t) < MAXLOG) {
            t = pow(xc, b);
            t *= pow(x, a);
            t /= a;
            t *= w;
            t *= std::tgamma(a + b) / (std::tgamma(a) * std::tgamma(b));
            goto done;
        }
        /* Resort to logarithms.  */
        y += t + std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
        y += std::log(w / a);
        if (y < MINLOG)
            t = 0.0;
        else
            t = std::exp(y);

    done:

        if (flag == 1) {
            if (t <= MACHEP)
                t = 1.0 - MACHEP;
            else
                t = 1.0 - t;
        }
        return t;
    }

    /* Continued fraction expansion #1
     * for incomplete beta integral
     */
    inline static double incbcf(double a, double b, double x) {
        double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
        double k1, k2, k3, k4, k5, k6, k7, k8;
        double r, t, ans, thresh;
        int n;

        k1 = a;
        k2 = a + b;
        k3 = a;
        k4 = a + 1.0;
        k5 = 1.0;
        k6 = b - 1.0;
        k7 = k4;
        k8 = a + 2.0;

        pkm2 = 0.0;
        qkm2 = 1.0;
        pkm1 = 1.0;
        qkm1 = 1.0;
        ans = 1.0;
        r = 1.0;
        n = 0;
        thresh = 3.0 * MACHEP;
        do {

            xk = -(x * k1 * k2) / (k3 * k4);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            xk = (x * k5 * k6) / (k7 * k8);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            if (qk != 0)
                r = pk / qk;
            if (r != 0) {
                t = std::abs((ans - r) / r);
                ans = r;
            } else
                t = 1.0;

            if (t < thresh)
                goto cdone;

            k1 += 1.0;
            k2 += 1.0;
            k3 += 2.0;
            k4 += 2.0;
            k5 += 1.0;
            k6 -= 1.0;
            k7 += 2.0;
            k8 += 2.0;

            if ((std::abs(qk) + std::abs(pk)) > big) {
                pkm2 *= biginv;
                pkm1 *= biginv;
                qkm2 *= biginv;
                qkm1 *= biginv;
            }
            if ((std::abs(qk) < biginv) || (std::abs(pk) < biginv)) {
                pkm2 *= big;
                pkm1 *= big;
                qkm2 *= big;
                qkm1 *= big;
            }
        } while (++n < 300);

    cdone:
        return (ans);
    }

    /* Continued fraction expansion #2
     * for incomplete beta integral
     */
    inline static double incbd(double a, double b, double x) {
        double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
        double k1, k2, k3, k4, k5, k6, k7, k8;
        double r, t, ans, z, thresh;
        int n;

        k1 = a;
        k2 = b - 1.0;
        k3 = a;
        k4 = a + 1.0;
        k5 = 1.0;
        k6 = a + b;
        k7 = a + 1.0;
        k8 = a + 2.0;

        pkm2 = 0.0;
        qkm2 = 1.0;
        pkm1 = 1.0;
        qkm1 = 1.0;
        z = x / (1.0 - x);
        ans = 1.0;
        r = 1.0;
        n = 0;
        thresh = 3.0 * MACHEP;
        do {

            xk = -(z * k1 * k2) / (k3 * k4);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            xk = (z * k5 * k6) / (k7 * k8);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1;
            pkm1 = pk;
            qkm2 = qkm1;
            qkm1 = qk;

            if (qk != 0)
                r = pk / qk;
            if (r != 0) {
                t = std::abs((ans - r) / r);
                ans = r;
            } else
                t = 1.0;

            if (t < thresh)
                goto cdone;

            k1 += 1.0;
            k2 -= 1.0;
            k3 += 2.0;
            k4 += 2.0;
            k5 += 1.0;
            k6 += 1.0;
            k7 += 2.0;
            k8 += 2.0;

            if ((std::abs(qk) + std::abs(pk)) > big) {
                pkm2 *= biginv;
                pkm1 *= biginv;
                qkm2 *= biginv;
                qkm1 *= biginv;
            }
            if ((std::abs(qk) < biginv) || (std::abs(pk) < biginv)) {
                pkm2 *= big;
                pkm1 *= big;
                qkm2 *= big;
                qkm1 *= big;
            }
        } while (++n < 300);
    cdone:
        return (ans);
    }

    /* Power series for incomplete beta integral.
       Use when b*x is small and x not too close to 1.  */
    inline static double pseries(double a, double b, double x) {
        double s, t, u, v, n, t1, z, ai;

        ai = 1.0 / a;
        u = (1.0 - b) * x;
        v = u / (a + 1.0);
        t1 = v;
        t = u;
        n = 2.0;
        s = 0.0;
        z = MACHEP * ai;
        while (std::abs(v) > z) {
            u = (n - b) * x / n;
            t *= u;
            v = t / (a + n);
            s += v;
            n += 1.0;
        }
        s += t1;
        s += ai;

        u = a * std::log(x);
        if ((a + b) < MAXGAM && std::abs(u) < MAXLOG) {
            t = std::tgamma(a + b) / (std::tgamma(a) * std::tgamma(b));
            s = s * t * pow(x, a);
        } else {
            t = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + u + std::log(s);
            if (t < MINLOG)
                s = 0.0;
            else
                s = std::exp(t);
        }
        return s;
    }

    /// Regularized lower incomplete gamma function
    inline double rlgamma(double a, double x) {
        const double epsilon = 0.000000000000001;

        if (a < 0 || x < 0)
            throw std::runtime_error("LLGamma: invalid arguments range!");

        if (x == 0)
            return 0.0;

        double ax = (a * std::log(x)) - x - std::lgamma(a);
        if (ax < -709.78271289338399)
            return a < x ? 1.0 : 0.0;

        if (x <= 1 || x <= a) {
                double r2 = a;
                double c2 = 1;
                double ans2 = 1;

            do {
                r2 = r2 + 1;
                c2 = c2 * x / r2;
                ans2 += c2;
            } while ((c2 / ans2) > epsilon);

            return std::exp(ax) * ans2 / a;
        }

        int c = 0;
        double y = 1 - a;
        double z = x + y + 1;
        double p3 = 1;
        double q3 = x;
        double p2 = x + 1;
        double q2 = z * x;
        double ans = p2 / q2;
        double error;

        do {
            c++;
            y += 1;
            z += 2;
            double yc = y * c;
            double p = (p2 * z) - (p3 * yc);
            double q = (q2 * z) - (q3 * yc);

            if (q != 0) {
                double nextans = p / q;
                error = std::abs((ans - nextans) / nextans);
                ans = nextans;
            } else {
                // zero div, skip
                error = 1;
            }

            // shift
            p3 = p2;
            p2 = p;
            q3 = q2;
            q2 = q;

            // normalize fraction when the numerator becomes large
            if (std::abs(p) > big) {
                p3 *= biginv;
                p2 *= biginv;
                q3 *= biginv;
                q2 *= biginv;
            }
        } while (error > epsilon);

        return 1.0 - (std::exp(ax) * ans);
    }
};
