/*
    hypothesis.h: A collection of quantile and quadrature routines
    for Z, Chi^2, and Student's T hypothesis tests.

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

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "cephes.h"

namespace hypothesis {
    /// Cumulative distribution function of the standard normal distribution
    inline double stdnormal_cdf(double x) {
        return std::erfc(-x/std::sqrt(2.0))*0.5;
    }

    /// Cumulative distribution function of the Chi^2 distribution
    inline double chi2_cdf(double x, int dof) {
        if (dof < 1 || x < 0) {
            return 0.0;
        } else if (dof == 2) {
            return 1.0 - std::exp(-0.5*x);
        } else {
            return cephes::rlgamma(0.5 * dof, 0.5 * x);
        }
    }

    /// Cumulative distribution function of Student's T distribution
    inline double students_t_cdf(double x, int dof) {
        if (x > 0)
            return 1-0.5*cephes::incbet(dof * 0.5, 0.5, dof/(x*x+dof));
        else
            return 0.5*cephes::incbet(dof * 0.5, 0.5, dof/(x*x+dof));
    }

    /// adaptive Simpson integration over an 1D interval
    inline double adaptiveSimpson(const std::function<double (double)> &f, double x0, double x1, double eps = 1e-6, int depth = 6) {
        int count = 0;
        /* Define an recursive lambda function for integration over subintervals */
        std::function<double (double, double, double, double, double, double, double, double, int)> integrate =
            [&](double a, double b, double c, double fa, double fb, double fc, double I, double eps, int depth) {
            /* Evaluate the function at two intermediate points */
            double d = 0.5 * (a + b), e = 0.5 * (b + c), fd = f(d), fe = f(e);

            /* Simpson integration over each subinterval */
            double h = c-a,
                  I0 = (1.0/12.0) * h * (fa + 4.0*fd + fb),
                  I1 = (1.0/12.0) * h * (fb + 4.0*fe + fc),
                  Ip = I0+I1;
            ++count;

            /* Stopping criterion from J.N. Lyness (1969)
              "Notes on the adaptive Simpson quadrature routine" */
            if (depth <= 0 || std::abs(Ip-I) < 15.0*eps) {
                // Richardson extrapolation
                return Ip + (1.0/15.0) * (Ip-I);
            }

            return integrate(a, d, b, fa, fd, fb, I0, 0.5*eps, depth-1) +
                   integrate(b, e, c, fb, fe, fc, I1, 0.5*eps, depth-1);
        };
        double a = x0, b = 0.5 * (x0+x1), c = x1;
        double fa = f(a), fb = f(b), fc = f(c);
        double I = (c-a) * (1.0/6.0) * (fa+4.0*fb+fc);
        return integrate(a, b, c, fa, fb, fc, I, eps, depth);
    }

    /// Nested adaptive Simpson integration over a 2D rectangle
    inline double adaptiveSimpson2D(const std::function<double (double, double)> &f, double x0, double y0,
            double x1, double y1, double eps = 1e-6, int depth = 6) {
        /* Lambda function that integrates over the X axis */
        auto integrate = [&](double y) {
            return adaptiveSimpson(std::bind(f, std::placeholders::_1, y), x0, x1, eps, depth);
        };
        double value = adaptiveSimpson(integrate, y0, y1, eps, depth);
        return value;
    }

    /**
     * Peform a Chi^2 test based on the given frequency tables
     *
     * \param nCells
     *   Total number of table cells
     *
     * \param obsFrequencies
     *   Observed cell frequencies in each cell
     *
     * \param expFrequencies
     *   Integrated cell frequencies in each cell (i.e. the noise-free reference)
     *
     * \param sampleCount
     *   Total observed sample count
     *
     * \param minExpFrequency
     *   Minimum expected cell frequency. The chi^2 test does not work reliably
     *   when the expected frequency in a cell is low (e.g. less than 5), because
     *   normality assumptions break down in this case. Therefore, the
     *   implementation will merge such low-frequency cells when they fall below
     *   the threshold specified here.
     *
     * \param significanceLevel
     *   The null hypothesis will be rejected when the associated
     *   p-value is below the significance level specified here.
     *
     * \param numTests
     *   Specifies the total number of tests that will be executed. If greater than one,
     *   the Sidak correction will be applied to the significance level. This is because
     *   by conducting multiple independent hypothesis tests in sequence, the probability
     *   of a failure increases accordingly.
     *
     * \return
     *   A pair of values containing the test result (success: \c true and failure: \c false)
     *   and a descriptive string
     */
    inline std::pair<bool, std::string> chi2_test(
            int nCells, const double *obsFrequencies, const double *expFrequencies,
            int sampleCount, double minExpFrequency, double significanceLevel, int numTests = 1) {

        struct Cell {
            double expFrequency;
            size_t index;
        };

        /* Sort all cells by their expected frequencies */
        std::vector<Cell> cells(nCells);
        for (size_t i=0; i<cells.size(); ++i) {
            cells[i].expFrequency = expFrequencies[i];
            cells[i].index = i;
        }
        std::sort(cells.begin(), cells.end(), [](const Cell &a, const Cell &b) {
            return a.expFrequency < b.expFrequency;
        });

        /* Compute the Chi^2 statistic and pool cells as necessary */
        double pooledFrequencies = 0, pooledExpFrequencies = 0, chsq = 0;
        int pooledCells = 0, dof = 0;

        std::ostringstream oss;
        for (const Cell &c : cells) {
            if (expFrequencies[c.index] < 0) {
                oss << "Encountered a negative expected number of samples ("
                    << expFrequencies[c.index]
                    << "). Rejecting the null hypothesis!" << std::endl;
                return std::make_pair(false, oss.str());
            } else if (expFrequencies[c.index] == 0) {
                if (obsFrequencies[c.index] > sampleCount * 1e-5) {
                    /* Uh oh: samples in a cell that should be completely empty
                       according to the probability density function. Ordinarily,
                       even a single sample requires immediate rejection of the null
                       hypothesis. But due to finite-precision computations and rounding
                       errors, this can occasionally happen without there being an
                       actual bug. Therefore, the criterion here is a bit more lenient. */

                    oss << "Encountered " << obsFrequencies[c.index] << " samples in a cell "
                        << "with expected frequency 0. Rejecting the null hypothesis!" << std::endl;
                    return std::make_pair(false, oss.str());
                }
            } else if (expFrequencies[c.index] < minExpFrequency) {
                /* Pool cells with low expected frequencies */
                pooledFrequencies += obsFrequencies[c.index];
                pooledExpFrequencies += expFrequencies[c.index];
                pooledCells++;
            } else if (pooledExpFrequencies > 0 && pooledExpFrequencies < minExpFrequency) {
                /* Keep on pooling cells until a sufficiently high
                   expected frequency is achieved. */
                pooledFrequencies += obsFrequencies[c.index];
                pooledExpFrequencies += expFrequencies[c.index];
                pooledCells++;
            } else {
                double diff = obsFrequencies[c.index] - expFrequencies[c.index];
                chsq += (diff*diff) / expFrequencies[c.index];
                ++dof;
            }
        }

        if (pooledExpFrequencies > 0 || pooledFrequencies > 0) {
            oss << "Pooled " << pooledCells << " to ensure sufficiently high expected "
                   "cell frequencies (>" << minExpFrequency << ")" << std::endl;
            double diff = pooledFrequencies - pooledExpFrequencies;
            chsq += (diff*diff) / pooledExpFrequencies;
            ++dof;
        }

        /* All parameters are assumed to be known, so there is no
           additional DF reduction due to model parameters */
        dof -= 1;

        if (dof <= 0) {
            oss << "The number of degrees of freedom (" << dof << ") is too low!" << std::endl;
            return std::make_pair(false, oss.str());
        }

        oss << "Chi^2 statistic = " << chsq << " (d.o.f. = " << dof << ")" << std::endl;

        /* Probability of obtaining a test statistic at least
           as extreme as the one observed under the assumption
           that the distributions match */
        double pval = 1 - (double) chi2_cdf(chsq, dof);

        /* Apply the Sidak correction term, since we'll be conducting multiple independent
           hypothesis tests. This accounts for the fact that the probability of a failure
           increases quickly when several hypothesis tests are run in sequence. */
        double alpha = 1.0 - std::pow(1.0 - significanceLevel, 1.0 / numTests);

        bool result = false;
        if (pval < alpha || !std::isfinite(pval)) {
            oss << "***** Rejected ***** the null hypothesis (p-value = " << pval << ", "
                "significance level = " << alpha << ")" << std::endl;
        } else {
            oss << "Accepted the null hypothesis (p-value = " << pval << ", "
                "significance level = " << alpha << ")" << std::endl;
            result = true;
        }
        return std::make_pair(result, oss.str());
    }

    /// Write 2D Chi^2 frequency tables to disk in a format that is nicely plottable by Octave and MATLAB
    inline void chi2_dump(int res1, int res2, const double *obsFrequencies, const double *expFrequencies, const std::string &filename) {
        std::ofstream f(filename);

        f << "obsFrequencies = [ ";
        for (int i=0; i<res1; ++i) {
            for (int j=0; j<res2; ++j) {
                f << obsFrequencies[i*res2+j];
                if (j+1 < res2)
                    f << ", ";
            }
            if (i+1 < res1)
                f << "; ";
        }
        f << " ];" << std::endl
            << "expFrequencies = [ ";
        for (int i=0; i<res1; ++i) {
            for (int j=0; j<res2; ++j) {
                f << expFrequencies[i*res2+j];
                if (j+1 < res2)
                    f << ", ";
            }
            if (i+1 < res1)
                f << "; ";
        }
        f << " ];" << std::endl
            << "colormap(jet);" << std::endl
            << "clf; subplot(2,1,1);" << std::endl
            << "imagesc(obsFrequencies);" << std::endl
            << "title('Observed frequencies');" << std::endl
            << "axis equal;" << std::endl
            << "subplot(2,1,2);" << std::endl
            << "imagesc(expFrequencies);" << std::endl
            << "axis equal;" << std::endl
            << "title('Expected frequencies');" << std::endl;
        f.close();
    }

    /**
     * Peform a two-sided t-test based on the given mean, variance and reference value
     *
     * This test analyzes whether the expected value of a random variable matches a
     * certain known value. When there is significant statistical "evidence"
     * against this hypothesis, the test fails.
     *
     * This is useful in checking whether a Monte Carlo method method converges
     * against the right value. Because statistical tests are able to handle the
     * inherent noise of these methods, they can be used to construct statistical
     * test suites not unlike the traditional unit tests used in software engineering.
     *
     * \param mean
     *   Estimated mean of the statistical estimator
     *
     * \param variance
     *   Estimated variance of the statistical estimator
     *
     * \param sampleCount
     *   Number of samples used to estimate \c mean and \c variance
     *
     * \param reference
     *   A known reference value ("true mean")
     *
     * \param significanceLevel
     *   The null hypothesis will be rejected when the associated
     *   p-value is below the significance level specified here.
     *
     * \param numTests
     *   Specifies the total number of tests that will be executed. If greater than one,
     *   the Sidak correction will be applied to the significance level. This is because
     *   by conducting multiple independent hypothesis tests in sequence, the probability
     *   of a failure increases accordingly.
     *
     * \return
     *   A pair of values containing the test result (success: \c true and failure: \c false)
     *   and a descriptive string
     */
    inline std::pair<bool, std::string>
    students_t_test(double mean, double variance, double reference,
                    int sampleCount, double significanceLevel, int numTests) {
        std::ostringstream oss;

        /* Compute the t statistic */
        double t = std::abs(mean - reference) * std::sqrt(sampleCount / std::max(variance, 1e-5));

        /* Determine the degrees of freedom, and instantiate a matching distribution object */
        int dof = sampleCount - 1;

        oss << "Sample mean = " << mean << " (reference value = " << reference << ")" << std::endl;
        oss << "Sample variance = " << variance << std::endl;
        oss << "t-statistic = " << t << " (d.o.f. = " << dof << ")" << std::endl;

        /* Compute the p-value */
        double pval = 2 * (1 - students_t_cdf(t, dof));

        /* Apply the Sidak correction term, since we'll be conducting multiple independent
           hypothesis tests. This accounts for the fact that the probability of a failure
           increases quickly when several hypothesis tests are run in sequence. */
        double alpha = 1.0 - std::pow(1.0 - significanceLevel, 1.0 / numTests);

        bool result = false;
        if (pval < alpha) {
            oss << "***** Rejected ***** the null hypothesis (p-value = " << pval << ", "
                   "significance level = " << alpha << ")" << std::endl;
        } else {
            oss << "Accepted the null hypothesis (p-value = " << pval << ", "
                   "significance level = " << alpha << ")" << std::endl;
            result = true;
        }
        return std::make_pair(result, oss.str());
    }
}; /* namespace hypothesis */
