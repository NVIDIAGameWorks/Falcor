/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#ifndef SAMPLES_PER_PIXEL
#error RenderPasses/WARDiffPathTracer/StaticParams.slang is missing defines
#endif

__exported import DiffRendering.SharedTypes;

static const uint kSamplesPerPixel = SAMPLES_PER_PIXEL;
static const uint kMaxBounces = MAX_BOUNCES;

static const DiffMode kDiffMode = DiffMode(DIFF_MODE);

static const bool kUseBSDFSampling = USE_BSDF_SAMPLING;
static const bool kUseNEE = USE_NEE;
static const bool kUseMIS = USE_MIS;

static const bool kUseWAR = USE_WAR;
static const uint kAuxSampleCount = AUX_SAMPLE_COUNT;
static const float kLog10vMFConcentration = LOG10_VMF_CONCENTRATION;
static const float kLog10vMFConcentrationScreen = LOG10_VMF_CONCENTRATION_SCREEN;
static const float kBoundaryTermBeta = BOUNDARY_TERM_BETA;
static const bool kUseAntitheticSampling = USE_ANTITHETIC_SAMPLING;
static const float kHarmonicGamma = HARMONIC_GAMMA;

static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
