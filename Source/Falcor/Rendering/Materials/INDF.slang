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
__exported import Utils.Sampling.SampleGeneratorInterface;

 /** Interface for NDFs (normal distribution functions) from microfacet theory.
 
     These are assumed to be "stretch-invariant", which means only a few methods
     need to be implemented for evaluating and sampling from the underlying
     distribution of slopes.
     In particular, it is sufficient to characterize the slope distribution in
     a "standard configuration", meaning it is isotropic with a fixed unit roughness
     of alpha = 1.0. This then allows derivation of more general anisotropic variants
     and accurate Smith masking functions.
 
     See "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs" by
     Eric Heitz 2014 for more details.
 */
 interface INDF
 {
     /** Evaluates the slope distribution in the standard configuration.
 
         \param[in] slope 2D slope.
         \return P22_std(slope).
     */
     float evalP22Std(float2 slope);
 
     /** Evaluates the projected area (sigma) of the heighfield in the standard configuration.
 
         \param[in] dir Evaluated direction in local space, stretched into the standard configuration.
         \return sigma_std(dir).
     */
     float evalSigmaStd(float3 dir);
 
     /** Samples a slope proportionally to P22_std(slope).
 
         \param[in,out] sg Sample generator.
         \return Sampled slope.
     */
     float2 sampleP22Std<S : ISampleGenerator>(inout S sg);
     
     /** Samples a slope proportionally to only the visible P22_std(slope).
 
         \param[in] wi Incident direction, stretched into the standard configuration.
         \param[in,out] sg Sample generator.
         \return Sampled slope.
     */
     float2 sampleVisibleP22Std<S : ISampleGenerator>(float3 wi, inout S sg);
 }
 