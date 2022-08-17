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
#include "Utils/Sampling/SampleGeneratorType.slangh"

__exported import Utils.Sampling.SampleGeneratorInterface;

/** The host sets the SAMPLE_GENERATOR_TYPE define to select sample generator.

    This code typedefs the chosen type to the type 'SampleGenerator'.
    All sample generators adheres to the same interface, but note that the
    size of the 'SampleGenerator' type may vary depending on their state size.

    If SAMPLE_GENERATOR_TYPE is not defined, a compile-time error is printed.

    The 'SampleGenerator.Padded' type holds a SampleGenerator plus additional
    padding to make the struct a multiple of 16B.
*/

#if defined(SAMPLE_GENERATOR_TYPE) && SAMPLE_GENERATOR_TYPE == SAMPLE_GENERATOR_TINY_UNIFORM
    import Utils.Sampling.TinyUniformSampleGenerator;
    typedef TinyUniformSampleGenerator SampleGenerator;
#elif defined(SAMPLE_GENERATOR_TYPE) && SAMPLE_GENERATOR_TYPE == SAMPLE_GENERATOR_UNIFORM
    import Utils.Sampling.UniformSampleGenerator;
    typedef UniformSampleGenerator SampleGenerator;
#endif

