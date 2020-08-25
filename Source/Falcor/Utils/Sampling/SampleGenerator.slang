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
#include "Utils/Sampling/SampleGeneratorType.slangh"

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

/** Convenience functions for generating 1D/2D/3D values in the range [0,1).

    Note: These are global instead of member functions in the sample generator
    interface, as there seems to be no way in Slang currently to specify default
    implementations without duplicating the code into all classes that implement
    the interace.
*/

float sampleNext1D(inout SampleGenerator sg)
{
    // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
    // In floating-point precision this also ensures that 1.0-u != 0.0.
    uint bits = sg.next();
    return (bits >> 8) * 0x1p-24;
}

float2 sampleNext2D(inout SampleGenerator sg)
{
    float2 sample;
    // Don't use the float2 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sg);
    sample.y = sampleNext1D(sg);
    return sample;
}

float3 sampleNext3D(inout SampleGenerator sg)
{
    float3 sample;
    // Don't use the float3 initializer to ensure consistent order of evaluation.
    sample.x = sampleNext1D(sg);
    sample.y = sampleNext1D(sg);
    sample.z = sampleNext1D(sg);
    return sample;
}
