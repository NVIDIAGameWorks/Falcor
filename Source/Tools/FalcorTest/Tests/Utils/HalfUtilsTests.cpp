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
#include "Testing/UnitTest.h"

/** Notes on IEEE 754 fp16 floating-point representation:

    1 sign bit, 5 exponent bits, 10 mantissa bits.

    If exponent is 11111: mantissa==0 is +-inf, mantissa!=0 it's NaN.
    If exponent is 00000: mantissa==0 it's +-0, mantissa!=0 it's a denorm with exponent 2^-14.
    Else normalized numbers with exponent offset by 15: value = sign x 2^(exp-15) x 1.mantissa.

    Finite positive fp16 numbers are encoded in [0x0000, 0x7c00). In signed decimal [0,31744).
    The fp16 values are in strictly increasing order from 0.0 to 65504.0.

    Finite negative fp16 numbers are encoded in [0x8000, 0xfc00). In signed decimal [-32768, -1024).
    The fp16 values are in strictly decreasing order from -0.0 to -65504.0.

    See also https://en.wikipedia.org/wiki/Half-precision_floating-point_format
*/

// TODO: Replace float/half conversion with platform-independent alternative that
// allows configuring the rounding modes to match what we detect on the GPU.

namespace Falcor
{
    namespace
    {
        /** Converts a finite fp32 number to fp16, rounding down to the nearest representable number.
        */
        uint16_t f32tof16_roundDown(float value)
        {
            uint16_t h = f32tof16(value);
            float res = f16tof32(h);
            if (res > value)
            {
                // Result was rounded up.
                // Depending on the value, the next smaller fp16 number is given by:
                // res < 0: +1 gives the next smaller
                // res > 0: -1 gives the next smaller
                // res == +-0: next smaller is 0x8001
                if (res < 0.f) h++;
                else if (res > 0.f) h--;
                else h = 0x8001;
            }
            return h;
        }

        /** Converts a finite fp32 number to fp16, rounding up to the nearest representable number.
        */
        uint16_t f32tof16_roundUp(float value)
        {
            uint16_t h = f32tof16(value);
            float res = f16tof32(h);
            if (res < value)
            {
                // Result was rounded down.
                // Depending on the value, the next larger fp16 number is given by:
                // res < 0: -1 gives the next larger
                // res > 0: +1 gives the next larger
                // res == +-0: next larger is 0x0001
                if (res < 0.f) h--;
                else if (res > 0.f) h++;
                else h = 0x0001;
            }
            return h;
        }

        /** Returns true if fp32 number can be exactly represented in fp16.
        */
        bool isExactFP16(float v) { return f16tof32(f32tof16(v)) == v; }

        /** Generates test data to verify round-to-nearest-even in fp32->fp16 conversion.
        */
        void generateFP16RNETestData(std::vector<float>& input, std::vector<float>& expected)
        {
            {
                float a = 0x1.0040p0f; // 1.0000 0000 01 00 0000 = 1 +  64/65536 (exactly representable)
                float b = 0x1.005fp0f; // 1.0000 0000 01 01 1111 = 1 +  95/65536 (slightly under)
                float c = 0x1.0060p0f; // 1.0000 0000 01 10 0000 = 1 +  96/65536 (half-way)
                float d = 0x1.0061p0f; // 1.0000 0000 01 10 0001 = 1 +  97/65536 (slightly over)
                float e = 0x1.0080p0f; // 1.0000 0000 10 00 0000 = 1 + 128/65536 (exactly representable)

                input.push_back(a);
                input.push_back(b);
                input.push_back(c);
                input.push_back(d);
                input.push_back(e);

                // Expected result using round-to-nearest even.
                expected.push_back(a);
                expected.push_back(a);
                expected.push_back(e); // Rounded up to nearest even mantissa
                expected.push_back(e);
                expected.push_back(e);
            }
            {
                float a = 0x1.0080p0f; // 1.0000 0000 1000 0000 = 1 + 128/65536 (exactly representable)
                float b = 0x1.00dfp0f; // 1.0000 0000 1001 1111 = 1 + 223/65536 (slightly under)
                float c = 0x1.00e0p0f; // 1.0000 0000 1110 0000 = 1 + 224/65536 (half-way)
                float d = 0x1.00e1p0f; // 1.0000 0000 1110 0001 = 1 + 225/65536 (slightly over)
                float e = 0x1.0100p0f; // 1.0000 0001 0000 0000 = 1 + 256/65536 (exactly representable)

                input.push_back(a);
                input.push_back(b);
                input.push_back(c);
                input.push_back(d);
                input.push_back(e);

                // Expected result using round-to-nearest even.
                expected.push_back(a);
                expected.push_back(a);
                expected.push_back(a); // Rounded down to nearest even mantissa
                expected.push_back(e);
                expected.push_back(e);
            }
        }

        std::vector<uint32_t> generateAllFiniteFP16()
        {
            std::vector<uint32_t> data;

            // Loop over all finite numbers in fp16.
            for (uint32_t i = 0; i < 0xfc00; i++)
            {
                if (i >= 0x7c00 && i < 0x8000) continue;    // Skip special values (inf, nan).
                assert(i <= 0xffff);
                data.push_back(i);
            }
            return data;
        }

        std::vector<float> generateFP16TestData(UnitTestContext& ctx)
        {
            std::vector<float> data;

            // Loop over all finite numbers in fp16.
            // Test that the exact number is unmodified by fp16 rounding.
            // Test that the number +- epsilon rounds correctly.
            for (uint16_t i = 0; i < 0xfc00; i++)
            {
                if (i >= 0x7c00 && i < 0x8000) continue;    // Skip special values (inf, nan).
                const float exact = f16tof32(i);

                // Compute numbers that are lighly smaller/larger than the exact fp32 value.
                float x = exact * (1.f + FLT_EPSILON);
                if (x != 0.f) EXPECT_NE(exact, x);

                float y = exact * (1.f - FLT_EPSILON);
                if (x != 0.f) EXPECT_NE(exact, y);

                // Store test values.
                data.push_back(exact);
                data.push_back(x);
                data.push_back(y);
            }

            return data;
        }
    }

    // This test currently fails due to difference in rounding modes for f32tof16() between CPU and GPU.
    // TODO: Currently disabled until we figure out the rounding modes and have a matching CPU library.
    GPU_TEST(FP32ToFP16Conversion, "Disabled due to lacking fp16 library (#391)")
    {
        std::vector<float> testData = generateFP16TestData(ctx);

        ctx.createProgram("Tests/Utils/HalfUtilsTests.cs.slang", "testFP32ToFP16");
        ctx.allocateStructuredBuffer("inputFloat", (uint32_t)testData.size(), testData.data(), testData.size() * sizeof(decltype(testData)::value_type));
        ctx.allocateStructuredBuffer("resultUint", (uint32_t)testData.size());
        ctx["CB"]["testSize"] = (uint32_t)testData.size();
        ctx.runProgram((uint32_t)testData.size(), 1, 1);

        // Verify results.
        const uint32_t* result = ctx.mapBuffer<const uint32_t>("resultUint");
        for (size_t i = 0; i < testData.size(); i++)
        {
            const float v = testData[i];
            EXPECT_EQ(result[i], (uint32_t)f32tof16(v)) << "v = " << v << " (i = " << i << ")";
        }
        ctx.unmapBuffer("resultUint");
    }

    GPU_TEST(FP16ToFP32Conversion)
    {
        std::vector<uint32_t> testData = generateAllFiniteFP16();

        ctx.createProgram("Tests/Utils/HalfUtilsTests.cs.slang", "testFP16ToFP32");
        ctx.allocateStructuredBuffer("inputUint", uint32_t(testData.size()), testData.data(), testData.size() * sizeof(decltype(testData)::value_type));
        ctx.allocateStructuredBuffer("resultFloat", uint32_t(testData.size()));
        ctx["CB"]["testSize"] = (uint32_t)testData.size();
        ctx.runProgram((uint32_t)testData.size(), 1, 1);

        // Verify results.
        const float* result = ctx.mapBuffer<const float>("resultFloat");
        for (size_t i = 1000; i < testData.size(); i++)
        {
            const uint32_t v = testData[i];
            EXPECT_EQ(result[i], f16tof32(v)) << "v = " << v << " (i = " << i << ")";
        }
        ctx.unmapBuffer("resultFloat");
    }

    /** Test our CPU-side functions for f32tof16 conversion with conservative rounding.
    */
    CPU_TEST(FP32ToFP16ConservativeRoundingCPU)
    {
        // Test assumptions on fp16 encoding.
        EXPECT_EQ(f16tof32(0x0000), 0.f);
        EXPECT_EQ(f16tof32(0x8000), -0.f);
        EXPECT_EQ(f16tof32(0x7c00), std::numeric_limits<float>::infinity());
        EXPECT_EQ(f16tof32(0xfc00), -std::numeric_limits<float>::infinity());

        // Test f32->f16 rounding functions on the CPU.
        std::vector<float> testData = generateFP16TestData(ctx);
        for (size_t i = 0; i < testData.size(); i++)
        {
            const float v = testData[i];
            if (isExactFP16(v))
            {
                // Make sure fp32 numbers exactly representable in fp16 are unmodified by rounding.
                EXPECT_EQ(f16tof32(f32tof16_roundUp(v)), v) << "i = " << i;
                EXPECT_EQ(f16tof32(f32tof16_roundDown(v)), v) << "i = " << i;
            }
            else
            {
                // Make sure fp32 numbers not-exactly representably in fp16 are conservatively rounded.
                EXPECT_GE(f16tof32(f32tof16_roundUp(v)), v) << "i = " << i;
                EXPECT_LE(f16tof32(f32tof16_roundDown(v)), v) << "i = " << i;
            }
        }
    }

    /** Test our GPU-side utils for f32tof16 conversion with conservative rounding.
        The test is written so that the conversion to fp16 is done on the GPU and the conversion
        back to fp32 on the CPU, to avoid shader compiler optimizations for interfering with the results.
    */
    GPU_TEST(FP32ToFP16ConservativeRoundingGPU)
    {
        std::vector<float> testData = generateFP16TestData(ctx);

        ctx.createProgram("Tests/Utils/HalfUtilsTests.cs.slang", "testFP32ToFP16ConservativeRounding");
        ctx.allocateStructuredBuffer("inputFloat", uint32_t(testData.size()), testData.data(), testData.size() * sizeof(decltype(testData)::value_type));
        ctx.allocateStructuredBuffer("resultUint", uint32_t(testData.size() * 2));
        ctx["CB"]["testSize"] = (uint32_t)testData.size();
        ctx.runProgram((uint32_t)testData.size(), 1, 1);

        // Verify results.
        const uint32_t* result = ctx.mapBuffer<const uint32_t>("resultUint");

        for (size_t i = 0; i < testData.size(); i++)
        {
            const float v = testData[i];
            if (isExactFP16(v))
            {
                // Make sure fp32 numbers exactly representable in fp16 are unmodified by rounding.
                EXPECT_EQ(f16tof32(result[2 * i + 0]), v) << "i = " << i;
                EXPECT_EQ(f16tof32(result[2 * i + 1]), v) << "i = " << i;
            }
            else
            {
                // Make sure fp32 numbers not-exactly representably in fp16 are conservatively rounded.
                EXPECT_GE(f16tof32(result[2 * i + 0]), v) << "i = " << i;
                EXPECT_LE(f16tof32(result[2 * i + 1]), v) << "i = " << i;
            }
        }

        ctx.unmapBuffer("resultUint");
    }

    // TODO: Currently disabled until we figure out the rounding modes and have a matching CPU library. See #391.
    // TODO: Look into the spec (is it even strictly spec'ed in HLSL?) and add utility function to detect the mode used.
    GPU_TEST(FP16RoundingModeGPU, "Disabled due to lacking fp16 library (#391)")
    {
        std::vector<float> input, expected;
        generateFP16RNETestData(input, expected);

        // TODO: The precise flag does not seem to be respected on pre-SM6.2 for this shader
        // The computation of the quantized value using 'y = f16tof32(f32tof16(x))' gets optimized to 'y = x' in the shader, despite the global precise flag.
        ctx.createProgram("Tests/Utils/HalfUtilsTests.cs.slang", "testFP16RoundingMode", Program::DefineList(), Shader::CompilerFlags::FloatingPointModePrecise, "6_2");
        ctx.allocateStructuredBuffer("inputFloat", (uint32_t)input.size(), input.data(), input.size() * sizeof(decltype(input)::value_type));
        ctx.allocateStructuredBuffer("resultFloat", (uint32_t)expected.size());
        ctx["CB"]["testSize"] = (uint32_t)input.size();
        ctx.runProgram((uint32_t)input.size(), 1, 1);

        // Verify results.
        const float* result = ctx.mapBuffer<const float>("resultFloat");

        for (size_t i = 0; i < expected.size(); i++)
        {
            float v = result[i];
            EXPECT_EQ(result[i], expected[i]) << "i = " << i;
        }

        ctx.unmapBuffer("resultFloat");
    }
}
