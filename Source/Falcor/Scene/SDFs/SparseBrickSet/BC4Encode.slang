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
#include "Utils/Math/MathConstants.slangh"

struct Codebook
{
    int codes[8];
};

struct BlockIndices
{
    uint indices[16];
};

void fixRange(inout int minValue, inout int maxValue, int steps)
{
    if (maxValue - minValue < steps)
    {
        maxValue = min(minValue + steps, 127);
        minValue = maxValue - minValue < steps ? max(-128, maxValue - steps) : minValue;
    }
}

void rowRange(int4 row, inout int min5, inout int max5, inout int min7, inout int max7)
{
    min7 = min(min(min(min(row.x, row.y), row.z), row.w), min7);
    max7 = max(max(max(max(row.x, row.y), row.z), row.w), max7);

    min5 = row.x != -128 && row.x < min5 ? row.x : min5;
    min5 = row.y != -128 && row.y < min5 ? row.y : min5;
    min5 = row.z != -128 && row.z < min5 ? row.z : min5;
    min5 = row.w != -128 && row.w < min5 ? row.w : min5;

    max5 = row.x != 127 && row.x > max5 ? row.x : max5;
    max5 = row.y != 127 && row.y > max5 ? row.y : max5;
    max5 = row.z != 127 && row.z > max5 ? row.z : max5;
    max5 = row.w != 127 && row.w > max5 ? row.w : max5;
}

int fitCodes(int4x4 block, Codebook codebook, out BlockIndices blockIndices)
{
    // Fit each alpha value to the codebook.
    int err = 0;
    for (int i = 0; i < 16; ++i)
    {
        // Find the least error and corresponding index.
        int value = block[i / 4][i % 4];
        int least = INT32_MAX;
        uint index = 0;
        for (uint j = 0; j < 8; ++j)
        {
            // Get the squared error from this code.
            int dist = value - codebook.codes[j];
            dist *= dist;

            // Compare with the best so far.
            if (dist < least)
            {
                least = dist;
                index = j;
            }
        }

        // Save this index and accumulate the error.
        blockIndices.indices[i] = index;
        err += least;
    }

    // Return the total error.
    return err;
}

uint64_t writeAlphaBlock(int alpha0, int alpha1, BlockIndices blockIndices)
{
    uint64_t compressedBlock = 0;

    // Write the first two bytes.
    compressedBlock |= uint64_t(alpha0 & 0xff);
    compressedBlock |= uint64_t(alpha1 & 0xff) << 8;

    // Pack the indices with 3 bits each.
    for (int i = 0; i < 16; ++i)
    {
        int index = blockIndices.indices[i];
        compressedBlock |= uint64_t(index & 0x7) << (3 * (i % 8) + 24 * (i / 8) + 16);
    }

    return compressedBlock;
}

uint64_t writeAlphaBlock5(int alpha0, int alpha1, BlockIndices blockIndices)
{
    // Check the relative values of the endpoints.
    if (alpha0 > alpha1)
    {
        // Swap the indices
        BlockIndices swappedBlockIndices;
        for (int i = 0; i < 16; ++i)
        {
            uint index = blockIndices.indices[i];
            if (index == 0)         swappedBlockIndices.indices[i] = 1;
            else if (index == 1)    swappedBlockIndices.indices[i] = 0;
            else if (index <= 5)    swappedBlockIndices.indices[i] = 7 - index;
            else                    swappedBlockIndices.indices[i] = index;
        }

        // Write the block.
        return writeAlphaBlock(alpha1, alpha0, swappedBlockIndices);
    }
    else
    {
        // Write the block.
        return writeAlphaBlock(alpha0, alpha1, blockIndices);
    }
}

uint64_t writeAlphaBlock7(int alpha0, int alpha1, BlockIndices blockIndices)
{
    // Check the relative values of the endpoints.
    if (alpha0 < alpha1)
    {
        // Swap the indices.
        BlockIndices swappedBlockIndices;
        for (int i = 0; i < 16; ++i)
        {
            uint index = blockIndices.indices[i];
            if (index == 0)         swappedBlockIndices.indices[i] = 1;
            else if (index == 1)    swappedBlockIndices.indices[i] = 0;
            else                    swappedBlockIndices.indices[i] = 9 - index;
        }

        // Write the block.
        return writeAlphaBlock(alpha1, alpha0, swappedBlockIndices);
    }
    else
    {
        // Write the block.
        return writeAlphaBlock(alpha0, alpha1, blockIndices);
    }
}

/** Compresses a 4x4 block of 8-bit snorm values.
*/
uint2 compressBlock(int4x4 block)
{
    // Get the range for 5-alpha and 7-alpha interpolation.
    int min5 = 127;
    int max5 = -128;
    int min7 = 127;
    int max7 = -128;

    rowRange(block[0], min5, max5, min7, max7);
    rowRange(block[1], min5, max5, min7, max7);
    rowRange(block[2], min5, max5, min7, max7);
    rowRange(block[3], min5, max5, min7, max7);

    min5 = min(min5, max5);
    min7 = min(min7, max7);

    // Fix the range to be the minimum in each case.
    fixRange(min5, max5, 5);
    fixRange(min7, max7, 7);

    // Set up the 5-alpha code book.
    Codebook codebook5;
    codebook5.codes[0] = min5;
    codebook5.codes[1] = max5;
    for (int i = 1; i < 5; ++i)
        codebook5.codes[1 + i] = (((5 - i) * min5 + i * max5) / 5);
    codebook5.codes[6] = -128;
    codebook5.codes[7] = 127;

    // Set up the 7-alpha code book.
    Codebook codebook7;
    codebook7.codes[0] = min7;
    codebook7.codes[1] = max7;
    for (int i = 1; i < 7; ++i)
        codebook7.codes[1 + i] = (((7 - i) * min7 + i * max7) / 7);

    // Fit the data to both code books.
    BlockIndices blockIndices5;
    BlockIndices blockIndices7;
    int err5 = fitCodes(block, codebook5, blockIndices5);
    int err7 = fitCodes(block, codebook7, blockIndices7);

    // Return the block with least error.
    uint64_t compressedBlock = err5 <= err7 ? writeAlphaBlock5(min5, max5, blockIndices5) : writeAlphaBlock7(min7, max7, blockIndices7);
    return uint2(compressedBlock & 0xffffffff, compressedBlock >> 32);
}
