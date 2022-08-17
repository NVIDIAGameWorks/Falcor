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
#pragma once
#include <algorithm>
#include <cstdint>
#include <climits>

// this file exposes a single function, CompressAlphaDxt5, which encodes a 4x4 set of uint8 alpha values into a single 64 bit BC4 encoded block
static void CompressAlphaDxt5(uint8_t* tile, void* block);

// derived from libsquish, alpha.cpp
/* -----------------------------------------------------------------------------
    Copyright (c) 2006 Simon Brown                          si@sjbrown.co.uk
    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to    deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:
    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
   -------------------------------------------------------------------------- */

static void FixRange(int& min, int& max, int steps)
{
    if (max - min < steps)
        max = std::min(min + steps, 255);
    if (max - min < steps)
        min = std::max(0, max - steps);
}

static int FitCodes(uint8_t const* tile, uint8_t const* codes, uint8_t* indices)
{
    // fit each alpha value to the codebook
    int err = 0;
    for (int i = 0; i < 16; ++i)
    {
        // find the least error and corresponding index
        int value = (int)(tile[i]);
        int least = INT_MAX;
        int index = 0;
        for (int j = 0; j < 8; ++j)
        {
            // get the squared error from this code
            int dist = (int)value - (int)codes[j];
            dist *= dist;

            // compare with the best so far
            if (dist < least)
            {
                least = dist;
                index = j;
            }
        }

        // save this index and accumulate the error
        indices[i] = (uint8_t)index;
        err += least;
    }

    // return the total error
    return err;
}

static void WriteAlphaBlock(int alpha0, int alpha1, uint8_t const* indices, void* block)
{
    uint8_t* bytes = reinterpret_cast<uint8_t*>(block);

    // write the first two bytes
    bytes[0] = (uint8_t)alpha0;
    bytes[1] = (uint8_t)alpha1;

    // pack the indices with 3 bits each
    uint8_t* dest = bytes + 2;
    uint8_t const* src = indices;
    for (int i = 0; i < 2; ++i)
    {
        // pack 8 3-bit values
        int value = 0;
        for (int j = 0; j < 8; ++j)
        {
            int index = *src++;
            value |= (index << 3 * j);
        }

        // store in 3 bytes
        for (int j = 0; j < 3; ++j)
        {
            int byte = (value >> 8 * j) & 0xff;
            *dest++ = (uint8_t)byte;
        }
    }
}

static void WriteAlphaBlock5(int alpha0, int alpha1, uint8_t const* indices, void* block)
{
    // check the relative values of the endpoints
    if (alpha0 > alpha1)
    {
        // swap the indices
        uint8_t swapped[16];
        for (int i = 0; i < 16; ++i)
        {
            uint8_t index = indices[i];
            if (index == 0)
                swapped[i] = 1;
            else if (index == 1)
                swapped[i] = 0;
            else if (index <= 5)
                swapped[i] = 7 - index;
            else
                swapped[i] = index;
        }

        // write the block
        WriteAlphaBlock(alpha1, alpha0, swapped, block);
    }
    else
    {
        // write the block
        WriteAlphaBlock(alpha0, alpha1, indices, block);
    }
}

static void WriteAlphaBlock7(int alpha0, int alpha1, uint8_t const* indices, void* block)
{
    // check the relative values of the endpoints
    if (alpha0 < alpha1)
    {
        // swap the indices
        uint8_t swapped[16];
        for (int i = 0; i < 16; ++i)
        {
            uint8_t index = indices[i];
            if (index == 0)
                swapped[i] = 1;
            else if (index == 1)
                swapped[i] = 0;
            else
                swapped[i] = 9 - index;
        }

        // write the block
        WriteAlphaBlock(alpha1, alpha0, swapped, block);
    }
    else
    {
        // write the block
        WriteAlphaBlock(alpha0, alpha1, indices, block);
    }
}


static void CompressAlphaDxt5(uint8_t* tile, void* block)
{
    // get the range for 5-alpha and 7-alpha interpolation
    int min5 = 255;
    int max5 = 0;
    int min7 = 255;
    int max7 = 0;
    for (int i = 0; i < 16; ++i)
    {
        // incorporate into the min/max
        int value = (int)(tile[i]);
        if (value < min7)
            min7 = value;
        if (value > max7)
            max7 = value;
        if (value != 0 && value < min5)
            min5 = value;
        if (value != 255 && value > max5)
            max5 = value;
    }

    // handle the case that no valid range was found
    if (min5 > max5)
        min5 = max5;
    if (min7 > max7)
        min7 = max7;

    // fix the range to be the minimum in each case
    FixRange(min5, max5, 5);
    FixRange(min7, max7, 7);

    // set up the 5-alpha code book
    uint8_t codes5[8];
    codes5[0] = (uint8_t)min5;
    codes5[1] = (uint8_t)max5;
    for (int i = 1; i < 5; ++i)
        codes5[1 + i] = (uint8_t)(((5 - i) * min5 + i * max5) / 5);
    codes5[6] = 0;
    codes5[7] = 255;

    // set up the 7-alpha code book
    uint8_t codes7[8];
    codes7[0] = (uint8_t)min7;
    codes7[1] = (uint8_t)max7;
    for (int i = 1; i < 7; ++i)
        codes7[1 + i] = (uint8_t)(((7 - i) * min7 + i * max7) / 7);

    // fit the data to both code books
    uint8_t indices5[16];
    uint8_t indices7[16];
    int err5 = FitCodes(tile, codes5, indices5);
    int err7 = FitCodes(tile, codes7, indices7);

    // save the block with least error
    if (err5 <= err7)
        WriteAlphaBlock5(min5, max5, indices5, block);
    else
        WriteAlphaBlock7(min7, max7, indices7, block);
}

