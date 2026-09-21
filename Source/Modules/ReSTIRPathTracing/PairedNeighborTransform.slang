/***************************************************************************
 # Copyright (c) 2015-26, NVIDIA CORPORATION. All rights reserved.
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
import Utils.Sampling.SampleGenerator;

StructuredBuffer<uint> pairingDeltaTextureSizes;
StructuredBuffer<uint> pairingDeltaTextures;

struct PairedNeighborTransform {
    int offsetX;
    int offsetY;
    int imageWidth;
    int imageHeight;
    bool flipX;
    bool flipY;
    bool transpose;

    __init(inout SampleGenerator sg, int imageWidth, int imageHeight)
    {
        uint r = sg.next();

        this.imageHeight = imageHeight;
        this.imageWidth = imageWidth;

        offsetX = (r & 0b00000000000000000111111111111111u) % imageWidth;
        r >>= 15;

        offsetY = (r & 0b00000000000000000011111111111111u) % imageHeight;
        r >>= 14;

        flipX = bool(r & 1);
        r >>= 1;

        flipY = bool(r & 1);
        r >>= 1;

        transpose = bool(r & 1);
    }

    void transformCoordinates(int neighborIndex, int x, int y, int imageWidth, int imageHeight, inout int out_x, inout int out_y) {
        int textureSize = pairingDeltaTextureSizes[neighborIndex];

        x = (x + offsetX) % textureSize;
        y = (y + offsetY) % textureSize;

        if (flipX) x = textureSize - 1 - x;
        if (flipY) y = textureSize - 1 - y;

        if (transpose)
        {
            int temp = x;
            x = y;
            y = temp;
        }
    
        out_x = x;
        out_y = y;
    }

    void transformDelta(int dx, int dy, inout int out_dx, inout int out_dy) {
        if (transpose)
        {
            int temp = dx;
            dx = dy;
            dy = temp;
        }
        if (flipX) dx = -dx;
        if (flipY) dy = -dy;
        out_dx = dx;
        out_dy = dy;
    }

    void unpackDelta(uint packed, inout int dx, inout int dy) {
        dx = (int)(packed & 0xFF) - 128;
        dy = (int)(packed >> 8) - 128;
    }

    uint readReuseTexture(int index, int x, int y) {
        int textureSize = pairingDeltaTextureSizes[index];
        return pairingDeltaTextures[256 * 256 * index + y * textureSize + x];
    }

    void readTransformedDelta( int neighborIndex, int x, int y, 
        inout int out_dx, inout int out_dy ) {
        int tx, ty, dx, dy;
        transformCoordinates(neighborIndex, x, y, imageWidth, imageHeight, tx, ty);
        uint packedDelta = readReuseTexture(neighborIndex, tx, ty);
        unpackDelta(packedDelta, dx, dy);
        transformDelta(dx, dy, out_dx, out_dy);
    }

    int applyDeltaPeriodic(int x, int dx, int width) {
        int r = (x + dx) % width;
        if (r < 0) r += width;
        return r;
    }
};
