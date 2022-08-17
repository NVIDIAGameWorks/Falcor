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
#include "Utils/Math/Vector.h"
#include <memory>

namespace Falcor
{
    /** This class represents a 2D table of 2D distributions.
        The basic model is that given coordinates x, y,
        we cut out a 2D `slice' from the 4D table and then
        sample it to obtain coordinates z, w.

        Within each distribution, the PDFs are linearly interpolated
        with respect to z, w. Also, slices are linearly interpolated
        from the table with respect to x, y.

        To achieve this goal, we first build the marginal/conditional
        distribution for each 2D slice of the 4D table similar to
        Pharr et al., with the only twist is that the PDF is linearly
        interpolated, i.e. the CDFs store the integral of a linearly
        interpolated PDF instead of the straight sum of the PDF.

        The actual interpolation/sampling at runtime happens on the GPU
        (see RGLCommon.slang)
    */
    class SamplableDistribution4D
    {
    public:
        SamplableDistribution4D(const float* pdf, uint4 size);

        const float* getPDF()         { return mPDF.get();         }

        const float* getMarginal()    { return mMarginal.get();    }

        const float* getConditional() { return mConditional.get(); }

    private:
        uint4 mSize;
        std::unique_ptr<float[]> mPDF;
        std::unique_ptr<float[]> mMarginal;
        std::unique_ptr<float[]> mConditional;

        void build2DSlice(int2 size, float* pdf, float* marginalCdf, float* conditionalCdf);
    };
}
