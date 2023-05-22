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
#pragma once
#include "Core/Macros.h"
#include "Core/API/Formats.h"
#include "Core/API/ResourceViews.h"
#include "Core/Pass/ComputePass.h"
#include <memory>

namespace Falcor
{
class RenderContext;

/**
 * Image processing utilities.
 */
class FALCOR_API ImageProcessing
{
public:
    /// Constructor.
    ImageProcessing(ref<Device> pDevice);

    /**
     * Copy single mip level and color channel from source to destination.
     * The views must have matching dimension and format type (float vs integer).
     * The source value is written to all color channels of the destination.
     * The function throws if the requirements are not fulfilled.
     * @param[in] pRenderContxt The render context.
     * @param[in] pSrc Resource view for source texture.
     * @param[in] pDst Unordered access view for destination texture.
     * @param[in] srcMask Mask specifying which source color channel to copy. Must be a single channel.
     */
    void copyColorChannel(
        RenderContext* pRenderContxt,
        const ref<ShaderResourceView>& pSrc,
        const ref<UnorderedAccessView>& pDst,
        const TextureChannelFlags srcMask
    );

private:
    ref<Device> mpDevice;
    ref<ComputePass> mpCopyFloatPass;
    ref<ComputePass> mpCopyIntPass;
};
} // namespace Falcor
