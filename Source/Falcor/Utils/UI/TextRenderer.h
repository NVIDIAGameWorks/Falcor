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
#include "Core/API/fwd.h"
#include "Core/API/Buffer.h"
#include "Core/API/FBO.h"
#include "Core/Pass/RasterPass.h"
#include "Utils/Math/Vector.h"

namespace Falcor
{
class Font;

/**
 * Class that renders text into the screen.
 */
class FALCOR_API TextRenderer
{
public:
    enum class Flags
    {
        None = 0x0,
        Shadowed = 0x1
    };

    TextRenderer(ref<Device> pDevice);
    ~TextRenderer();

    /**
     * Render text
     * @param[in] pRenderContext A render-context which will be used to dispatch the draw
     * @param[in] text The text to draw. It can include newlines, tabs, carriage returns and regular ASCII characters.
     * @param[in] pDstFbo The target FBO
     * @param[in] pos Text position
     */
    void render(RenderContext* pRenderContext, const std::string& text, const ref<Fbo>& pDstFbo, float2 pos);

    /**
     * Returns the color of the text being rendered
     * @return current color The text color
     */
    const float3& getColor() const { return mColor; }

    /**
     * Set the color of the text being rendered
     * @param[in] color The text color
     */
    void setColor(const float3& color) { mColor = color; }

    /**
     * Get the active flags
     */
    Flags getFlags() const { return mFlags; }

    /**
     * Set the flags
     */
    void setFlags(Flags flags) { mFlags = flags; }

private:
    void setCbData(const ref<Fbo>& pDstFbo);
    void renderText(RenderContext* pRenderContext, const std::string& text, const ref<Fbo>& pDstFbo, float2 pos);

    // Use 3 rotating VAOs to avoid stalling the GPU.
    // A better way would be to upload the data using an upload heap.
    static constexpr uint32_t kVaoCount = 3;

    ref<Device> mpDevice;
    Flags mFlags = Flags::Shadowed;
    float3 mColor = float3(1.f);
    ref<Vao> mpVaos[kVaoCount];
    uint32_t mVaoIndex = 0;
    ref<RasterPass> mpPass;
    std::unique_ptr<Font> mpFont;
};

FALCOR_ENUM_CLASS_OPERATORS(TextRenderer::Flags);
} // namespace Falcor
