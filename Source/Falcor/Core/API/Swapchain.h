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
#include "fwd.h"
#include "Texture.h"
#include "Formats.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include <slang-gfx.h>

namespace Falcor
{
class FALCOR_API Swapchain : public Object
{
    FALCOR_OBJECT(Swapchain)
public:
    struct Desc
    {
        ResourceFormat format{ResourceFormat::Unknown};
        uint32_t width{0};
        uint32_t height{0};
        uint32_t imageCount{3};
        bool enableVSync{false};
    };

    /**
     * Constructor. Throws an exception if creation fails.
     * @param desc Swapchain description.
     * @param windowHandle Handle of window to create swapchain for.
     */
    Swapchain(ref<Device> pDevice, const Desc& desc, WindowHandle windowHandle);

    const Desc& getDesc() const { return mDesc; }

    /// Returns the back buffer image at `index`.
    const ref<Texture>& getImage(uint32_t index) const;

    /// Present the next image in the swapchain.
    void present();

    /// Returns the index of next back buffer image that will be presented in the next
    /// `present` call. If the swapchain is invalid/out-of-date, this method returns -1.
    int acquireNextImage();

    /// Resizes the back buffers of this swapchain. All render target views and framebuffers
    /// referencing the back buffer images must be freed before calling this method.
    /// Note: This method calls Device::wait().
    void resize(uint32_t width, uint32_t height);

    /// Check if the window is occluded.
    bool isOccluded();

    /// Toggle full screen mode.
    void setFullScreenMode(bool mode);

    gfx::ISwapchain* getGfxSwapchain() const { return mGfxSwapchain; }

private:
    void prepareImages();

    ref<Device> mpDevice;
    Desc mDesc;
    Slang::ComPtr<gfx::ISwapchain> mGfxSwapchain;
    std::vector<ref<Texture>> mImages;
};
} // namespace Falcor
