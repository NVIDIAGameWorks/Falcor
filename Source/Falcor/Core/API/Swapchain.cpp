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
#include "Swapchain.h"
#include "Device.h"
#include "GFXAPI.h"
#include "GFXHelpers.h"

namespace Falcor
{

Swapchain::Swapchain(ref<Device> pDevice, const Desc& desc, WindowHandle windowHandle) : mpDevice(pDevice), mDesc(desc)
{
    FALCOR_ASSERT(mpDevice);

    FALCOR_CHECK(desc.format != ResourceFormat::Unknown, "Invalid format");
    FALCOR_CHECK(desc.width > 0, "Invalid width");
    FALCOR_CHECK(desc.height > 0, "Invalid height");
    FALCOR_CHECK(desc.imageCount > 0, "Invalid image count");

    gfx::ISwapchain::Desc gfxDesc = {};
    gfxDesc.format = getGFXFormat(desc.format);
    gfxDesc.width = desc.width;
    gfxDesc.height = desc.height;
    gfxDesc.imageCount = desc.imageCount;
    gfxDesc.enableVSync = desc.enableVSync;
    gfxDesc.queue = mpDevice->getGfxCommandQueue();
#if FALCOR_WINDOWS
    gfx::WindowHandle gfxWindowHandle = gfx::WindowHandle::FromHwnd(windowHandle);
#elif FALCOR_LINUX
    gfx::WindowHandle gfxWindowHandle = gfx::WindowHandle::FromXWindow(windowHandle.pDisplay, windowHandle.window);
#endif
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createSwapchain(gfxDesc, gfxWindowHandle, mGfxSwapchain.writeRef()));

    prepareImages();
}

const ref<Texture>& Swapchain::getImage(uint32_t index) const
{
    FALCOR_ASSERT(index <= mImages.size());
    return mImages[index];
}

void Swapchain::present()
{
    FALCOR_GFX_CALL(mGfxSwapchain->present());
}

int Swapchain::acquireNextImage()
{
    return mGfxSwapchain->acquireNextImage();
}

void Swapchain::resize(uint32_t width, uint32_t height)
{
    FALCOR_CHECK(width > 0, "Invalid width");
    FALCOR_CHECK(height > 0, "Invalid height");

    mImages.clear();
    mpDevice->wait();
    FALCOR_GFX_CALL(mGfxSwapchain->resize(width, height));
    prepareImages();
}

bool Swapchain::isOccluded()
{
    return mGfxSwapchain->isOccluded();
}

void Swapchain::setFullScreenMode(bool mode)
{
    FALCOR_GFX_CALL(mGfxSwapchain->setFullScreenMode(mode));
}

void Swapchain::prepareImages()
{
    for (uint32_t i = 0; i < mDesc.imageCount; ++i)
    {
        Slang::ComPtr<gfx::ITextureResource> resource;
        FALCOR_GFX_CALL(mGfxSwapchain->getImage(i, resource.writeRef()));
        mImages.push_back(mpDevice->createTextureFromResource(
            resource,
            Texture::Type::Texture2D,
            mDesc.format,
            mDesc.width,
            mDesc.height,
            1,
            1,
            1,
            1,
            ResourceBindFlags::RenderTarget,
            Resource::State::Undefined
        ));
    }
}

} // namespace Falcor
