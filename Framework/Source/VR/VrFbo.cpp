/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "Framework.h"
#include "VrFbo.h"
#include "OpenVR/VRSystem.h"
#include "OpenVR/VRDisplay.h"
#include "Graphics/FboHelper.h"
#include "API/Texture.h"

namespace Falcor
{
    glm::ivec2 getHmdRenderSize()
    {
        VRSystem* pVrSystem = VRSystem::instance();
        VRDisplay* pDisplay = pVrSystem->getHMD().get();
        glm::ivec2 renderSize = pDisplay->getRecommendedRenderSize();
        return renderSize;
    }

    VrFbo::UniquePtr VrFbo::create(const Fbo::Desc& desc, uint32_t width, uint32_t height)
    {
        width = (width == 0) ? getHmdRenderSize().x : width;
        height = (height == 0) ? getHmdRenderSize().y : height;

        // Create the FBO
        VrFbo::UniquePtr pVrFbo = std::make_unique<VrFbo>();
        pVrFbo->mpFbo = FboHelper::create2D(width, height, desc, 2);

        // create the textures
        // in the future we should use SRVs directly
        // or some other way to avoid copying resources

        pVrFbo->mpLeftView = Texture::create2D(width, height, desc.getColorTargetFormat(0),1,1);
        pVrFbo->mpRightView = Texture::create2D(width, height, desc.getColorTargetFormat(0),1,1);

        return pVrFbo;
    }

    void VrFbo::submitToHmd(RenderContext* pRenderCtx) const
    {
        VRSystem* pVrSystem = VRSystem::instance();

        uint32_t ltSrcSubresourceIdx = mpFbo->getColorTexture(0)->getSubresourceIndex(0, 0);
        uint32_t rtSrcSubresourceIdx = mpFbo->getColorTexture(0)->getSubresourceIndex(1, 0);

        uint32_t ltDstSubresourceIdx = mpLeftView->getSubresourceIndex(0, 0);
        uint32_t rtDstSubresourceIdx = mpRightView->getSubresourceIndex(0, 0);

        pRenderCtx->copySubresource(mpLeftView.get(),  ltDstSubresourceIdx, mpFbo->getColorTexture(0).get(), ltSrcSubresourceIdx);
        pRenderCtx->copySubresource(mpRightView.get(), rtDstSubresourceIdx, mpFbo->getColorTexture(0).get(), rtSrcSubresourceIdx);

        pVrSystem->submit(VRDisplay::Eye::Left, mpLeftView, pRenderCtx);
        pVrSystem->submit(VRDisplay::Eye::Right, mpRightView, pRenderCtx);
    }
}