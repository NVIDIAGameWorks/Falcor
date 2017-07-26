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

#pragma once
#include <memory>
#include "API/FBO.h"
#include "API/Texture.h"
#include "glm/vec2.hpp"
#include <vector>
#include "OpenVR/VRDisplay.h"

namespace Falcor
{
    class VrFbo
    {
    public:
        using UniquePtr = std::unique_ptr<VrFbo>;
        /** Create a new VrFbo. It will create array resources for color and depth. It will also create views into each array-slice
            \param[in] desc FBO description
            \param[in] width The width of the FBO. Optional, by default will use the HMD render-target size
            \param[in] height The height of the FBO. Optional, by default will use the HMD render-target size
        */
        static UniquePtr create(const Fbo::Desc& desc, uint32_t width = 0, uint32_t height = 0);

        /** Submit the color target into the HMD
        */
        void submitToHmd(RenderContext* pRenderCtx) const;

        /** Get the FBO
        */
        Fbo::SharedPtr getFbo() const { return mpFbo; }

        /** Get the resource view to an eye's resource view
        */
        Texture::SharedPtr getEyeResourceView(VRDisplay::Eye eye) const { return (eye == VRDisplay::Eye::Left) ? mpLeftView : mpRightView; }

    private:
        Fbo::SharedPtr mpFbo;
        Texture::SharedPtr mpLeftView;
        Texture::SharedPtr mpRightView;
    };
}