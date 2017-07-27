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
#include <string>
#include <vector>
#include "glm/vec4.hpp"
#include "API/FBO.h"
#include "API/Texture.h"

namespace Falcor
{
    /*!
    *  \addtogroup Falcor
    *  @{
    
        Helper function to create an Fbo object and the required textures.
    */

    namespace FboHelper
    {
        /** Create a color-only 2D framebuffer.
            \param[in] width Width of the render-targets.
            \param[in] height Height of the render-targets.
            \param[in] fboDesc Struct specifying the frame buffer's attachments and formats.
            \param[in] arraySize Optional. The number of array slices in the texture.
            \param[in] mipLevels Optional. The number of mip levels to create. You can use Texture#kMaxPossible to create the entire chain
        */
        Fbo::SharedPtr create2D(uint32_t width, uint32_t height, const Fbo::Desc& fboDesc, uint32_t arraySize = 1, uint32_t mipLevels = 1);

        /** Create a color-only cubemap framebuffer.
            \param[in] width width of the render-targets.
            \param[in] height height of the render-targets.
            \param[in] fboDesc Struct specifying the frame buffer's attachments and formats.
            \param[in] arraySize Optional. The number of cubes in the texture.
            \param[in] mipLevels Optional. The number of mip levels to create. You can use Texture#kMaxPossible to create the entire chain
        */
        Fbo::SharedPtr createCubemap(uint32_t width, uint32_t height, const Fbo::Desc& fboDesc, uint32_t arraySize = 1, uint32_t mipLevels = 1);
    }
}