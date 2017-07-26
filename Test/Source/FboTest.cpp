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
#include "FboTest.h"
#include <stdio.h>

void FboTest::addTests()
{
    addTestToList<TestDepthStencilAttach>();
    addTestToList<TestColorAttach>();
    addTestToList<TestGetWidthHeight>();
    addTestToList<TestCreate2D>();
    addTestToList<TestCreateCubemap>();
}

testing_func(FboTest, TestDepthStencilAttach)
{
    //attachDepthStencilTarget, the fx this is meant to test, calls applyDepthAttachment, which is unimplemented in d3d12
    const uint32_t numDimensions = 5;
    const uint32_t widths[numDimensions] = { 1920u, 1440u, 1280u, 1600u, 1280u };
    const uint32_t heights[numDimensions] = { 1080u, 900u, 800u, 1200u, 960u };
    const uint32_t numFormats = 4;
    //D24Unorm crashes in texture, not sure why
    const ResourceFormat formats[numFormats] = { ResourceFormat::D32Float, ResourceFormat::D16Unorm, ResourceFormat::D32FloatS8X24, ResourceFormat::D24UnormS8 };
    
    //Attaching Textures with Various properties
    Fbo::SharedPtr fbo = Fbo::create();
    for (uint32_t i = 0; i < numDimensions; ++i)
    {
        for (uint32_t j = 0; j < numDimensions; ++j)
        {
            for (uint32_t k = 0; k < 2; ++k)
            {
                //Create and attach tex
                Texture::SharedPtr tex = Texture::create2D(widths[i], heights[j], formats[k], 1u,
                    Resource::kMaxPossible, (const void*)nullptr, Resource::BindFlags::DepthStencil);
                fbo->attachDepthStencilTarget(tex);

                //Check properties
                if (fbo->getDepthStencilTexture()->getWidth() != widths[i] ||
                    fbo->getDepthStencilTexture()->getHeight() != heights[j] ||
                    fbo->getDesc().getDepthStencilFormat() != formats[k] ||
                    fbo->getDesc().isDepthStencilUav() /*should be false without uav flag on texture*/)
                {
                    return test_fail("Fbo properties are incorrect after attaching depth stencil texture");
                }
            }
        }
    }

    //Not testing UAV == True b/c not currently supported w/ depth stencil

    //Attaching nullptr
    fbo->attachDepthStencilTarget(nullptr, 0u, 0u, 0u);
    if (fbo->getDesc().getDepthStencilFormat() != ResourceFormat::Unknown ||
        fbo->getDesc().isDepthStencilUav() /*should be false from nullptr texture*/)
    {
        return test_fail("Fbo properties are incorrect after attaching depth stencil texture with nullptr");
    }

    return test_pass();
}

testing_func(FboTest, TestColorAttach)
{
    const uint32_t numFormats = 6u;
    const ResourceFormat formats[numFormats] = { ResourceFormat::R8Unorm, ResourceFormat::R16Snorm, ResourceFormat::RGBA32Float,
        ResourceFormat::RG8Uint, ResourceFormat::RG16Int, ResourceFormat::RG32Uint };
    //The H/W aspects of this function are tested in TestGetWidthHeight, const texture size should be fine
    const uint32_t width = 800u;
    const uint32_t height = 600u;
    const uint32_t numTextureTypes = 4u; //1d, 2d, 3d, cube

    uint32_t ctIndex = 0;
    Fbo::SharedPtr fbo = Fbo::create();
    //Texture Format
    //Starts at 1 to skip format::unknown
    for (uint32 i = 0; i < numFormats; ++i)
    {
        //UAV flag
        for (uint32_t j = 0; j < 2; ++j)
        {
            Resource::BindFlags flags = Resource::BindFlags::RenderTarget;
            if (j == 0)
            {
                flags |= Resource::BindFlags::UnorderedAccess;
            }

            //texture type
            for (uint32_t k = 0; k < numTextureTypes; ++k)
            {
                Texture::SharedPtr tex = nullptr;
                switch (k)
                {
                case 0: 
                    tex = Texture::create1D(width, formats[i], 1u, Resource::kMaxPossible, (const void*)nullptr, flags);
                    break;
                case 1:
                    tex = Texture::create2D(width, height, formats[i], 1u, Resource::kMaxPossible, (const void*)nullptr, flags);
                    break;
                case 2:
                    tex = Texture::create3D(width, height, 1u, formats[i], Resource::kMaxPossible, (const void*)nullptr, flags);
                    break;
                case 3:
                    tex = Texture::createCube(width, height, formats[i], 1u, Resource::kMaxPossible, (const void*)nullptr, flags);
                    break;
                default: 
                    should_not_get_here();
                }

                //wrap around ct index if necessary
                if (ctIndex >= Fbo::getMaxColorTargetCount())
                {
                    ctIndex = 0;
                }

                //attach color target
                fbo->attachColorTarget(tex, ctIndex);

                //Check properties
                //Check uav
                if ((flags & Resource::BindFlags::UnorderedAccess) != Resource::BindFlags::None)
                {
                    if (!fbo->getDesc().isColorTargetUav(ctIndex))
                    {
                        return test_fail("Fbo color target missing uav flag despite texture being created with unordered access flag");
                    }
                }
                else
                {
                    if (fbo->getDesc().isColorTargetUav(ctIndex))
                    {
                        return test_fail("Fbo color target has uav flag despite texture being created without unordered access flag");
                    }
                }

                //check format
                if (formats[i] != fbo->getDesc().getColorTargetFormat(ctIndex))
                {
                    return test_fail("Fbo color target format does not match the format of the attached texture");
                }

                ++ctIndex;
            }
        }
    }

    return test_pass();
}

testing_func(FboTest, TestGetWidthHeight)
{
    const uint32_t numResolutions = 7; //should eq max color target count - 1
    const uint32_t widths[numResolutions] = { 1920u, 1440u, 1280u, 1600u, 1280u, 800u, 400u };
    const uint32_t heights[numResolutions] = { 1080u, 900u, 800u, 1200u, 960u, 700u, 350u };

    Fbo::SharedPtr fbo = Fbo::create();
    //For each  texture resolution
    for (uint32_t i = 0; i < numResolutions; ++i)
    {
        //create texture
        Texture::SharedPtr tex = Texture::create2D(widths[i], heights[i], ResourceFormat::RGBA32Float, 1u, 
            Resource::kMaxPossible, (const void*)nullptr, Resource::BindFlags::RenderTarget);
        uint32_t curWidth = -1;
        uint32_t curHeight = -1;
        for (uint32_t j = 0; j < i + 1; ++j)
        {
            curWidth = min(curWidth, widths[i]);
            curHeight = min(curHeight, heights[i]);
            //Attach, update current dimensions based on logic in verifyAttachment
            fbo->attachColorTarget(tex, j);
        }

        //Check if dimensions match
        if (fbo->getWidth() != curWidth || fbo->getHeight() != curHeight)
        {
            return test_fail("Fbo dimensions do not match dimensions used to set");
        }
    }

    return test_pass();
}

testing_func(FboTest, TestCreate2D)
{
    if (isStressCreateCorrect(false))
    {
        return test_pass();
    }
    else
    {
        return test_fail("Fbo properties do not match properties used to create it");
    }
}

testing_func(FboTest, TestCreateCubemap)
{
    if (isStressCreateCorrect(true))
    {
        return test_pass();
    }
    else
    {
        return test_fail("Fbo properties do not match properties used to create it");
    }
}

bool FboTest::isStressCreateCorrect(bool cubemap)
{
    const uint32_t numResolutions = 6;
    const uint32_t widths[numResolutions] = { 1920u, 1440u, 1280u, 1600u, 1280u, 800u };
    const uint32_t heights[numResolutions] = { 1080u, 900u, 800u, 1200u, 960u, 700u };
    //Not too many formats, tested more thoroughly in color/depth attach
    const uint32_t numTestFormats = 2;
    const ResourceFormat resources[numTestFormats] = { ResourceFormat::RGBA32Float, ResourceFormat::RGBA32Uint };
    const ResourceFormat depths[numTestFormats] = { ResourceFormat::D16Unorm, ResourceFormat::D32Float };
    
    Fbo::Desc desc;
    //formats
    //skip past resource format 0, unknown.
    for (uint32_t i = 0; i < numTestFormats; ++i)
    {
        ResourceFormat format = resources[i];
        //bool for isUav
        for (uint32_t j = 0; j < 2; ++j)
        {
            //Depthstencil uav currently not supported
            desc.setDepthStencilTarget(depths[j], false);
            desc.setSampleCount(1u);
            //set properties for all color targets
            for (uint32_t k = 0; k < Fbo::getMaxColorTargetCount(); ++k)
            {
                 desc.setColorTarget(k, format, j != 0u);
            }
            //create fbos with created desc for each resolution
            for (uint32_t x = 0; x < numResolutions; ++x)
            {
                for (uint32_t y = 0; y < numResolutions; ++y)
                {
                    Fbo::SharedPtr pFbo = gpDevice->getSwapChainFbo();
                    if (cubemap)
                    {
                        pFbo = FboHelper::createCubemap(widths[x], heights[y], desc);
                    }
                    else
                    {
                        pFbo = FboHelper::create2D(widths[x], heights[y], desc);
                    }

                    if (!isFboCorrect(pFbo, widths[x], heights[y], desc))
                    {
                        return false;
                    }

                    //sets resource state so before and after states are different, which is required by present
                    gpDevice->getRenderContext()->resourceBarrier(gpDevice->getSwapChainFbo()->getColorTexture(0).get(), Resource::State::GenericRead);
                    //frees resources
                    gpDevice->present();
                }
            }
        }
    }

    return true;
}

bool FboTest::isFboCorrect(Fbo::SharedPtr fbo, const uint32_t& w, const uint32_t& h, const Fbo::Desc& desc)
{
    //check non color target properties
    if (fbo->getWidth() != w || fbo->getHeight() != h || fbo->getDesc().getSampleCount() != desc.getSampleCount() ||
        fbo->getDesc().isDepthStencilUav() != desc.isDepthStencilUav() || 
        fbo->getDepthStencilTexture()->getFormat() != desc.getDepthStencilFormat())
    {
        return false;
    }

    //check color target properties
    for (uint32_t i = 0; i < Fbo::getMaxColorTargetCount(); ++i)
    {
        if (fbo->getColorTexture(i) != nullptr)
        {
            if (fbo->getColorTexture(i)->getFormat() != desc.getColorTargetFormat(i) ||
                fbo->getDesc().isColorTargetUav(i) != desc.isColorTargetUav(i))
            {
                return false;
            }
        }
    }

    return true;
}

int main()
{
    FboTest fbot;
    fbot.init(true);
    fbot.run();
    return 0;
}
