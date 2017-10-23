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
#include "Framework.h"
#include "API/Device.h"
#include "VR/OpenVR/VRSystem.h"

namespace Falcor
{
    Device::SharedPtr gpDevice;
    
    Device::SharedPtr Device::create(Window::SharedPtr& pWindow, const Device::Desc& desc)
    {
        if (gpDevice)
        {
            logError("Falcor only supports a single device");
            return false;
        }
        gpDevice = SharedPtr(new Device(pWindow));
        if (gpDevice->init(desc) == false) { gpDevice = nullptr;}
        return gpDevice;
    }

    bool Device::init(const Desc& desc)
    {
        if (desc.enableVR) VRSystem::start(desc.enableVsync);

        const uint32_t kDirectQueueIndex = (uint32_t)LowLevelContextData::CommandQueueType::Direct;
        assert(desc.cmdQueues[kDirectQueueIndex] > 0);
        if (apiInit(desc) == false) return false;

        mpRenderContext = RenderContext::create(mCmdQueues[kDirectQueueIndex][0]);

        // Create the descriptor pools
        DescriptorPool::Desc poolDesc;
        // For DX12 there is no difference between the different SRV/UAV types. For Vulkan it matters, hence the #ifdef
        poolDesc.setDescCount(DescriptorPool::Type::TextureSrv, 16 * 1024).setDescCount(DescriptorPool::Type::Sampler, 2048).setShaderVisible(true);
#ifndef FALCOR_D3D12
        poolDesc.setDescCount(DescriptorPool::Type::Cbv, 16 * 1024).setDescCount(DescriptorPool::Type::TextureUav, 16 * 1024);
        poolDesc.setDescCount(DescriptorPool::Type::StructuredBufferSrv, 2 * 1024).setDescCount(DescriptorPool::Type::StructuredBufferUav, 2 * 1024).setDescCount(DescriptorPool::Type::TypedBufferSrv, 2 * 1024).setDescCount(DescriptorPool::Type::TypedBufferUav, 2 * 1024);
#endif
        mpGpuDescPool = DescriptorPool::create(poolDesc, mpRenderContext->getLowLevelData()->getFence());
        poolDesc.setShaderVisible(false).setDescCount(DescriptorPool::Type::Rtv, 16 * 1024).setDescCount(DescriptorPool::Type::Dsv, 1024);
        mpCpuDescPool = DescriptorPool::create(poolDesc, mpRenderContext->getLowLevelData()->getFence());

        if(mpRenderContext) mpRenderContext->reset();

        mVsyncOn = desc.enableVsync;

        // Create the swap-chain
        mpResourceAllocator = ResourceAllocator::create(1024 * 1024 * 2, mpRenderContext->getLowLevelData()->getFence());
        if (createSwapChain(desc.colorFormat) == false)
        {
            return false;
        }

        mpFrameFence = GpuFence::create();

        // Update the FBOs
        if (updateDefaultFBO(mpWindow->getClientAreaWidth(), mpWindow->getClientAreaHeight(), desc.colorFormat, desc.depthFormat) == false)
        {
            return false;
        }

        if (desc.enableVR && VRSystem::instance()) VRSystem::instance()->initDisplayAndController(mpRenderContext);
        gpDevice->mTimestampQueryHeap = QueryHeap::create(QueryHeap::Type::Timestamp, 128 * 1024 * 1024);

        return true;
    }

    void Device::releaseFboData()
    {
        // First, delete all FBOs
        for (uint32_t i = 0; i < arraysize(mpSwapChainFbos); i++)
        {
            mpSwapChainFbos[i]->attachColorTarget(nullptr, 0);
            mpSwapChainFbos[i]->attachDepthStencilTarget(nullptr);
        }

        // Now execute all deferred releases
        decltype(mDeferredReleases)().swap(mDeferredReleases);
    }

    bool Device::updateDefaultFBO(uint32_t width, uint32_t height, ResourceFormat colorFormat, ResourceFormat depthFormat)
    {
        std::vector<ResourceHandle> apiHandles(kSwapChainBuffers);
        getApiFboData(width, height, colorFormat, depthFormat, apiHandles, mCurrentBackBufferIndex);

        for (uint32_t i = 0; i < kSwapChainBuffers; i++)
        {
            // Create a texture object
            auto pColorTex = Texture::SharedPtr(new Texture(width, height, 1, 1, 1, 1, colorFormat, Texture::Type::Texture2D, Texture::BindFlags::RenderTarget));
            pColorTex->mApiHandle = apiHandles[i];

            // Create the FBO if it's required
            if (mpSwapChainFbos[i] == nullptr)
            {
                mpSwapChainFbos[i] = Fbo::create();
            }

            mpSwapChainFbos[i]->attachColorTarget(pColorTex, 0);

            // Create a depth texture
            if (depthFormat != ResourceFormat::Unknown)
            {
                auto pDepth = Texture::create2D(width, height, depthFormat, 1, 1, nullptr, Texture::BindFlags::DepthStencil);
                mpSwapChainFbos[i]->attachDepthStencilTarget(pDepth);
            }
        }
        return true;
    }

    Fbo::SharedPtr Device::getSwapChainFbo() const
    {
        return mpSwapChainFbos[mCurrentBackBufferIndex];
    }

    void Device::releaseResource(ApiObjectHandle pResource)
    {
        if (pResource)
        {
            // Some static objects get here when the application exits
            if(this)
            {
                mDeferredReleases.push({ mpFrameFence->getCpuValue(), pResource });
            }
        }
    }

    void Device::executeDeferredReleases()
    {
        mpResourceAllocator->executeDeferredReleases();
        uint64_t gpuVal = mpFrameFence->getGpuValue();
        while (mDeferredReleases.size() && mDeferredReleases.front().frameID <= gpuVal)
        {
            mDeferredReleases.pop();
        }
        mpCpuDescPool->executeDeferredReleases();
        mpGpuDescPool->executeDeferredReleases();
    }

    void Device::toggleVSync(bool enable)
    {
        mVsyncOn = enable;
    }

    void Device::cleanup()
    {
        mpRenderContext->flush(true);
        // Release all the bound resources. Need to do that before deleting the RenderContext
        mpRenderContext->setGraphicsState(nullptr);
        mpRenderContext->setGraphicsVars(nullptr);
        mpRenderContext->setComputeState(nullptr);
        mpRenderContext->setComputeVars(nullptr);

        for (uint32_t i = 0; i < arraysize(mCmdQueues); i++) mCmdQueues[i].clear();
        for (uint32_t i = 0; i < arraysize(mpSwapChainFbos); i++) mpSwapChainFbos[i].reset();
        mDeferredReleases = decltype(mDeferredReleases)();

        mpRenderContext.reset();
        mpResourceAllocator.reset();
        mpCpuDescPool.reset();
        mpGpuDescPool.reset();
        mpFrameFence.reset();

        destroyApiObjects();
        mpWindow.reset();
    }

    void Device::present()
    {
        mpRenderContext->resourceBarrier(mpSwapChainFbos[mCurrentBackBufferIndex]->getColorTexture(0).get(), Resource::State::Present);
        mpRenderContext->flush();
        apiPresent();
        mpFrameFence->gpuSignal(mpRenderContext->getLowLevelData()->getCommandQueue());
        executeDeferredReleases();
        mpRenderContext->reset();
        mFrameID++;
    }

    void Device::flushAndSync()
    {
        mpRenderContext->flush(true);
        executeDeferredReleases();
    }

    Fbo::SharedPtr Device::resizeSwapChain(uint32_t width, uint32_t height)
    {
        mpRenderContext->flush(true);

        // Store the FBO parameters
        ResourceFormat colorFormat = mpSwapChainFbos[0]->getColorTexture(0)->getFormat();
        const auto& pDepth = mpSwapChainFbos[0]->getDepthStencilTexture();
        ResourceFormat depthFormat = pDepth ? pDepth->getFormat() : ResourceFormat::Unknown;
        assert(mpSwapChainFbos[0]->getSampleCount() == 1);

        // Delete all the FBOs
        releaseFboData();
        apiResizeSwapChain(width, height, colorFormat);
        updateDefaultFBO(width, height, colorFormat, depthFormat);

        return getSwapChainFbo();
    }
}