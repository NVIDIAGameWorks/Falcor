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
#include "NGXWrapper.h"

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>

#if FALCOR_HAS_D3D12
#include <d3d12.h>
#endif

#ifdef FALCOR_VK
#include <vulkan/vulkan.h>
#include <nvsdk_ngx_vk.h>
#include <nvrhi/vulkan.h>
#include <nvsdk_ngx_helpers_vk.h>
#endif

#include <cstdio>
#include <cstdarg>

namespace Falcor
{
    namespace
    {
        const uint64_t kAppID = 231313132;

        std::string resultToString(NVSDK_NGX_Result result)
        {
            char buf[1024];
            snprintf(buf, sizeof(buf), "(code: 0x%08x, info: %ls)", result, GetNGXResultAsString(result));
            buf[sizeof(buf) - 1] = '\0';
            return std::string(buf);
        }
    }

    NGXWrapper::NGXWrapper(Device* pDevice, const wchar_t* logFolder)
        : mpDevice(pDevice)
    {
        initializeNGX(logFolder);
    }

    NGXWrapper::~NGXWrapper()
    {
        if (mInitialized)
        {
            releaseDLSS();
        }
        shutdownNGX();
    }

    void NGXWrapper::initializeNGX(const wchar_t* logFolder)
    {
        NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;

#if FALCOR_HAS_D3D12
        result = NVSDK_NGX_D3D12_Init(kAppID, logFolder, mpDevice->getD3D12Handle());
#endif

#ifdef FALCOR_VK
        VkDevice vkDevice = mpDevice->getApiHandle();;
        VkPhysicalDevice vkPhysicalDevice = mpDevice->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);
        VkInstance vkInstance = mpDevice->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
        result = NVSDK_NGX_VULKAN_Init(kAppID, logFolder, vkInstance, vkPhysicalDevice, vkDevice);
#endif

        if (NVSDK_NGX_FAILED(result))
        {
            if (result == NVSDK_NGX_Result_FAIL_FeatureNotSupported || result == NVSDK_NGX_Result_FAIL_PlatformError)
            {
                throw RuntimeError("NVIDIA NGX is not available on this hardware/platform " + resultToString(result));
            }
            else
            {
                throw RuntimeError("Failed to initialize NGX " + resultToString(result));
            }
        }

        mInitialized = true;

#if FALCOR_HAS_D3D12
        result = NVSDK_NGX_D3D12_GetParameters(&mpParameters);
#endif

#ifdef FALCOR_VK
        result = NVSDK_NGX_VULKAN_GetParameters(&mpParameters);
#endif

        if (NVSDK_NGX_FAILED(result))
        {
            throw RuntimeError("NVSDK_NGX_D3D12_GetParameters failed " + resultToString(result));
        }

    // Currently, the SDK and this sample are not in sync.  The sample is a bit forward looking,
    // in this case.  This will likely be resolved very shortly, and therefore, the code below
    // should be thought of as needed for a smooth user experience.
#if defined(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver)        \
    && defined (NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor) \
    && defined (NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor)

        // If NGX Successfully initialized then it should set those flags in return
        int needsUpdatedDriver = 0;
        if (!NVSDK_NGX_FAILED(mpParameters->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver)) && needsUpdatedDriver)
        {
            std::string message = "NVIDIA DLSS cannot be loaded due to outdated driver.";
            unsigned int majorVersion = 0;
            unsigned int minorVersion = 0;
            if (!NVSDK_NGX_FAILED(mpParameters->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &majorVersion)) &&
                ! NVSDK_NGX_FAILED(mpParameters->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minorVersion)))
            {
                message += "Minimum driver version required: " + std::to_string(majorVersion) + "." + std::to_string(minorVersion);
            }
            throw RuntimeError(message);
        }
#endif

        int dlssAvailable  = 0;
        result = mpParameters->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);
        if (NVSDK_NGX_FAILED(result) || !dlssAvailable)
        {
            throw RuntimeError("NVIDIA DLSS not available on this hardward/platform " + resultToString(result));
        }
    }

    void NGXWrapper::shutdownNGX()
    {
        if (mInitialized)
        {
            // mpDevice->waitForIdle();
            mpDevice->flushAndSync();

            if (mpFeature != nullptr) releaseDLSS();

#if FALCOR_HAS_D3D12
            NVSDK_NGX_D3D12_Shutdown();
#endif

#ifdef FALCOR_VK
            NVSDK_NGX_VULKAN_Shutdown();
#endif

            mInitialized = false;
        }
    }

    void NGXWrapper::initializeDLSS(
        RenderContext* pRenderContext,
        uint2 maxRenderSize,
        uint2 displayOutSize,
        Texture* pTarget,
        bool isContentHDR,
        bool depthInverted,
        NVSDK_NGX_PerfQuality_Value perfQuality)
    {
        unsigned int CreationNodeMask = 1;
        unsigned int VisibilityNodeMask = 1;

        bool lowResolutionMotionVectors = 1; // we let the Snippet do the upsampling of the motion vector
        // Next create features
        int createFlags = NVSDK_NGX_DLSS_Feature_Flags_None;
        createFlags |= lowResolutionMotionVectors ? NVSDK_NGX_DLSS_Feature_Flags_MVLowRes : 0;
        createFlags |= isContentHDR ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0;
        createFlags |= depthInverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0;

        NVSDK_NGX_DLSS_Create_Params createParams = {};

        createParams.Feature.InWidth = maxRenderSize.x;
        createParams.Feature.InHeight = maxRenderSize.y;
        createParams.Feature.InTargetWidth = displayOutSize.x;
        createParams.Feature.InTargetHeight = displayOutSize.y;
        createParams.Feature.InPerfQualityValue = perfQuality;
        createParams.InFeatureCreateFlags = createFlags;

#if FALCOR_HAS_D3D12
        pRenderContext->flush();

        ID3D12GraphicsCommandList* d3d12CommandList = pRenderContext->getLowLevelData()->getD3D12CommandList();

        NVSDK_NGX_Result result = NGX_D3D12_CREATE_DLSS_EXT(d3d12CommandList, CreationNodeMask, VisibilityNodeMask, &mpFeature, mpParameters, &createParams);
        if (NVSDK_NGX_FAILED(result))
        {
            throw RuntimeError("Failed to create DLSS feature " + resultToString(result));
        }

        pRenderContext->flush();
#endif

#ifdef FALCOR_VK
        nvrhi::CommandListHandle commandList = mpDevice->createCommandList();
        commandList->open();

        vk::Format targetFormat = nvrhi::vulkan::convertFormat(pTarget->GetDesc().format);
        VkCommandBuffer vkCommandBuffer = commandList->getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer);

        commandList->close();

        NVSDK_NGX_Result result = NGX_VULKAN_CREATE_DLSS_EXT(vkCommandBuffer, CreationNodeMask, VisibilityNodeMask, &mpFeature, mpParameters, &createParams);
        if (NVSDK_NGX_FAILED(result))
        {
            throw RuntimeError("Failed to create DLSS feature " + resultToString(result));
        }

        mpDevice->executeCommandList(commandList);
#endif
    }

    void NGXWrapper::releaseDLSS()
    {
        mpDevice->flushAndSync();

        if (mpFeature)
        {
#if FALCOR_HAS_D3D12
            NVSDK_NGX_D3D12_ReleaseFeature(mpFeature);
#endif
#ifdef FALCOR_VK
            NVSDK_NGX_VULKAN_ReleaseFeature(mpFeature);
#endif
            mpFeature = nullptr;
        }
    }

    NGXWrapper::OptimalSettings NGXWrapper::queryOptimalSettings(uint2 displaySize, NVSDK_NGX_PerfQuality_Value perfQuality) const
    {
        OptimalSettings settings;

        NVSDK_NGX_Result result = NGX_DLSS_GET_OPTIMAL_SETTINGS(mpParameters,
            displaySize.x, displaySize.y, perfQuality,
            &settings.optimalRenderSize.x, &settings.optimalRenderSize.y,
            &settings.maxRenderSize.x, &settings.maxRenderSize.y,
            &settings.minRenderSize.x, &settings.minRenderSize.y,
            &settings.sharpness);

        if (NVSDK_NGX_FAILED(result))
        {
            throw RuntimeError("Querying optimal settings failed " + resultToString(result));
        }

        // Depending on what version of DLSS DLL is being used, a sharpness of > 1.f was possible.
        settings.sharpness = clamp(settings.sharpness, -1.f, 1.f);

        return settings;
    }

#if 0
    NVSDK_NGX_Resource_VK TextureToResourceVK(nvrhi::ITexture * tex, nvrhi::TextureSubresourceSet subresources)
    {
        nvrhi::TextureDesc desc = tex->GetDesc();
        NVSDK_NGX_Resource_VK resourceVK = {};
        VkImageView imageView = tex->getNativeView(nvrhi::ObjectTypes::VK_ImageView, nvrhi::Format::UNKNOWN, subresources);
        VkFormat format = (VkFormat)nvrhi::vulkan::convertFormat(desc.format);
        VkImage image = tex->getNativeView(nvrhi::ObjectTypes::VK_Image, nvrhi::Format::UNKNOWN, subresources);
        VkImageSubresourceRange subresourceRange = { 1, subresources.baseMipLevel, subresources.numMipLevels, subresources.baseArraySlice, subresources.numArraySlices };

        return NVSDK_NGX_Create_ImageView_Resource_VK(imageView, image, subresourceRange, format, desc.width, desc.height, desc.isUAV);
    }
#endif

    bool NGXWrapper::evaluateDLSS(
                RenderContext* pRenderContext,
                Texture* pUnresolvedColor,
                Texture* pResolvedColor,
                Texture* pMotionVectors,
                Texture* pDepth,
                Texture* pExposure,
                bool resetAccumulation,
                float sharpness,
                float2 jitterOffset,
                float2 motionVectorScale) const
    {
        if (!mpFeature) return false;

        // In DLSS v2, the target is already upsampled (while in v1, the upsampling is handled in a later pass)
        FALCOR_ASSERT(pResolvedColor->getWidth() > pUnresolvedColor->getWidth() &&
            pResolvedColor->getHeight() > pUnresolvedColor->getHeight());

        bool success = true;

#if FALCOR_HAS_D3D12
        pRenderContext->resourceBarrier(pUnresolvedColor, Resource::State::ShaderResource);
        pRenderContext->resourceBarrier(pMotionVectors, Resource::State::ShaderResource);
        pRenderContext->resourceBarrier(pDepth, Resource::State::ShaderResource);
        pRenderContext->resourceBarrier(pResolvedColor, Resource::State::UnorderedAccess);

        ID3D12GraphicsCommandList* d3dCommandList = pRenderContext->getLowLevelData()->getD3D12CommandList();
        ID3D12Resource* unresolvedColorBuffer = pUnresolvedColor->getD3D12Handle();
        ID3D12Resource* motionVectorsBuffer = pMotionVectors->getD3D12Handle();
        ID3D12Resource* resolvedColorBuffer = pResolvedColor->getD3D12Handle();
        ID3D12Resource* depthBuffer = pDepth->getD3D12Handle();
        ID3D12Resource* exposureBuffer = pExposure ? pExposure->getD3D12Handle() : nullptr;

        NVSDK_NGX_D3D12_DLSS_Eval_Params evalParams = {};

        evalParams.Feature.pInColor = unresolvedColorBuffer;
        evalParams.Feature.pInOutput = resolvedColorBuffer;
        evalParams.Feature.InSharpness = sharpness;
        evalParams.pInDepth = depthBuffer;
        evalParams.pInMotionVectors = motionVectorsBuffer;
        evalParams.InJitterOffsetX = jitterOffset.x;
        evalParams.InJitterOffsetY = jitterOffset.y;
        evalParams.InReset = resetAccumulation ? 1 : 0;;
        evalParams.InRenderSubrectDimensions.Width = pUnresolvedColor->getWidth();
        evalParams.InRenderSubrectDimensions.Height = pUnresolvedColor->getHeight();
        evalParams.InMVScaleX = motionVectorScale.x;
        evalParams.InMVScaleY = motionVectorScale.y;
        evalParams.pInExposureTexture = exposureBuffer;

        NVSDK_NGX_Result result = NGX_D3D12_EVALUATE_DLSS_EXT(d3dCommandList, mpFeature, mpParameters, &evalParams);
        if (NVSDK_NGX_FAILED(result))
        {
            logWarning("Failed to NVSDK_NGX_D3D12_EvaluateFeature for DLSS: {}", resultToString(result));
            success = false;
        }

        pRenderContext->setPendingCommands(true);
        pRenderContext->uavBarrier(pResolvedColor);
        // TODO: Get rid of the flush
        pRenderContext->flush();
#endif // FALCOR_HAS_D3D12

#ifdef FALCOR_VK
        commandList->endTrackingTextureState(pUnresolvedColor, nvrhi::AllSubresources, nvrhi::ResourceStates::SHADER_RESOURCE);
        commandList->endTrackingTextureState(pResolvedColor, nvrhi::AllSubresources, nvrhi::ResourceStates::UNORDERED_ACCESS);
        commandList->endTrackingTextureState(pMotionVectors, nvrhi::AllSubresources, nvrhi::ResourceStates::SHADER_RESOURCE);
        commandList->endTrackingTextureState(pDepth, nvrhi::AllSubresources, nvrhi::ResourceStates::SHADER_RESOURCE);

        VkCommandBuffer vkCommandbuffer = commandList->getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer);
        nvrhi::TextureSubresourceSet subresources = view->GetSubresources();

        NVSDK_NGX_Resource_VK unresolvedColorResource = TextureToResourceVK(pUnresolvedColor, subresources);
        NVSDK_NGX_Resource_VK resolvedColorResource = TextureToResourceVK(pResolvedColor, subresources);
        NVSDK_NGX_Resource_VK motionVectorsResource = TextureToResourceVK(pMotionVectors, subresources);
        NVSDK_NGX_Resource_VK depthResource = TextureToResourceVK(pDepth, subresources);

        NVSDK_NGX_VK_DLSS_Eval_Params evalParams = {};

        evalParams.Feature.pInColor = &unresolvedColorResource;
        evalParams.Feature.pInOutput = &resolvedColorResource;
        evalParams.pInDepth = &depthResource;
        evalParams.pInMotionVectors = &motionVectorsResource;
        evalParams.InJitterOffsetX = jitterOffset.x;
        evalParams.InJitterOffsetY = jitterOffset.y;
        evalParams.Feature.InSharpness = sharpness;
        evalParams.InReset = Reset;
        evalParams.InMVScaleX = motionVectorScale.x;
        evalParams.InMVScaleY = motionVectorScale.y;

        NVSDK_NGX_Result result = NGX_VULKAN_EVALUATE_DLSS_EXT(vkCommandbuffer, mpFeature, mpParameters, &evalParams);
        if (NVSDK_NGX_FAILED(result))
        {
            logWarning("Failed to NVSDK_NGX_VULKAN_EvaluateFeature for DLSS: {}", resultToString(Result));
            success = false;
        }

        commandList->clearState();
#endif

        return success;
    }

} // namespace Falcor
