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
#include "NGXWrapper.h"
#include "Core/API/NativeHandleTraits.h"
#include "Core/API/NativeFormats.h"

#if FALCOR_HAS_D3D12
#include <d3d12.h>
#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#endif

#if FALCOR_HAS_VULKAN
#include <vulkan/vulkan.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#endif

#include <cstdio>
#include <cstdarg>

#define THROW_IF_FAILED(call)                                                     \
    {                                                                             \
        NVSDK_NGX_Result result_ = call;                                          \
        if (NVSDK_NGX_FAILED(result_))                                            \
            FALCOR_THROW(#call " failed with error {}", resultToString(result_)); \
    }

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

#if FALCOR_HAS_VULKAN
VkImageAspectFlags getAspectMaskFromFormat(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_S8_UINT:
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}
#endif

} // namespace

NGXWrapper::NGXWrapper(
    ref<Device> pDevice,
    const std::filesystem::path& applicationDataPath,
    const std::filesystem::path& featureSearchPath
)
    : mpDevice(pDevice)
{
    initializeNGX(applicationDataPath, featureSearchPath);
}

NGXWrapper::~NGXWrapper()
{
    shutdownNGX();
}

void NGXWrapper::initializeNGX(const std::filesystem::path& applicationDataPath, const std::filesystem::path& featureSearchPath)
{
    NVSDK_NGX_Result result = NVSDK_NGX_Result_Fail;

    NVSDK_NGX_FeatureCommonInfo featureInfo = {};
    const wchar_t* pathList[] = {featureSearchPath.c_str()};
    featureInfo.PathListInfo.Length = 1;
    featureInfo.PathListInfo.Path = const_cast<wchar_t**>(&pathList[0]);

    switch (mpDevice->getType())
    {
    case Device::Type::D3D12:
#if FALCOR_HAS_D3D12
        result = NVSDK_NGX_D3D12_Init(kAppID, applicationDataPath.c_str(), mpDevice->getNativeHandle().as<ID3D12Device*>(), &featureInfo);
#endif
        break;
    case Device::Type::Vulkan:
#if FALCOR_HAS_VULKAN
        result = NVSDK_NGX_VULKAN_Init(
            kAppID,
            applicationDataPath.c_str(),
            mpDevice->getNativeHandle(0).as<VkInstance>(),
            mpDevice->getNativeHandle(1).as<VkPhysicalDevice>(),
            mpDevice->getNativeHandle(2).as<VkDevice>(),
            nullptr,
            nullptr,
            &featureInfo
        );
#endif
        break;
    }

    if (NVSDK_NGX_FAILED(result))
    {
        if (result == NVSDK_NGX_Result_FAIL_FeatureNotSupported || result == NVSDK_NGX_Result_FAIL_PlatformError)
        {
            FALCOR_THROW("NVIDIA NGX is not available on this hardware/platform " + resultToString(result));
        }
        else
        {
            FALCOR_THROW("Failed to initialize NGX " + resultToString(result));
        }
    }

    mInitialized = true;

    switch (mpDevice->getType())
    {
    case Device::Type::D3D12:
#if FALCOR_HAS_D3D12
        THROW_IF_FAILED(NVSDK_NGX_D3D12_GetCapabilityParameters(&mpParameters));
#endif
        break;
    case Device::Type::Vulkan:
#if FALCOR_HAS_VULKAN
        THROW_IF_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&mpParameters));
#endif
        break;
    }

    // Currently, the SDK and this sample are not in sync.  The sample is a bit forward looking,
    // in this case.  This will likely be resolved very shortly, and therefore, the code below
    // should be thought of as needed for a smooth user experience.
#if defined(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver) && defined(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor) && \
    defined(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor)

    // If NGX Successfully initialized then it should set those flags in return
    int needsUpdatedDriver = 0;
    if (!NVSDK_NGX_FAILED(mpParameters->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver)) &&
        needsUpdatedDriver)
    {
        std::string message = "NVIDIA DLSS cannot be loaded due to outdated driver.";
        unsigned int majorVersion = 0;
        unsigned int minorVersion = 0;
        if (!NVSDK_NGX_FAILED(mpParameters->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor, &majorVersion)) &&
            !NVSDK_NGX_FAILED(mpParameters->Get(NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor, &minorVersion)))
        {
            message += fmt::format("\nMinimum driver version required: {}.{}", majorVersion, minorVersion);
        }
        FALCOR_THROW(message);
    }
#endif

    int dlssAvailable = 0;
    result = mpParameters->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssAvailable);
    if (NVSDK_NGX_FAILED(result) || !dlssAvailable)
    {
        FALCOR_THROW("NVIDIA DLSS not available on this hardward/platform " + resultToString(result));
    }
}

void NGXWrapper::shutdownNGX()
{
    if (mInitialized)
    {
        mpDevice->wait();

        if (mpFeature != nullptr)
            releaseDLSS();

        switch (mpDevice->getType())
        {
        case Device::Type::D3D12:
#if FALCOR_HAS_D3D12
            THROW_IF_FAILED(NVSDK_NGX_D3D12_DestroyParameters(mpParameters));
            THROW_IF_FAILED(NVSDK_NGX_D3D12_Shutdown1(mpDevice->getNativeHandle().as<ID3D12Device*>()));
#endif
            break;
        case Device::Type::Vulkan:
#if FALCOR_HAS_VULKAN
            THROW_IF_FAILED(NVSDK_NGX_VULKAN_DestroyParameters(mpParameters));
            THROW_IF_FAILED(NVSDK_NGX_VULKAN_Shutdown1(mpDevice->getNativeHandle(2).as<VkDevice>()));
#endif
            break;
        }

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
    NVSDK_NGX_PerfQuality_Value perfQuality
)
{
    unsigned int creationNodeMask = 1;
    unsigned int visibilityNodeMask = 1;

    // Next create features
    int createFlags = NVSDK_NGX_DLSS_Feature_Flags_None;
    createFlags |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    createFlags |= isContentHDR ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0;
    createFlags |= depthInverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0;

    NVSDK_NGX_DLSS_Create_Params dlssParams = {};

    dlssParams.Feature.InWidth = maxRenderSize.x;
    dlssParams.Feature.InHeight = maxRenderSize.y;
    dlssParams.Feature.InTargetWidth = displayOutSize.x;
    dlssParams.Feature.InTargetHeight = displayOutSize.y;
    dlssParams.Feature.InPerfQualityValue = perfQuality;
    dlssParams.InFeatureCreateFlags = createFlags;

    switch (mpDevice->getType())
    {
    case Device::Type::D3D12:
    {
#if FALCOR_HAS_D3D12
        pRenderContext->submit();
        ID3D12GraphicsCommandList* pCommandList =
            pRenderContext->getLowLevelData()->getCommandBufferNativeHandle().as<ID3D12GraphicsCommandList*>();
        THROW_IF_FAILED(NGX_D3D12_CREATE_DLSS_EXT(pCommandList, creationNodeMask, visibilityNodeMask, &mpFeature, mpParameters, &dlssParams)
        );
        pRenderContext->submit();
#endif
        break;
    }
    case Device::Type::Vulkan:
    {
#if FALCOR_HAS_VULKAN
        pRenderContext->submit();
        VkCommandBuffer vkCommandBuffer = pRenderContext->getLowLevelData()->getCommandBufferNativeHandle().as<VkCommandBuffer>();
        THROW_IF_FAILED(
            NGX_VULKAN_CREATE_DLSS_EXT(vkCommandBuffer, creationNodeMask, visibilityNodeMask, &mpFeature, mpParameters, &dlssParams)
        );
        pRenderContext->submit();
#endif
        break;
    }
    }
}

void NGXWrapper::releaseDLSS()
{
    if (mpFeature)
    {
        mpDevice->wait();

        switch (mpDevice->getType())
        {
        case Device::Type::D3D12:
#if FALCOR_HAS_D3D12
            THROW_IF_FAILED(NVSDK_NGX_D3D12_ReleaseFeature(mpFeature));
#endif
            break;
        case Device::Type::Vulkan:
#if FALCOR_HAS_VULKAN
            THROW_IF_FAILED(NVSDK_NGX_VULKAN_ReleaseFeature(mpFeature));
#endif
            break;
        }
        mpFeature = nullptr;
    }
}

NGXWrapper::OptimalSettings NGXWrapper::queryOptimalSettings(uint2 displaySize, NVSDK_NGX_PerfQuality_Value perfQuality) const
{
    OptimalSettings settings;

    THROW_IF_FAILED(NGX_DLSS_GET_OPTIMAL_SETTINGS(
        mpParameters,
        displaySize.x,
        displaySize.y,
        perfQuality,
        &settings.optimalRenderSize.x,
        &settings.optimalRenderSize.y,
        &settings.maxRenderSize.x,
        &settings.maxRenderSize.y,
        &settings.minRenderSize.x,
        &settings.minRenderSize.y,
        &settings.sharpness
    ));

    // Depending on what version of DLSS DLL is being used, a sharpness of > 1.f was possible.
    settings.sharpness = math::clamp(settings.sharpness, -1.f, 1.f);

    return settings;
}

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
    float2 motionVectorScale
) const
{
    if (!mpFeature)
        return false;

    // In DLSS v2, the target is already upsampled (while in v1, the upsampling is handled in a later pass)
    FALCOR_ASSERT(pResolvedColor->getWidth() > pUnresolvedColor->getWidth() && pResolvedColor->getHeight() > pUnresolvedColor->getHeight());

    bool success = true;

    switch (mpDevice->getType())
    {
    case Device::Type::D3D12:
    {
#if FALCOR_HAS_D3D12
        pRenderContext->resourceBarrier(pUnresolvedColor, Resource::State::ShaderResource);
        pRenderContext->resourceBarrier(pMotionVectors, Resource::State::ShaderResource);
        pRenderContext->resourceBarrier(pDepth, Resource::State::ShaderResource);
        pRenderContext->resourceBarrier(pResolvedColor, Resource::State::UnorderedAccess);

        ID3D12Resource* unresolvedColorBuffer = pUnresolvedColor->getNativeHandle().as<ID3D12Resource*>();
        ID3D12Resource* motionVectorsBuffer = pMotionVectors->getNativeHandle().as<ID3D12Resource*>();
        ID3D12Resource* resolvedColorBuffer = pResolvedColor->getNativeHandle().as<ID3D12Resource*>();
        ID3D12Resource* depthBuffer = pDepth->getNativeHandle().as<ID3D12Resource*>();
        ID3D12Resource* exposureBuffer = pExposure ? pExposure->getNativeHandle().as<ID3D12Resource*>() : nullptr;

        NVSDK_NGX_D3D12_DLSS_Eval_Params evalParams = {};

        evalParams.Feature.pInColor = unresolvedColorBuffer;
        evalParams.Feature.pInOutput = resolvedColorBuffer;
        evalParams.Feature.InSharpness = sharpness;
        evalParams.pInDepth = depthBuffer;
        evalParams.pInMotionVectors = motionVectorsBuffer;
        evalParams.InJitterOffsetX = jitterOffset.x;
        evalParams.InJitterOffsetY = jitterOffset.y;
        evalParams.InReset = resetAccumulation ? 1 : 0;
        evalParams.InRenderSubrectDimensions.Width = pUnresolvedColor->getWidth();
        evalParams.InRenderSubrectDimensions.Height = pUnresolvedColor->getHeight();
        evalParams.InMVScaleX = motionVectorScale.x;
        evalParams.InMVScaleY = motionVectorScale.y;
        evalParams.pInExposureTexture = exposureBuffer;

        ID3D12GraphicsCommandList* pCommandList =
            pRenderContext->getLowLevelData()->getCommandBufferNativeHandle().as<ID3D12GraphicsCommandList*>();
        NVSDK_NGX_Result result = NGX_D3D12_EVALUATE_DLSS_EXT(pCommandList, mpFeature, mpParameters, &evalParams);
        if (NVSDK_NGX_FAILED(result))
        {
            logWarning("Failed to NGX_D3D12_EVALUATE_DLSS_EXT for DLSS: {}", resultToString(result));
            success = false;
        }

        pRenderContext->setPendingCommands(true);
        pRenderContext->uavBarrier(pResolvedColor);
        // TODO: Get rid of the flush
        pRenderContext->submit();
#endif // FALCOR_HAS_D3D12
        break;
    }
    case Device::Type::Vulkan:
    {
#if FALCOR_HAS_VULKAN
        pRenderContext->resourceBarrier(pUnresolvedColor, Resource::State::ShaderResource);
        pRenderContext->resourceBarrier(pMotionVectors, Resource::State::ShaderResource);
        pRenderContext->resourceBarrier(pDepth, Resource::State::ShaderResource);
        pRenderContext->resourceBarrier(pResolvedColor, Resource::State::UnorderedAccess);

        auto getImageView = [](Texture* pTexture, bool isUAV = false) -> NVSDK_NGX_Resource_VK
        {
            if (!pTexture)
                return {};

            VkImageView imageView = isUAV ? pTexture->getUAV()->getNativeHandle().as<VkImageView>()
                                          : pTexture->getSRV(0, 1)->getNativeHandle().as<VkImageView>();
            VkImage image = pTexture->getNativeHandle().as<VkImage>();
            VkFormat format = getVulkanFormat(pTexture->getFormat());
            VkImageSubresourceRange range;
            range.aspectMask = getAspectMaskFromFormat(format);
            range.baseMipLevel = 0;
            range.levelCount = 1;
            range.baseArrayLayer = 0;
            range.layerCount = 1;
            return NVSDK_NGX_Create_ImageView_Resource_VK(
                imageView, image, range, format, pTexture->getWidth(), pTexture->getHeight(), isUAV
            );
        };

        NVSDK_NGX_Resource_VK unresolvedColorBuffer = getImageView(pUnresolvedColor);
        NVSDK_NGX_Resource_VK motionVectorsBuffer = getImageView(pMotionVectors);
        NVSDK_NGX_Resource_VK resolvedColorBuffer = getImageView(pResolvedColor, true);
        NVSDK_NGX_Resource_VK depthBuffer = getImageView(pDepth);
        NVSDK_NGX_Resource_VK exposureBuffer = getImageView(pExposure);

        NVSDK_NGX_VK_DLSS_Eval_Params evalParams = {};

        evalParams.Feature.pInColor = &unresolvedColorBuffer;
        evalParams.Feature.pInOutput = &resolvedColorBuffer;
        evalParams.Feature.InSharpness = sharpness;
        evalParams.pInDepth = &depthBuffer;
        evalParams.pInMotionVectors = &motionVectorsBuffer;
        evalParams.InJitterOffsetX = jitterOffset.x;
        evalParams.InJitterOffsetY = jitterOffset.y;
        evalParams.InReset = resetAccumulation ? 1 : 0;
        evalParams.InRenderSubrectDimensions.Width = pUnresolvedColor->getWidth();
        evalParams.InRenderSubrectDimensions.Height = pUnresolvedColor->getHeight();
        evalParams.InMVScaleX = motionVectorScale.x;
        evalParams.InMVScaleY = motionVectorScale.y;
        evalParams.pInExposureTexture = &exposureBuffer;

        VkCommandBuffer vkCommandBuffer = pRenderContext->getLowLevelData()->getCommandBufferNativeHandle().as<VkCommandBuffer>();
        NVSDK_NGX_Result result = NGX_VULKAN_EVALUATE_DLSS_EXT(vkCommandBuffer, mpFeature, mpParameters, &evalParams);
        if (NVSDK_NGX_FAILED(result))
        {
            logWarning("Failed to NGX_VULKAN_EVALUATE_DLSS_EXT for DLSS: {}", resultToString(result));
            success = false;
        }

        pRenderContext->setPendingCommands(true);
        pRenderContext->uavBarrier(pResolvedColor);
        // TODO: Get rid of the flush
        pRenderContext->submit();
#endif
        break;
    }
    }

    return success;
}

} // namespace Falcor
