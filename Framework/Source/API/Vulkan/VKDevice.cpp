/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "API/Device.h"
#include "API/LowLevel/DescriptorPool.h"
#include "API/LowLevel/GpuFence.h"
#include "API/Vulkan/FalcorVK.h"
#include <set>
#include "Falcor.h"
#include "VR/OpenVR/VRSystem.h"

// #define VK_REPORT_PERF_WARNINGS // Uncomment this to see performance warnings
namespace Falcor
{
#ifdef DEFAULT_ENABLE_DEBUG_LAYER
    VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
        VkDebugReportFlagsEXT       flags,
        VkDebugReportObjectTypeEXT  objectType,
        uint64_t                    object,
        size_t                      location,
        int32_t                     messageCode,
        const char*                 pLayerPrefix,
        const char*                 pMessage,
        void*                       pUserData)
    {
        std::string type = "FalcorVK ";
        type += ((flags | VK_DEBUG_REPORT_ERROR_BIT_EXT) ? "Error: " : "Warning: ");
        printToDebugWindow(type + std::string(pMessage) + "\n");
        return VK_FALSE;
    }
#endif

    uint32_t getMaxViewportCount()
    {
        assert(gpDevice);
        return gpDevice->getPhysicalDeviceLimits().maxViewports;
    }

    struct DeviceApiData
    {
        VkSwapchainKHR swapchain;
        VkPhysicalDeviceProperties properties;
        uint32_t falcorToVulkanQueueType[Device::kQueueTypeCount];
        uint32_t vkMemoryTypeBits[(uint32_t)Device::MemoryType::Count];
        VkPhysicalDeviceLimits deviceLimits;
        std::vector<VkExtensionProperties> deviceExtensions;

        struct  
        {
            std::vector<VkFence> f;
            uint32_t cur = 0;
        } presentFences;
#ifdef DEFAULT_ENABLE_DEBUG_LAYER
        VkDebugReportCallbackEXT debugReportCallbackHandle;
#endif
    };

    static uint32_t getMemoryBits(VkPhysicalDevice physicalDevice, VkMemoryPropertyFlagBits memFlagBits)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        uint32_t bits = 0;
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((memProperties.memoryTypes[i].propertyFlags & memFlagBits) == memFlagBits)
            {
                bits |= (1 << i);
            }
        }
        return bits;
    }

    static uint32_t getCurrentBackBufferIndex(VkDevice device, uint32_t backBufferCount, DeviceApiData* pApiData)
    {
        VkFence fence = pApiData->presentFences.f[pApiData->presentFences.cur];
        vk_call(vkWaitForFences(device, 1, &fence, false, -1));

        pApiData->presentFences.cur = (pApiData->presentFences.cur + 1) % backBufferCount;
        fence = pApiData->presentFences.f[pApiData->presentFences.cur];
        vkResetFences(device, 1, &fence);
        uint32_t newIndex;
        vk_call(vkAcquireNextImageKHR(device, pApiData->swapchain, std::numeric_limits<uint64_t>::max(), nullptr, fence, &newIndex));
        return newIndex;
    }

    static bool initMemoryTypes(VkPhysicalDevice physicalDevice, DeviceApiData* pApiData)
    {
        VkMemoryPropertyFlagBits bits[(uint32_t)Device::MemoryType::Count];
        bits[(uint32_t)Device::MemoryType::Default] = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        bits[(uint32_t)Device::MemoryType::Upload] = VkMemoryPropertyFlagBits(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        bits[(uint32_t)Device::MemoryType::Readback] = VkMemoryPropertyFlagBits(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

        for(uint32_t i = 0 ; i < arraysize(bits) ; i++)
        {
            pApiData->vkMemoryTypeBits[i] = getMemoryBits(physicalDevice, bits[i]);
            if (pApiData->vkMemoryTypeBits[i] == 0)
            {
                logError("Missing memory type " + std::to_string(i));
                return false;
            }
        }
        return true;
    }

    bool Device::getApiFboData(uint32_t width, uint32_t height, ResourceFormat colorFormat, ResourceFormat depthFormat, std::vector<ResourceHandle>& apiHandles, uint32_t& currentBackBufferIndex)
    {
        uint32_t imageCount = 0;
        vkGetSwapchainImagesKHR(mApiHandle, mpApiData->swapchain, &imageCount, nullptr);
        assert(imageCount == apiHandles.size());

        std::vector<VkImage> swapchainImages(imageCount);
        vkGetSwapchainImagesKHR(mApiHandle, mpApiData->swapchain, &imageCount, swapchainImages.data());
        for (size_t i = 0; i < swapchainImages.size(); i++)
        {
            apiHandles[i] = ResourceHandle::create(swapchainImages[i], nullptr);
        }

        // Get the back-buffer
        mCurrentBackBufferIndex = getCurrentBackBufferIndex(mApiHandle, mSwapChainBufferCount, mpApiData);
        return true;
    }

    void Device::destroyApiObjects()
    {
        PFN_vkDestroyDebugReportCallbackEXT DestroyDebugReportCallback = VK_NULL_HANDLE;
        DestroyDebugReportCallback = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(mApiHandle, "vkDestroyDebugReportCallbackEXT");
        if(DestroyDebugReportCallback)
        {
            DestroyDebugReportCallback(mApiHandle, mpApiData->debugReportCallbackHandle, nullptr);
        }
        vkDestroySwapchainKHR(mApiHandle, mpApiData->swapchain, nullptr);
        for (auto& f : mpApiData->presentFences.f)
        {
            vkDestroyFence(mApiHandle, f, nullptr);
        }
        safe_delete(mpApiData);
    }

    static std::vector<VkLayerProperties> enumarateInstanceLayersProperties()
    {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layerProperties(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());
        
        for (const VkLayerProperties& layer : layerProperties)
        {
            logInfo("Available Vulkan Layer: " + std::string(layer.layerName) + " - VK Spec Version: " + std::to_string(layer.specVersion) + " - Implementation Version: " + std::to_string(layer.implementationVersion));
        }

        return layerProperties;
    }

    static bool isLayerSupported(const std::string& layer, const std::vector<VkLayerProperties>& supportedLayers)
    {
        for (const auto& l : supportedLayers)
        {
            if (l.layerName == layer) return true;
        }
        return false;
    }

    void enableLayerIfPresent(const char* layerName, const std::vector<VkLayerProperties>& supportedLayers, std::vector<const char*>& requiredLayers)
    {
        if (isLayerSupported(layerName, supportedLayers))
        {
            requiredLayers.push_back(layerName);
        }
        else
        {
            logWarning("Can't enable requested Vulkan layer " + std::string(layerName) + ". Something bad might happen. Or not, depends on the layer.");
        }
    }

    static std::vector<VkExtensionProperties> enumarateInstanceExtensions()
    {
        // Enumerate implicitly available extensions. The debug layers above just have VK_EXT_debug_report
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> supportedExtensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, supportedExtensions.data());

        for (const VkExtensionProperties& extension : supportedExtensions)
        {
            logInfo("Available Instance Extension: " + std::string(extension.extensionName) + " - VK Spec Version: " + std::to_string(extension.specVersion));
        }

        return supportedExtensions;
    }

    static void initDebugCallback(VkInstance instance, VkDebugReportCallbackEXT* pCallback)
    {
        VkDebugReportCallbackCreateInfoEXT callbackCreateInfo = {};
        callbackCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
        callbackCreateInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
#ifdef VK_REPORT_PERF_WARNINGS
        callbackCreateInfo.flags |= VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
#endif
#ifdef DEFAULT_ENABLE_DEBUG_LAYER
        callbackCreateInfo.pfnCallback = &debugReportCallback;
#endif

        // Function to create a debug callback has to be dynamically queried from the instance...
        PFN_vkCreateDebugReportCallbackEXT CreateDebugReportCallback = VK_NULL_HANDLE;
        CreateDebugReportCallback = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");

        if (VK_FAILED(CreateDebugReportCallback(instance, &callbackCreateInfo, nullptr, pCallback)))
        {
            logWarning("Could not initialize debug report callbacks.");
        }
    }


    static bool isExtensionSupported(const std::string& str, const std::vector<VkExtensionProperties>& vec)
    {
        for (const auto& s : vec)
        {
            if (str == std::string(s.extensionName)) return true;
        }
        return false;
    }

    static void appendVrExtensions(std::vector<const char*>& vkExt, const std::vector<std::string>& vrSystemExt, const std::vector<VkExtensionProperties>& supportedExt)
    {
        for (const auto& a : vrSystemExt)
        {
            if (isExtensionSupported(a, supportedExt) == false)
            {
                logError("Can't start OpenVR. Missing device extension " + a);
            }
            else
            {
                vkExt.push_back(a.c_str());
            }
        }
    }

    VkInstance createInstance(DeviceApiData* pData, const Device::Desc& desc)
    {
        // Initialize the layers
        const auto layerProperties = enumarateInstanceLayersProperties();
        std::vector<const char*> requiredLayers;

        if (desc.enableDebugLayer)
        {
            enableLayerIfPresent("VK_LAYER_LUNARG_standard_validation", layerProperties, requiredLayers);
            enableLayerIfPresent("VK_LAYER_NV_nsight", layerProperties, requiredLayers);
        }

        // Initialize the extensions
        std::vector<VkExtensionProperties> supportedExtensions = enumarateInstanceExtensions();

        // Extensions to use when creating instance
        std::vector<const char*> requiredExtensions = { 
            "VK_KHR_surface",
#ifdef _WIN32
            "VK_KHR_win32_surface"
#else
            "VK_KHR_xlib_surface"
#endif
        };

        if (desc.enableDebugLayer) { requiredExtensions.push_back("VK_EXT_debug_report"); }

        // Get the VR extensions
        std::vector<std::string> vrExt;
        if (desc.enableVR)
        {
            vrExt = VRSystem::getRequiredVkInstanceExtensions();
            appendVrExtensions(requiredExtensions, vrExt, supportedExtensions);
        }

        VkApplicationInfo appInfo = {};
        appInfo.pEngineName = "Falcor";
        appInfo.engineVersion = VK_MAKE_VERSION(FALCOR_MAJOR_VERSION, FALCOR_MINOR_VERSION, 0);
        appInfo.apiVersion = VK_MAKE_VERSION(desc.apiMajorVersion, desc.apiMinorVersion, 0);

        VkInstanceCreateInfo instanceCreateInfo = {};
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pApplicationInfo = &appInfo;
        instanceCreateInfo.enabledLayerCount = (uint32_t)requiredLayers.size();
        instanceCreateInfo.ppEnabledLayerNames = requiredLayers.data();
        instanceCreateInfo.enabledExtensionCount = (uint32_t)requiredExtensions.size();
        instanceCreateInfo.ppEnabledExtensionNames = requiredExtensions.data();

        VkInstance instance;
        if (VK_FAILED(vkCreateInstance(&instanceCreateInfo, nullptr, &instance)))
        {
            logError("Failed to create Vulkan instance");
            return nullptr;
        }

        // Hook up callbacks for VK_EXT_debug_report
        if (desc.enableDebugLayer)
        {
            initDebugCallback(instance, &pData->debugReportCallbackHandle);
        }

        return instance;
    }

    /** Select best physical device based on memory
    */
    VkPhysicalDevice selectPhysicalDevice(const std::vector<VkPhysicalDevice>& devices)
    {
        VkPhysicalDevice bestDevice = VK_NULL_HANDLE;
        uint64_t bestMemory = 0;

        for (const VkPhysicalDevice& device : devices)
        {
            VkPhysicalDeviceMemoryProperties properties;
            vkGetPhysicalDeviceMemoryProperties(device, &properties);

            // Get local memory size from device
            uint64_t deviceMemory = 0;
            for (uint32_t i = 0; i < properties.memoryHeapCount; i++)
            {
                if ((properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) > 0)
                {
                    deviceMemory = properties.memoryHeaps[i].size;
                    break;
                }
            }

            // Save if best found so far
            if (bestDevice == VK_NULL_HANDLE || deviceMemory > bestMemory)
            {
                bestDevice = device;
                bestMemory = deviceMemory;
            }
        }

        return bestDevice;
    }

    VkPhysicalDevice initPhysicalDevice(VkInstance instance, DeviceApiData* pData)
    {
        // Enumerate devices
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        assert(count > 0);

        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());

        // Pick a device
        VkPhysicalDevice physicalDevice = selectPhysicalDevice(devices);
        vkGetPhysicalDeviceProperties(physicalDevice, &pData->properties);
        pData->deviceLimits = pData->properties.limits;

        // Get queue families and match them to what type they are
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

        // Init indices
        for (uint32_t i = 0 ; i < arraysize(pData->falcorToVulkanQueueType) ; i++)
        {
            pData->falcorToVulkanQueueType[i]= (uint32_t)-1;
        }

        // Determine which queue is what type
        uint32_t& graphicsQueueIndex = pData->falcorToVulkanQueueType[(uint32_t)LowLevelContextData::CommandQueueType::Direct];
        uint32_t& computeQueueIndex = pData->falcorToVulkanQueueType[(uint32_t)LowLevelContextData::CommandQueueType::Compute];
        uint32_t& transferQueue = pData->falcorToVulkanQueueType[(uint32_t)LowLevelContextData::CommandQueueType::Copy];

        for (uint32_t i = 0; i < (uint32_t)queueFamilyProperties.size(); i++)
        {
            VkQueueFlags flags = queueFamilyProperties[i].queueFlags;

            if ((flags & VK_QUEUE_GRAPHICS_BIT) != 0 && graphicsQueueIndex == (uint32_t)-1)
            {
                graphicsQueueIndex = i;
            }
            else if ((flags & VK_QUEUE_COMPUTE_BIT) != 0 && computeQueueIndex == (uint32_t)-1)
            {
                computeQueueIndex = i;
            }
            else if ((flags & VK_QUEUE_TRANSFER_BIT) != 0 && transferQueue == (uint32_t)-1)
            {
                transferQueue = i;
            }
        }

        return physicalDevice;
    }

    static void initDeviceQueuesInfo(const Device::Desc& desc, const DeviceApiData *pData, std::vector<VkDeviceQueueCreateInfo>& queueInfos, std::vector<CommandQueueHandle> cmdQueues[Device::kQueueTypeCount], std::vector<std::vector<float>>& queuePriorities)
    {
        queuePriorities.resize(arraysize(pData->falcorToVulkanQueueType));

        // Set up info to create queues for each type
        for (uint32_t type = 0; type < arraysize(pData->falcorToVulkanQueueType); type++)
        {
            const uint32_t queueCount = desc.cmdQueues[type];
            queuePriorities[type].resize(queueCount, 1.0f); // Setting all priority at max for now            
            cmdQueues[type].resize(queueCount); // Save how many queues of each type there will be so we can retrieve them easier after device creation

            VkDeviceQueueCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            info.queueCount = queueCount;
            info.queueFamilyIndex = pData->falcorToVulkanQueueType[type];
            info.pQueuePriorities = queuePriorities[type].data();

            if (info.queueCount > 0)
            {
                queueInfos.push_back(info);
            }
        }
    }

    VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, DeviceApiData *pData, const Device::Desc& desc, std::vector<CommandQueueHandle> cmdQueues[Device::kQueueTypeCount])
    {
        // Features
        VkPhysicalDeviceFeatures requiredFeatures;
        vkGetPhysicalDeviceFeatures(physicalDevice, &requiredFeatures);

        // Queues
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        std::vector<std::vector<float>> queuePriorities;
        initDeviceQueuesInfo(desc, pData, queueInfos, cmdQueues, queuePriorities);

        // Extensions
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
        pData->deviceExtensions.resize(extensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, pData->deviceExtensions.data());

        for (const VkExtensionProperties& extension : pData->deviceExtensions)
        {
            logInfo("Available Device Extension: " + std::string(extension.extensionName) + " - VK Spec Version: " + std::to_string(extension.specVersion));
        }

        std::vector<const char*> extensionNames = { "VK_KHR_swapchain" };
        assert(isExtensionSupported(extensionNames[0], pData->deviceExtensions));

        std::vector<std::string> requiredOpenVRExt;
        if (desc.enableVR)
        {
            requiredOpenVRExt = VRSystem::getRequiredVkDeviceExtensions(physicalDevice);
            appendVrExtensions(extensionNames, requiredOpenVRExt, pData->deviceExtensions);
        }

        for (const auto& a : desc.requiredExtensions)
        {
            if (isExtensionSupported(a, pData->deviceExtensions))
            {
                extensionNames.push_back(a.c_str());
            }
            else
            {
                logWarning("The device doesn't support the requested '" + a + "` extension");
            }
        }

        // Logical Device
        VkDeviceCreateInfo deviceInfo = {};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = (uint32_t)queueInfos.size();
        deviceInfo.pQueueCreateInfos = queueInfos.data();
        deviceInfo.enabledExtensionCount = (uint32_t)extensionNames.size();
        deviceInfo.ppEnabledExtensionNames = extensionNames.data();
        deviceInfo.pEnabledFeatures = &requiredFeatures;

        VkDevice device;
        if (VK_FAILED(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device)))
        {
            logError("Could not create Vulkan logical device.");
            return nullptr;
        }

        // Get the queues we created
        for (uint32_t type = 0; type < arraysize(pData->falcorToVulkanQueueType); type++)
        {
            for (uint32_t i = 0; i < (uint32_t)cmdQueues[type].size(); i++)
            {
                vkGetDeviceQueue(device, pData->falcorToVulkanQueueType[type], i, &cmdQueues[type][i]);
            }
        }

        return device;
    }

    VkSurfaceKHR createSurface(VkInstance instance, VkPhysicalDevice physicalDevice, DeviceApiData *pData, const Window* pWindow)
    {
        VkSurfaceKHR surface;

#ifdef _WIN32
        VkWin32SurfaceCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        createInfo.hwnd = pWindow->getApiHandle();
        createInfo.hinstance = GetModuleHandle(nullptr);

        VkResult result = vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface);
#else
        VkXlibSurfaceCreateInfoKHR createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
        createInfo.dpy = pWindow->getApiHandle().pDisplay;
        createInfo.window = pWindow->getApiHandle().window;

        VkResult result = vkCreateXlibSurfaceKHR(instance, &createInfo, nullptr, &surface);
#endif

        if (VK_FAILED(result))
        {
            logError("Could not create Vulkan surface.");
            return nullptr;
        }

        VkBool32 supported = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, pData->falcorToVulkanQueueType[uint32_t(LowLevelContextData::CommandQueueType::Direct)], surface, &supported);
        assert(supported);

        return surface;
    }

    bool Device::createSwapChain(ResourceFormat colorFormat)
    {
        // Select/Validate SwapChain creation settings
        // Surface size
        VkSurfaceCapabilitiesKHR surfaceCapabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mApiHandle, mApiHandle, &surfaceCapabilities);
        assert(surfaceCapabilities.supportedUsageFlags & (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT));

        VkExtent2D swapchainExtent = {};
        if (surfaceCapabilities.currentExtent.width == (uint32_t)-1)
        {
            swapchainExtent.width = mpWindow->getClientAreaWidth();
            swapchainExtent.height = mpWindow->getClientAreaWidth();
        }
        else
        {
            swapchainExtent = surfaceCapabilities.currentExtent;
        }

        // Validate Surface format
        if (isSrgbFormat(colorFormat) == false)
        {
            logError("Can't create a swap-chain with linear-space color format");
            return false;
        }

        const VkFormat requestedFormat = getVkFormat(colorFormat);
        const VkColorSpaceKHR requestedColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;

        uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(mApiHandle, mApiHandle, &formatCount, nullptr);
        std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(mApiHandle, mApiHandle, &formatCount, surfaceFormats.data());

        bool formatValid = false;
        for (const VkSurfaceFormatKHR& format : surfaceFormats)
        {
            if (format.format == requestedFormat && format.colorSpace == requestedColorSpace)
            {
                formatValid = true;
                break;
            }
        }

        if (formatValid == false)
        {
            logError("Requested Swapchain format is not available");
            return false;
        }

        // Select present mode
        uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(mApiHandle, mApiHandle, &presentModeCount, nullptr);
        std::vector<VkPresentModeKHR> presentModes(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(mApiHandle, mApiHandle, &presentModeCount, presentModes.data());

        // Select present mode, FIFO for VSync, otherwise preferring MAILBOX -> IMMEDIATE -> FIFO
        VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR;
        if (mVsyncOn == false)
        {
            for (size_t i = 0; i < presentModeCount; i++)
            {
                if (presentModes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
                {
                    presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
                    break;
                }
                else if (presentModes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
                {
                    presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                }
            }
        }

        // Swapchain Creation
        VkSwapchainCreateInfoKHR info = {};
        info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        info.surface = mApiHandle;
        uint32 maxImageCount = surfaceCapabilities.maxImageCount ? surfaceCapabilities.maxImageCount : UINT32_MAX; // 0 means no limit on the number of images
        info.minImageCount = clamp(kDefaultSwapChainBuffers, surfaceCapabilities.minImageCount, maxImageCount);
        info.imageFormat = requestedFormat;
        info.imageColorSpace = requestedColorSpace;
        info.imageExtent = { swapchainExtent.width, swapchainExtent.height };
        info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        info.preTransform = surfaceCapabilities.currentTransform;
        info.imageArrayLayers = 1;
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.queueFamilyIndexCount = 0;     // Only needed if VK_SHARING_MODE_CONCURRENT
        info.pQueueFamilyIndices = nullptr; // Only needed if VK_SHARING_MODE_CONCURRENT
        info.presentMode = presentMode;
        info.clipped = true;
        info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        info.oldSwapchain = VK_NULL_HANDLE;

        if (VK_FAILED(vkCreateSwapchainKHR(mApiHandle, &info, nullptr, &mpApiData->swapchain)))
        {
            logError("Could not create swapchain.");
            return false;
        }

        vkGetSwapchainImagesKHR(mApiHandle, mpApiData->swapchain, &mSwapChainBufferCount, nullptr);

        return true;
    }

    void Device::apiPresent()
    {
        VkPresentInfoKHR info = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
        info.swapchainCount = 1;
        info.pSwapchains = &mpApiData->swapchain;
        info.pImageIndices = &mCurrentBackBufferIndex;
        vk_call(vkQueuePresentKHR(mpRenderContext->getLowLevelData()->getCommandQueue(), &info));
        mCurrentBackBufferIndex = getCurrentBackBufferIndex(mApiHandle, mSwapChainBufferCount, mpApiData);
    }

    bool Device::apiInit(const Desc& desc)
    {
        mRgb32FloatSupported = false;

        mpApiData = new DeviceApiData;
        VkInstance instance = createInstance(mpApiData, desc);
        if (!instance) return false;
        VkPhysicalDevice physicalDevice = initPhysicalDevice(instance, mpApiData);
        if (!physicalDevice) return false;
        VkSurfaceKHR surface = createSurface(instance, physicalDevice, mpApiData, mpWindow.get());
        if (!surface) return false;
        VkDevice device = createLogicalDevice(physicalDevice, mpApiData, desc, mCmdQueues);
        if (!device) return false;
        if (initMemoryTypes(physicalDevice, mpApiData) == false) return false;

        mApiHandle = DeviceHandle::create(instance, physicalDevice, device, surface);
        mGpuTimestampFrequency = getPhysicalDeviceLimits().timestampPeriod / (1000 * 1000);

        if (createSwapChain(desc.colorFormat) == false)
        {
            return false;
        }

        mpApiData->presentFences.f.resize(mSwapChainBufferCount);
        for (auto& f : mpApiData->presentFences.f)
        {
            VkFenceCreateInfo info = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            vk_call(vkCreateFence(device, &info, nullptr, &f));
        }

        mpRenderContext = RenderContext::create(mCmdQueues[(uint32_t)LowLevelContextData::CommandQueueType::Direct][0]);

        return true;
    }

    void Device::apiResizeSwapChain(uint32_t width, uint32_t height, ResourceFormat colorFormat)
    {
        vkDestroySwapchainKHR(mApiHandle, mpApiData->swapchain, nullptr);
        createSwapChain(colorFormat);
    }

    bool Device::isWindowOccluded() const
    {
        // #VKTODO Is there a test for it?
        return false;
    }

    bool Device::isExtensionSupported(const std::string& name) const
    {
        return Falcor::isExtensionSupported(name, mpApiData->deviceExtensions);
    }

    ApiCommandQueueType Device::getApiCommandQueueType(LowLevelContextData::CommandQueueType type) const
    {
        return mpApiData->falcorToVulkanQueueType[(uint32_t)type];
    }

    uint32_t Device::getVkMemoryType(MemoryType falcorType, uint32_t memoryTypeBits) const
    {
        uint32_t mask = mpApiData->vkMemoryTypeBits[(uint32_t)falcorType] & memoryTypeBits;
        assert(mask != 0);
        return bitScanForward(mask);
    }

    const VkPhysicalDeviceLimits& Device::getPhysicalDeviceLimits() const
    {
        return mpApiData->deviceLimits;
    }

    uint32_t Device::getDeviceVendorID() const
    {
        return mpApiData->properties.vendorID;
    }
}
