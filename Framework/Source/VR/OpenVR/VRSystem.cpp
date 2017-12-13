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

#include "FalcorConfig.h"

#include "VRSystem.h"
#include "openvr.h"
#include "glm/glm.hpp"
#include "glm/gtx/transform.hpp"
#include "API/Device.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
#ifdef FALCOR_D3D12
    static vr::D3D12TextureData_t prepareSubmitData(const Texture::SharedConstPtr& pTex, RenderContext* pRenderCtx)
    {
        vr::D3D12TextureData_t submitTex;
        submitTex.m_pResource = pTex->getApiHandle();
        submitTex.m_pCommandQueue = pRenderCtx->getLowLevelData()->getCommandQueue();
        submitTex.m_nNodeMask = 0;
        return submitTex;
    }

    static vr::ETextureType getVrTextureType()
    {
        return vr::TextureType_DirectX12;
    }

#elif defined FALCOR_VK
    static vr::VRVulkanTextureData_t prepareSubmitData(const Texture::SharedConstPtr& pTex, RenderContext* pRenderCtx)
    {
        vr::VRVulkanTextureData_t data;
        data.m_nImage = (uint64_t)(VkImage)pTex->getApiHandle();
        data.m_pDevice = gpDevice->getApiHandle();
        data.m_pPhysicalDevice = gpDevice->getApiHandle();
        data.m_pInstance = gpDevice->getApiHandle();
        data.m_pQueue = pRenderCtx->getLowLevelData()->getCommandQueue();
        data.m_nQueueFamilyIndex = gpDevice->getApiCommandQueueType(LowLevelContextData::CommandQueueType::Direct);
        data.m_nWidth = pTex->getWidth();
        data.m_nHeight = pTex->getHeight();
        data.m_nFormat = getVkFormat(pTex->getFormat());
        data.m_nSampleCount = pTex->getSampleCount();

        return data;
    }

    static vr::ETextureType getVrTextureType()
    {
        return vr::TextureType_Vulkan;
    }
#else
#error VRSystem doesnt support the selected API backend
#endif


    VRSystem* VRSystem::spVrSystem = nullptr;

    // Private default constructor
    VRSystem::VRSystem() : mpHMD(0), mpCompositor(0), mLastError(0), mReadyToRender(false)
    {
    }

    VRSystem* VRSystem::start(bool enableVSync)
    {
        if(spVrSystem)
        {
            logWarning("Trying to reinitialize the VR system. Call is ignored");
            return spVrSystem;
        }

        // Create our VRSystem object and apply developer-specified parameters
        spVrSystem = new VRSystem;
        spVrSystem->mVSyncEnabled = enableVSync;
        
        // Initialize the HMD system and check for initialization errors
        vr::HmdError hmdError;
        spVrSystem->mpHMD = vr::VR_Init(&hmdError, vr::VRApplication_Scene);
        if(spVrSystem->mpHMD == NULL)
        {
            spVrSystem->mLastError = 1; // FIX
            spVrSystem->mLastErrorMsg = std::string(VR_GetVRInitErrorAsEnglishDescription(hmdError));
            cleanup();
            return spVrSystem;
        }

        // Initialize our compositor
        hmdError = vr::VRInitError_None;
        spVrSystem->mpCompositor = (vr::IVRCompositor*)vr::VR_GetGenericInterface(vr::IVRCompositor_Version, &hmdError);
        if(hmdError != vr::VRInitError_None)
        {
            spVrSystem->mLastError = 2; // FIX
            spVrSystem->mLastErrorMsg = std::string(VR_GetVRInitErrorAsEnglishDescription(hmdError));
            cleanup();
            return spVrSystem;
        }

        // // IVRCompositor::GetLastError has been removed. Errors are reported in the log.
        //// Check if the compositor has any error message to show
        //uint32_t errStrSize = spVrSystem->mpCompositor->GetLastError( NULL, 0 );
        //if (errStrSize > 1)
        //{
        //    char *buf = (char *)malloc( errStrSize );
        //    spVrSystem->mpCompositor->GetLastError( buf, errStrSize );
        //    spVrSystem->mLastError = 3; // FIX
        //    spVrSystem->mLastErrorMsg = std::string("Compositor Init Error: ") + std::string(buf);
        //    free( buf );
        //    return spVrSystem;
        //}

        // Initialize the class that can describe our play-size area.  If this fails, it's not fatal...  We just won't
        //    have any idea if our HMD has been set up or what the size of our play area is.
        hmdError = vr::VRInitError_None;
        spVrSystem->mpChaperone = (vr::IVRChaperone *)vr::VR_GetGenericInterface(vr::IVRChaperone_Version, &hmdError);
        if(!spVrSystem->mpChaperone)
        {
            spVrSystem->mpChaperone = 0;
        }

        vr::HmdColor_t col;
        col.r = 1.0f;
        col.g = 0.0f;
        col.b = 0.0f;
        col.a = 1.0f;
        spVrSystem->mpChaperone->SetSceneColor(col);

        // Initialize the class that can provide renderable models of controllers, etc.  If this fails, it's non-fatal,
        //    we just won't have access to internal geometry.
        spVrSystem->mpModels = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &hmdError);
        if(!spVrSystem->mpModels)
        {
            spVrSystem->mpModels = 0;
        }

        // Create a lens distortion VBO if we're in GL mode
        spVrSystem->createDistortionVBO();

        // Allocate some data we'll need over the lifetime of this object
        spVrSystem->mDevicePoses = new vr::TrackedDevicePose_t[vr::k_unMaxTrackedDeviceCount];
        if(!spVrSystem->mDevicePoses)
        {
            spVrSystem->mLastError = 4; // FIX
            spVrSystem->mLastErrorMsg = "Memory allocation error in VRSystem::create()!";
            cleanup();
            return spVrSystem;
        }

        return spVrSystem;
    }

    void VRSystem::initDisplayAndController(RenderContext::SharedPtr pRenderContext)
    {
        mpContext = pRenderContext;

        // Create a display/hmd object for our system
        spVrSystem->mDisplay = VRDisplay::create(spVrSystem->mpHMD, spVrSystem->mpModels);
        spVrSystem->mpHMDModel = spVrSystem->mDisplay->getRenderableModel();

        // Create controllers for our system
        spVrSystem->mControllers[0] = VRController::create(spVrSystem->mpHMD, spVrSystem->mpModels);
        spVrSystem->mControllers[1] = VRController::create(spVrSystem->mpHMD, spVrSystem->mpModels);

        // Create lighthouse trackers for our system
        spVrSystem->mTracker[0] = VRTrackerBox::create(spVrSystem->mpHMD, spVrSystem->mpModels);
        spVrSystem->mTracker[1] = VRTrackerBox::create(spVrSystem->mpHMD, spVrSystem->mpModels);

        // Create a play area
        spVrSystem->mPlayArea = VRPlayArea::create(spVrSystem->mpHMD, spVrSystem->mpChaperone);

        // All right!  Done with basic setup.  If we get this far, we should be ready and able to render
        //    (even if we have issues with controllers, etc).
        spVrSystem->mReadyToRender = true;
    }

    void VRSystem::cleanup(void)
    {
        if(spVrSystem)
        {
            if (spVrSystem->mDevicePoses)
            {
                delete [] spVrSystem->mDevicePoses;
                spVrSystem->mDevicePoses = nullptr;
            }
            vr::VR_Shutdown();
            safe_delete(spVrSystem);
        }
    }

    bool VRSystem::isReady(void)
    {
        return mReadyToRender;
    }


    uint32_t VRSystem::getError(std::string *errMessage)
    {
        if(errMessage != NULL)
        {
            *errMessage = mLastErrorMsg;
            mLastErrorMsg = "";
        }
        uint32_t tmpError = mLastError;
        mLastError = 0;
        return tmpError;
    }


    VRController::SharedPtr VRSystem::getControllerFromDeviceID(int32_t devID)
    {
        if(mControllers[0]->getDeviceID() == devID || mControllers[0]->getDeviceID() < 0) return mControllers[0];
        if(mControllers[1]->getDeviceID() == devID || mControllers[1]->getDeviceID() < 0) return mControllers[1];
        return nullptr;
    }

    VRTrackerBox::SharedPtr VRSystem::getTrackerFromDeviceID(int32_t devID)
    {
        if(mTracker[0]->getDeviceID() == devID) return mTracker[0];
        if(mTracker[1]->getDeviceID() == devID) return mTracker[1];
        return nullptr;
    }

    bool VRSystem::pollEvents(void)
    {
        bool processed = false;
        if(!mpHMD) return processed;

        vr::VREvent_t event;
        vr::TrackedDeviceClass curDevice;
        while(mpHMD->PollNextEvent(&event, sizeof(event)))
        {
            // Figure out what type of device we've got creating events
            curDevice = mpHMD->GetTrackedDeviceClass(event.trackedDeviceIndex);

            VRController::SharedPtr pController;
            VRTrackerBox::SharedPtr pTracker;

            if (curDevice == vr::TrackedDeviceClass_Controller)
            {
                pController = getControllerFromDeviceID(event.trackedDeviceIndex);
            }
            if (curDevice == vr::TrackedDeviceClass_TrackingReference)
            {
                pTracker = getTrackerFromDeviceID(event.trackedDeviceIndex);
            }


            // Process the event
            processed = true;
            switch(event.eventType)
            {
                // If we just attached a controller, get a representative model to use to display it.
            case vr::VREvent_TrackedDeviceActivated:
                // Make sure we have a model to render controllers
                if(curDevice == vr::TrackedDeviceClass_Controller)
                {
                    // Ensure we attach our controllers to the appropriate OpenVR IDs, if we've never seen this controller before
                    if(!pController)
                    {
                        pController = (mControllers[0]->getDeviceID() < 0) ? mControllers[0] : mControllers[1];
                        pController->updateOnActivate(event.trackedDeviceIndex);
                    }

                    // Check to see if we have a contoller model yet.  If not, get one.
                    if(!mpControlModel || !mpAxisModel)
                    {
                        mpControlModel = pController->getRenderableModel();
                        mpAxisModel = pController->getRenderableAxes();
                    }

                    printf("Controller %d activated; OpenVR device ID %d\n", (mControllers[0]->getDeviceID() == event.trackedDeviceIndex ? 0 : 1), event.trackedDeviceIndex);
                }
                else if(curDevice == vr::TrackedDeviceClass_TrackingReference)
                {
                    // Ensure we attach our tracker to the appropriate OpenVR IDs, if we've never seen this controller before
                    if(!pTracker)
                    {
                        pTracker = (mTracker[0]->getDeviceID() < 0) ? mTracker[0] : mTracker[1];
                        pTracker->updateOnActivate(event.trackedDeviceIndex);
                    }

                    // Check to see if we have a tracker/lighthouse model yet.  If not, get one.
                    if(!mpLighthouseModel)
                    {
//                        mpLighthouseModel = pTracker->getRenderableModel();
                    }

                    printf("Lighthouse tracker %d activated; OpenVR device ID %d\n", (mTracker[0]->getDeviceID() == event.trackedDeviceIndex ? 0 : 1), event.trackedDeviceIndex);
                }
                break;
            case vr::VREvent_TrackedDeviceDeactivated:
                if(pController)
                {
                    pController->updateActivatedState(false);
                }
                break;
            case vr::VREvent_TrackedDeviceUpdated:
                break;
            case vr::VREvent_ButtonPress:
                if(pController)
                {
                    pController->updateOnButtonEvent(1ull << event.data.controller.button, VRController::ButtonEvent::Press);
                }
                break;
            case vr::VREvent_ButtonUnpress:
                if(pController)
                {
                    pController->updateOnButtonEvent(1ull << event.data.controller.button, VRController::ButtonEvent::Unpress);
                }
                break;
            case vr::VREvent_ButtonTouch:
                if(pController)
                {
                    pController->updateOnButtonEvent(1ull << event.data.controller.button, VRController::ButtonEvent::Touch);
                }
                break;
            case vr::VREvent_ButtonUntouch:
                if(pController)
                {
                    pController->updateOnButtonEvent(1ull << event.data.controller.button, VRController::ButtonEvent::Untouch);
                }
                break;
            }
        }

        return processed;
    }

    void VRSystem::refreshTracking(void)
    {
        if(!mpHMD || !mpCompositor) return;

        // Query the compositor to return current positions of all tracked devices
        mpCompositor->WaitGetPoses(mDevicePoses, vr::k_unMaxTrackedDeviceCount, NULL, 0);

        // Now that we have the poses for all of our devices, cycle through them to update their state
        for(uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; i++)
        {
            if(mpHMD->IsTrackedDeviceConnected(i))
            {
                vr::TrackedDeviceClass curDevice = mpHMD->GetTrackedDeviceClass(i);
                if(curDevice == vr::TrackedDeviceClass_HMD)
                {
                    if(mDisplay)
                    {
                        mDisplay->updateOnNewPose(&mDevicePoses[i]); // vr::k_unTrackedDeviceIndex_Hmd] );
                    }
                }
                else if(curDevice == vr::TrackedDeviceClass_Controller)
                {
                    // Determine which contoller this is
                    VRController::SharedPtr curController = getControllerFromDeviceID(i);
                    if(curController)
                    {
                        curController->updateOnNewPose(&mDevicePoses[i], i);
                    }
                }
                else if(curDevice == vr::TrackedDeviceClass_TrackingReference)
                {
                    VRTrackerBox::SharedPtr curTracker = getTrackerFromDeviceID(i);
                    if(curTracker)
                    {
                        curTracker->updateOnNewPose(&mDevicePoses[i], i);
                    }
                }

                // Other types of devices?
            }
        }

    }

    void VRSystem::refresh()
    {
        if (isReady())
        {
            // Get the VR data
            pollEvents();
            refreshTracking();
        }
    }

    bool VRSystem::submit(VRDisplay::Eye whichEye, const Texture::SharedConstPtr& pDisplayTex, RenderContext* pRenderCtx)
    {
        if (!mpCompositor) return false;

        auto submitTex = prepareSubmitData(pDisplayTex, pRenderCtx);
        vr::Texture_t subTex;
        subTex.eType = getVrTextureType();
        subTex.handle = &submitTex;
        subTex.eColorSpace = isSrgbFormat(pDisplayTex->getFormat()) ? vr::EColorSpace::ColorSpace_Gamma : vr::EColorSpace::ColorSpace_Linear;

        mpCompositor->Submit((whichEye == VRDisplay::Eye::Right) ? vr::Eye_Right : vr::Eye_Left, &subTex, NULL);
        return true;
    }


    // Create a VBO that can be used in a separate pass to transform a flat image into the distorted image needed to
    //     be displayed on the HMD screen.  The compositor already does this for you, but if you want to see the 
    //     distorted image yourself on your non-HMD screen, you'll need this to do the transformation.
    void VRSystem::createDistortionVBO(void)
    {
        //// Declare internal structure type only needed in this method...
        //struct VertexDataLens
        //{
        //    float position[2];
        //    float texCoordRed[2];
        //    float texCoordGreen[2];
        //    float texCoordBlue[2];
        //};

        //// HACK!  This may only work for OpenGL renderers...  TBD.  (This code came from the OpenVR OpenGL sample app.)
        //if(!mpHMD) return;

        //uint32_t m_iLensGridSegmentCountH = 43;
        //uint32_t m_iLensGridSegmentCountV = 43;

        //float w = (float)(1.0 / float(m_iLensGridSegmentCountH - 1));
        //float h = (float)(1.0 / float(m_iLensGridSegmentCountV - 1));
        //float u, v = 0;
        //std::vector<VertexDataLens> vVerts(0);
        //std::vector<uint32_t> vIndices;
        //uint32_t a, b, c, d;

        //// Distortion vertex positions
        //for(int eye = 0; eye < 2; eye++)
        //{
        //    float Xoffset = (eye == 0) ? -1.0f : 0.0f;
        //    for(int y = 0; y<int(m_iLensGridSegmentCountV); y++)
        //    {
        //        for(int x = 0; x<int(m_iLensGridSegmentCountH); x++)
        //        {
        //            u = x*w; v = 1 - y*h;
        //            vr::DistortionCoordinates_t dc0 = mpHMD->ComputeDistortion((eye == 0) ? vr::Eye_Left : vr::Eye_Right, u, v);
        //            vVerts.push_back({{Xoffset + u, -1 + 2 * y*h}, {dc0.rfRed[0], 1 - dc0.rfRed[1]}, {dc0.rfGreen[0], 1 - dc0.rfGreen[1]}, {dc0.rfBlue[0], 1 - dc0.rfBlue[1]}});
        //        }
        //    }

        //    uint32_t offset = (eye == 0) ? 0 : (m_iLensGridSegmentCountH)*(m_iLensGridSegmentCountV);
        //    for(uint32_t y = 0; y < m_iLensGridSegmentCountV - 1; y++)
        //    {
        //        for(uint32_t x = 0; x < m_iLensGridSegmentCountH - 1; x++)
        //        {
        //            a = m_iLensGridSegmentCountH*y + x + offset;
        //            b = m_iLensGridSegmentCountH*y + x + 1 + offset;
        //            c = (y + 1)*m_iLensGridSegmentCountH + x + 1 + offset;
        //            d = (y + 1)*m_iLensGridSegmentCountH + x + offset;
        //            vIndices.push_back(a);
        //            vIndices.push_back(b);
        //            vIndices.push_back(c);

        //            vIndices.push_back(a);
        //            vIndices.push_back(c);
        //            vIndices.push_back(d);
        //        }
        //    }
        //}

        //SimpleModelImporter::VertexFormat vaoLayout;
        //vaoLayout.attribs.push_back({SimpleModelImporter::AttribType::Position, 2, AttribFormat::AttribFormat_F32});
        //vaoLayout.attribs.push_back({SimpleModelImporter::AttribType::User0, 2, AttribFormat::AttribFormat_F32});  // texcoord red
        //vaoLayout.attribs.push_back({SimpleModelImporter::AttribType::User1, 2, AttribFormat::AttribFormat_F32});  // texcoord green
        //vaoLayout.attribs.push_back({SimpleModelImporter::AttribType::User2, 2, AttribFormat::AttribFormat_F32});  // texcoord blue
        //mpLensDistortModel[0] = SimpleModelImporter::create(vaoLayout,
        //    uint32_t(vVerts.size()*sizeof(VertexDataLens)), &vVerts[0],
        //    uint32_t(vIndices.size()*sizeof(uint32_t) / 2), &vIndices[0]);
        //mpLensDistortModel[1] = SimpleModelImporter::create(vaoLayout,
        //    uint32_t(vVerts.size()*sizeof(VertexDataLens)), &vVerts[0],
        //    uint32_t(vIndices.size()*sizeof(uint32_t) / 2), &vIndices[vIndices.size() / 2]);

    }

    bool VRSystem::getTimeSinceLastVsync(float *pfSecondsSinceLastVsync, uint64_t *pulFrameCounter)
    {
        return mpHMD->GetTimeSinceLastVsync(pfSecondsSinceLastVsync, pulFrameCounter);
    }

#ifdef _DEBUG
    void VRSystem::checkInit()
    {
        if(spVrSystem == nullptr)
        {
            logWarning("VR system not initialized");
        }
    }
#endif

#ifdef FALCOR_VK
    std::vector<std::string> VRSystem::getRequiredVkInstanceExtensions()
    {
        uint32_t size = spVrSystem->mpCompositor->GetVulkanInstanceExtensionsRequired(nullptr, 0);
        std::vector<char> charVec(size);
        spVrSystem->mpCompositor->GetVulkanInstanceExtensionsRequired(charVec.data(), size);
        std::string str(charVec.data());

        std::vector<std::string> ext = splitString(str, " ");
        return ext;
    }

    std::vector<std::string> VRSystem::getRequiredVkDeviceExtensions(VkPhysicalDevice device)
    {
        uint32_t size = spVrSystem->mpCompositor->GetVulkanDeviceExtensionsRequired(device, nullptr, 0);
        std::vector<char> charVec(size);
        spVrSystem->mpCompositor->GetVulkanDeviceExtensionsRequired(device, charVec.data(), size);
        std::string str(charVec.data());

        std::vector<std::string> ext = splitString(str, " ");
        return ext;
    }
#endif
}