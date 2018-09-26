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
#include "VRController.h"
#include "VRTrackerBox.h"
#include "VRDisplay.h"
#include "Graphics/Camera/Camera.h"

namespace Falcor
{
    // Private namespace
    namespace
    {
        // Matrix to convert OpenVR matrix to OpenGL matrix.
        glm::mat4 convertOpenVRMatrix34(vr::HmdMatrix34_t mat)
        {
            return glm::mat4(mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0f,
                mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0f,
                mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0f,
                mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f);
        }

        // Matrix to convert OpenVR matrix to OpenGL matrix.
        glm::mat4 convertOpenVRMatrix44(vr::HmdMatrix44_t mat)
        {
#ifdef FALCOR_VK
            // Vulkan clip-space is +Y down
            const float m11 = -mat.m[1][1];
#else
            const float m11 = mat.m[1][1];
#endif
            return glm::mat4(
                mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
                mat.m[0][1],         m11, mat.m[2][1], mat.m[3][1],
                mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
                mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]);
        }
    }

    VRDisplay::SharedPtr VRDisplay::create(vr::IVRSystem *vrSys, vr::IVRRenderModels *modelClass)
    {
        SharedPtr ctrl = SharedPtr(new VRDisplay);

        // Setup default tracker (inactive)
        ctrl->mIsTracking = false;
        ctrl->mPosition = glm::vec3(0.0f);
        ctrl->mWorldMat = glm::mat4();
        ctrl->mNearFarPlanes = glm::vec2(0.01f, 20.0f);

        // Setup internal OpenVR state
        ctrl->mpVrSys = vrSys;
        ctrl->mpRenderModels = modelClass;
        ctrl->mDeviceID = vr::k_unTrackedDeviceIndex_Hmd;

        ctrl->mOffsetMats[(uint32_t)Eye::Left] = glm::inverse(convertOpenVRMatrix34(vrSys->GetEyeToHeadTransform(vr::Eye_Left)));
        ctrl->mOffsetMats[(uint32_t)Eye::Right] = glm::inverse(convertOpenVRMatrix34(vrSys->GetEyeToHeadTransform(vr::Eye_Right)));

        // Grab guesses for projection matricies, so we have *something* in case the user asks for a matrix prior to specifying a near/far plane
        //   -> We're totally guessing at near/far planes here.  User needs to set near/far planes for real on their own.
        ctrl->mProjMats[(uint32_t)Eye::Left] = convertOpenVRMatrix44(vrSys->GetProjectionMatrix(vr::Eye_Left, ctrl->mNearFarPlanes.x, ctrl->mNearFarPlanes.y));
        ctrl->mProjMats[(uint32_t)Eye::Right] = convertOpenVRMatrix44(vrSys->GetProjectionMatrix(vr::Eye_Right, ctrl->mNearFarPlanes.x, ctrl->mNearFarPlanes.y));

        // Get the recommended per-eye render size
        uint32_t recSize[2];
        vrSys->GetRecommendedRenderTargetSize(&recSize[0], &recSize[1]);
        ctrl->mRecRenderSz = glm::ivec2(recSize[0], recSize[1]);
        ctrl->mAspectRatio = float(recSize[0]) / float(recSize[1]);

        float recTanFovY = ctrl->mProjMats[0][1][1];
        float tanFovY = 1 / recTanFovY;
        float fovY = 2 * atan(tanFovY);
        ctrl->mFovY = fovY;

        // // GetWindowBounds has been moved to IVRDisplayComponent
        //
        //// Get the native screen size (and the compositor offset, which should be (0,0) if rendering on the HMD in 'fullscreen' mode)
        //int32_t  screenPos[2];
        //uint32_t dispSize[2];
        //ctrl->GetWindowBounds( &screenPos[0], &screenPos[1], &dispSize[0], &dispSize[1] );
        //ctrl->mNativeDisplayRes = glm::ivec2( dispSize[0], dispSize[1] );
        //ctrl->mCompositorOffset = glm::ivec2( screenPos[0], screenPos[1] );


        // See if this device has a renderable model
        uint32_t unRequiredBufferLen = ctrl->mpVrSys->GetStringTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_RenderModelName_String, NULL, 0, NULL);
        if(unRequiredBufferLen != 0)
        {
            // If so, get a string describing it
            char *pchBuffer = new char[unRequiredBufferLen];
            unRequiredBufferLen = ctrl->mpVrSys->GetStringTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_RenderModelName_String, pchBuffer, unRequiredBufferLen, NULL);
            ctrl->mModelName = pchBuffer;
            delete[] pchBuffer;
        }


        return ctrl;
    }

    void VRDisplay::setDepthRange(float nearZ, float farZ)
    {
        mNearFarPlanes = glm::vec2(nearZ, farZ);
        mProjMats[(uint32_t)Eye::Left] = convertOpenVRMatrix44(mpVrSys->GetProjectionMatrix(vr::Eye_Left, mNearFarPlanes.x, mNearFarPlanes.y));
        mProjMats[(uint32_t)Eye::Right] = convertOpenVRMatrix44(mpVrSys->GetProjectionMatrix(vr::Eye_Right, mNearFarPlanes.x, mNearFarPlanes.y));
    }

    void VRDisplay::updateOnNewPose(vr::TrackedDevicePose_t *newPose)
    {
        if(!newPose->bPoseIsValid)
        {
            mIsTracking = false;
            return;
        }
        mIsTracking = true;

        mWorldMat = glm::inverse(convertOpenVRMatrix34(newPose->mDeviceToAbsoluteTracking));

        // Since the matrix is inverted (i.e., GL camera is *truly* at (0,0,0) and we're moving the scene instead), 
        //     we can't simply apply the matrix to (0,0,0) as with other matrices.  We need to do that, then invert
        //     the translation.
        glm::vec4 camTransform = mWorldMat * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        mPosition = glm::vec3(-camTransform.x, -camTransform.y, -camTransform.z);

        // In case the user needs a view matrix frequently, premultiply it here.
        mViewMats[0] = mOffsetMats[0] * mWorldMat;
        mViewMats[1] = mOffsetMats[1] * mWorldMat;

    }

    Model::SharedPtr VRDisplay::getRenderableModel(Texture::SharedPtr overrideTexture)
    {
        // If we already got a model, it won't change.  so go ahead and return the same one.
        if(mpRenderableModel)
        {
            return mpRenderableModel;
        }

        // Go ahead and pull our model from the OpenVR system
        vr::RenderModel_t* pModel = nullptr;

        vr::EVRRenderModelError modelErr;
        do
        {
            modelErr = mpRenderModels->LoadRenderModel_Async(mModelName.c_str(), &pModel);
        } while(modelErr == vr::VRRenderModelError_Loading);

        if(modelErr != vr::VRRenderModelError_None)
            return nullptr;

        // create and populate the texture
        mpModelTexture = overrideTexture;
        vr::RenderModel_TextureMap_t *pTexture = nullptr;

        vr::EVRRenderModelError texErr;
        do
        {
            texErr = mpRenderModels->LoadTexture_Async(pModel->diffuseTextureId, &pTexture);
        } while(texErr == vr::VRRenderModelError_Loading);

        if(texErr != vr::VRRenderModelError_None)
            return nullptr;

        if(!overrideTexture)
            mpModelTexture = Texture::create2D(pTexture->unWidth, pTexture->unHeight,
            ResourceFormat::RGBA8Unorm, 1, Texture::kMaxPossible,
            (const void *)pTexture->rubTextureMapData);

        // OpenVR models use uint16_t index buffers.  Convert to uint32_t for Falcor.
        uint32_t *newIdxBuf = (uint32_t *)malloc(pModel->unTriangleCount * 3 * sizeof(uint32_t));
        for(uint32_t i = 0; i < pModel->unTriangleCount * 3; i++)
            newIdxBuf[i] = pModel->rIndexData[i];

        // Use the SimpleModelImporter to create a Falcor model from memory.
        SimpleModelImporter::VertexFormat vertLayout;
        vertLayout.attribs.push_back({SimpleModelImporter::AttribType::Position, 3, AttribFormat::AttribFormat_F32});
        vertLayout.attribs.push_back({SimpleModelImporter::AttribType::Normal, 3, AttribFormat::AttribFormat_F32});
        vertLayout.attribs.push_back({SimpleModelImporter::AttribType::TexCoord, 2, AttribFormat::AttribFormat_F32});
        Model::SharedPtr ctrlModel = SimpleModelImporter::create(vertLayout,
            sizeof(vr::RenderModel_Vertex_t) * pModel->unVertexCount,
            pModel->rVertexData,
            pModel->unTriangleCount * 3 * sizeof(uint32_t),
            newIdxBuf,
            mpModelTexture);

        // Free our temporary memory
        free(newIdxBuf);

        return ctrlModel;
    }
}
