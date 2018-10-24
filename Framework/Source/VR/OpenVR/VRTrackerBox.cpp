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

using namespace Falcor;

// Private namespace
namespace
{
    // Matrix to convert OpenVR matrix to OpenGL matrix.
    glm::mat4 convertOpenVRMatrix34( vr::HmdMatrix34_t mat )
    {
        return glm::mat4( mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0f,
                          mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0f,
                          mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0f,
                          mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f );
    }
}


VRTrackerBox::SharedPtr VRTrackerBox::create( vr::IVRSystem *vrSys, vr::IVRRenderModels *modelClass )
{
    SharedPtr ctrl = SharedPtr( new VRTrackerBox );

    // Setup default tracker (inactive)
    ctrl->mIsActive = false;
    ctrl->mTrackerCenter = glm::vec3( 0.0f );
    ctrl->mWorldMatrix = glm::mat4();

    // Setup internal OpenVR state
    ctrl->mpVrSys = vrSys;
    ctrl->mpRenderModels = modelClass;
    ctrl->mDeviceID = -1; // not yet known.

    return ctrl;
}

Model::SharedPtr VRTrackerBox::getRenderableModel( Texture::SharedPtr overrideTexture )
{
    // If we already got a model, it won't change.  so go ahead and return the same one.
    if ( mpRenderableModel )
    {
        return mpRenderableModel;
    }

    // Go ahead and pull our model from the OpenVR system
    vr::RenderModel_t* pModel = nullptr;

    vr::EVRRenderModelError modelErr;
    do
    {
        modelErr = mpRenderModels->LoadRenderModel_Async(mModelName.c_str(), &pModel);
    }
    while(modelErr == vr::VRRenderModelError_Loading);

    if (modelErr != vr::VRRenderModelError_None || !pModel)
        return nullptr;

    // create and populate the texture
    mpModelTexture = overrideTexture;
    vr::RenderModel_TextureMap_t *pTexture = nullptr;

    vr::EVRRenderModelError texErr;
    do
    {
        texErr = mpRenderModels->LoadTexture_Async(pModel->diffuseTextureId, &pTexture);
    }while(texErr == vr::VRRenderModelError_Loading);

    if (texErr != vr::VRRenderModelError_None || !pTexture)
        return nullptr;

    if ( !overrideTexture )
        mpModelTexture = Texture::create2D( pTexture->unWidth, pTexture->unHeight,
        ResourceFormat::RGBA8Unorm, 1, Texture::kMaxPossible,
        (const void *) pTexture->rubTextureMapData );

    // OpenVR models use uint16_t index buffers.  Convert to uint32_t for Falcor.
    uint32_t *newIdxBuf = (uint32_t *) malloc( pModel->unTriangleCount * 3 * sizeof( uint32_t ) );
    for ( uint32_t i = 0; i < pModel->unTriangleCount * 3; i++ )
        newIdxBuf[i] = pModel->rIndexData[i];

    // Use the SimpleModelImporter to create a Falcor model from memory.
    SimpleModelImporter::VertexFormat vertLayout;
    vertLayout.attribs.push_back( { SimpleModelImporter::AttribType::Position, 3, AttribFormat::AttribFormat_F32 } );
    vertLayout.attribs.push_back( { SimpleModelImporter::AttribType::Normal, 3, AttribFormat::AttribFormat_F32 } );
    vertLayout.attribs.push_back( { SimpleModelImporter::AttribType::TexCoord, 2, AttribFormat::AttribFormat_F32 } );
    Model::SharedPtr ctrlModel = SimpleModelImporter::create( vertLayout,
                                                              sizeof( vr::RenderModel_Vertex_t ) * pModel->unVertexCount,
                                                              pModel->rVertexData,
                                                              pModel->unTriangleCount * 3 * sizeof( uint32_t ),
                                                              newIdxBuf,
                                                              mpModelTexture );

    // Free our temporary memory
    free( newIdxBuf );

    return ctrlModel;
}

void VRTrackerBox::updateOnActivate( int32_t deviceID )
{
    // Remember which device ID we are (though this is transient; OpenVR changes it as new devices are added)
    mDeviceID = deviceID;
    mIsActive = true;

    // See if this device has a renderable model
    uint32_t unRequiredBufferLen = mpVrSys->GetStringTrackedDeviceProperty( mDeviceID, vr::Prop_RenderModelName_String, NULL, 0, NULL );
    if ( unRequiredBufferLen != 0 )
    {
        // If so, get a string describing it
        char *pchBuffer = new char[unRequiredBufferLen];
        unRequiredBufferLen = mpVrSys->GetStringTrackedDeviceProperty( mDeviceID, vr::Prop_RenderModelName_String, pchBuffer, unRequiredBufferLen, NULL );
        mModelName = pchBuffer;
        delete[] pchBuffer;
    }
}

void VRTrackerBox::updateOnNewPose( vr::TrackedDevicePose_t *newPose, int32_t deviceID )
{
    // Can't update pose if things aren't initialized properly
    if ( !mpVrSys || !newPose ) return;

    // Set some internal active/inactive state
    mIsActive = newPose->bDeviceIsConnected && newPose->bPoseIsValid;
    mDeviceID = deviceID;

    // Nothing is really valid beyond this point...
    if ( !newPose->bPoseIsValid ) return;

    mWorldMatrix = convertOpenVRMatrix34( newPose->mDeviceToAbsoluteTracking );
    mTrackerCenter = glm::vec3( mWorldMatrix * glm::vec4( 0, 0, 0, 1 ) );
}
