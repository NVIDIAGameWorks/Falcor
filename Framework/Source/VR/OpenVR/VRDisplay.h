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

// This is wrapper class around OpenVR displays (aka HMDs).  This is a fairly
//     important class, providing access to rendering view and project matrices
//     (for both eyes), plus the recommended render target size.
//
//  Chris Wyman (12/15/2015)

#pragma once

#include "Framework.h"
#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "API/FBO.h"
#include "API/RenderContext.h"
#include "Graphics/Program/Program.h"
#include "API/Texture.h"
#include "Graphics/Model/Model.h"

// Forward declare OpenVR system class types to remove "openvr.h" dependencies from Falcor headers
namespace vr
{
    class IVRSystem;              // A generic system with basic API to talk with a VR system
    class IVRCompositor;          // A class that composite rendered results with appropriate distortion into the HMD
    class IVRRenderModels;        // A class giving access to renderable geometry for things like controllers
    class IVRChaperone;
    class IVRRenderModels;
    struct TrackedDevicePose_t;
}

namespace Falcor
{
    // Forward declare the VRSystem class to avoid header cycles.
    class VRSystem;
    class Camera;

    /** High-level OpenVR display abstraction
    */
    class VRDisplay
    {
    public:
        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Types and enums
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Declare standard Falcor shared pointer types.
        using SharedPtr = std::shared_ptr<VRDisplay>;
        using SharedConstPtr = std::shared_ptr<const VRDisplay>;

        enum class Eye
        {
            Right = 0,
            Left = 1,
        };


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Constructors & destructors
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Controller constructor.  Automatically called by other wrapper classes.
        static SharedPtr create( vr::IVRSystem *vrSys, vr::IVRRenderModels *modelClass = 0 );


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Simple accessor methods
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if the HMD is correctly being tracked
        bool isTracking( void ) const { return mIsTracking; }

        // Gets position of center of HMD in world-space
        glm::vec3 getPosition( void ) const { return mPosition; }

        // Gets model-space to world-space transform for the HMD
        glm::mat4 getWorldMatrix( void ) const { return mWorldMat; }

        // Gets the projection matrix for a specified eye
        glm::mat4 getProjectionMatrix( VRDisplay::Eye whichEye ) const { return mProjMats[(uint32_t)whichEye]; }
        
        // Gets the offset matrix from center of HMD to specified eye.
        glm::mat4 getOffsetMatrix(VRDisplay::Eye whichEye) const { return mOffsetMats[(uint32_t)whichEye]; }

        // Get's the view matrix for specified eye (equiv. to getOffsetMatrix(whichEye) * getToWorldMatrix())
        glm::mat4 getViewMatrix(VRDisplay::Eye whichEye) const { return mViewMats[(uint32_t)whichEye]; }

        // Gets the HMD's native resolution (i.e., the size it shows up in Windows as a "monitor")
        glm::ivec2 getNativeResolution( void ) const { return mNativeDisplayRes; }

        // Returns the recommended size to render at (*independently* for each eye)
        glm::ivec2 getRecommendedRenderSize( void ) const { return mRecRenderSz; }

        float getFovY() const { return mFovY; }
        float getAspectRatio() const { return mAspectRatio; }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Mutator methods
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Set near & far values for the projection matrix.  Without calling this, defaults to [0.01...20.0]
        void setDepthRange( float nearZ, float farZ );

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // HMD geometric accessors.  (Note: HMD models are often pretty generic, since it's not usually 
        //       useful to see them.  This may get a model, it just may not be representative of the
        //       HMD that is actually being used...)
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Get a Falcor model representing the lighthouse tracker.  All trackers will return the same geometry,
        //    though you *can* pass your own texture in if you'd like to override OpenVR's baked AO texture.
        Model::SharedPtr   getRenderableModel( Texture::SharedPtr overrideTexture = nullptr );

        // When rendering, if you directly need the texture for the model, you can grab it here
        Texture::SharedPtr getRenderableModelTexture( void ) { return mpModelTexture; }


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Wrapper-specific methods;  Shouldn't need to use outside of OpenVR wrapper code, generally.
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // When the HMD is updated, the new pose should be passed in here to update all our positioning state
        void updateOnNewPose( vr::TrackedDevicePose_t *newPose );

    private:
        bool                 mIsTracking;
        glm::mat4            mOffsetMats[2];
        glm::mat4            mViewMats[2];
        glm::mat4            mProjMats[2];
        glm::mat4            mWorldMat;
        glm::vec3            mPosition;
        glm::ivec2           mRecRenderSz;
        glm::ivec2           mNativeDisplayRes;
        glm::ivec2           mCompositorOffset;
        glm::vec2            mNearFarPlanes;

        int32_t              mDeviceID;
        vr::IVRRenderModels *mpRenderModels;
        vr::IVRSystem       *mpVrSys;
        std::string          mModelName;

        Texture::SharedPtr   mpModelTexture;
        Model::SharedPtr     mpRenderableModel;

        float mAspectRatio;
        float mFovY;
    };

} // end namespace Falcor
