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

// This is wrapper class around OpenVR tracker boxes (i.e., Vive's Lighthouses).
//     It gives position, active status, and a renderable model of the tracker.
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

    /** High-level OpenVR controller abstraction
    */
    class VRTrackerBox
    {
    public:
        // Declare standard Falcor shared pointer types.
        using SharedPtr = std::shared_ptr<VRTrackerBox>;
        using SharedConstPtr = std::shared_ptr<const VRTrackerBox>;

        // Controller constructor.  Automatically called by other wrapper classes.
        static SharedPtr create( vr::IVRSystem *vrSys, vr::IVRRenderModels *modelClass = 0 );

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Simple accessor methods
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if the controller is active an/or tracking.
        bool isActive( void ) const { return mIsActive; }

        // Get a matrix transforming from model/object to world coordinates based on tracking data
        glm::mat4 getToWorldMatrix( void ) const { return mWorldMatrix; }

        // Get the center of the controller in world space
        glm::vec3 getPosition( void ) const { return mTrackerCenter; }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Tracker box geometric accessors
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Get a Falcor model representing the lighthouse tracker.  All trackers will return the same geometry,
        //    though you *can* pass your own texture in if you'd like to override OpenVR's baked AO texture.
        Model::SharedPtr   getRenderableModel( Texture::SharedPtr overrideTexture = nullptr );

        // When rendering, if you directly need the texture for the model, you can grab it here
        Texture::SharedPtr getRenderableModelTexture( void ) { return mpModelTexture; }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Wrapper-specific methods;  Shouldn't need to use outside of OpenVR wrapper code, generally.
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // When the tracker detects it has moved (or it can no longer track correctly), this gets called
        void updateOnEnvironmentChange( void ) {} // TBD

        // When a new controller pose is provided by OpenVR, call this to update state viewable by above
        //     accessors.  deviceID is OpenVR's ID for the objects it tracks.
        void updateOnNewPose( vr::TrackedDevicePose_t *newPose, int32_t deviceID );

        // When the controller first activates, there'a a bunch of state info about the controller
        //     OpenVR has that we don't -- the name of the model inside the DLL, and other controller
        //     properties.  When the controller is first activated, calling this method allows us
        //     to know what we're dealing with.  Without calling this, some more advanced properties
        //     of this class (i.e., getting the model geometry) won't work correctly.
        void updateOnActivate( int32_t deviceID );

        // If we need to query OpenVR about this controller, you need its internal OpenVR deviceID
        int32_t getDeviceID( void ) const { return mDeviceID; }

    private:
        bool       mIsActive;            // Is the tracker currently active?
        glm::vec3  mTrackerCenter;       // The center of the controller
        glm::mat4  mWorldMatrix;         // The model-to-world matrix for the controller

        Model::SharedPtr mpRenderableModel;  // A Falcor renderable model.  Can be NULL if OpenVR has no model for the tracker!
        Texture::SharedPtr mpModelTexture;

        // Internals for accessing OpenVR state.  Not user accessible
        vr::IVRRenderModels *mpRenderModels;
        vr::IVRSystem       *mpVrSys;
        int32_t              mDeviceID;
        std::string          mModelName;
       
    };

}
