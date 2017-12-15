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

// This is wrapper class around OpenVR play areas, and can provide information
//     about the play area size (to appropriately bound user interactions),
//     as well as determining id the play area is fully set up and calibrated
//     (otherwise you can't really trust what it says).
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
    class VRPlayArea
    {
    public:
        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Types and enums
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Declare standard Falcor shared pointer types.
        using SharedPtr = std::shared_ptr<VRPlayArea>;
        using SharedConstPtr = std::shared_ptr<const VRPlayArea>;

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Constructors & destructors
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Controller constructor.  Automatically called by other wrapper classes.
        static SharedPtr create( vr::IVRSystem *vrSys, vr::IVRChaperone *chaperone );

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Simple accessor methods
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Returns if OpenVR thinks it is calibrated.  Partially calibrated means there's enough info to
        //    run, but they not know where the play area is or may think they've moved since calibration
        bool       isCalibrated( void ) const;
        bool       isPartiallyCalibrated( void ) const;

        // Get's the area available for movement.  Returns vec2(a,b).  If no play area is available, a = b = 0.
        //    Otherwise the play area's x-bounds of the scene go from [-a..a], the z-bounds go from [-b..b], and
        //    the y-bounds go from [0..room-height].  I don't see any definition of the height anywhere, but it
        //    looks to be about 2.5 meters, despite the actual room height.
        glm::vec2  getSize( void ) const;

    private:
        vr::IVRSystem    *mpVrSys;
        vr::IVRChaperone *mpChaperone;
    };

} // end namespace Falcor
