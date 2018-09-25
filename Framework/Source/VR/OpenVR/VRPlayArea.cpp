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
#include "VRPlayArea.h"

using namespace Falcor;


VRPlayArea::SharedPtr VRPlayArea::create( vr::IVRSystem *vrSys, vr::IVRChaperone *chaperone )
{
    SharedPtr play = SharedPtr( new VRPlayArea );
    play->mpChaperone = chaperone;
    play->mpVrSys = vrSys;
    return play;
}

bool       VRPlayArea::isCalibrated( void ) const
{
    if ( !mpChaperone ) return false;
    if ( mpChaperone->GetCalibrationState() != vr::ChaperoneCalibrationState_OK )
        return false;
    return true;
}

bool       VRPlayArea::isPartiallyCalibrated( void ) const
{
    if ( !mpChaperone ) return false;
    vr::ChaperoneCalibrationState state = mpChaperone->GetCalibrationState();
    if ( state < vr::ChaperoneCalibrationState_Error )
        return true;
    return false;
}

glm::vec2  VRPlayArea::getSize( void ) const
{
    float hasData = false;
    float x, z;
    if ( mpChaperone )
    {
        hasData = mpChaperone->GetPlayAreaSize( &x, &z );
    }
    // 0.5 so that the room goes from [-x...x] x [0...2.5] x [-y...y]
    return hasData ? glm::vec2( 0.5*x, 0.5*z ) : glm::vec2( 0 );
}
