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
#pragma once
#include <string>

namespace Falcor
{
    namespace SceneKeys
    {
        // Values only used in the importer
#ifdef SCENE_IMPORTER
        static const char* kInclude = "include";

        // Not supported in exporter yet
        static const char* kLightProbes = "light_probes";

        // Keys for values in older scene versions that are not exported anymore
        static const char* kCamFovY = "fovY";
        static const char* kActivePath = "active_path";
        static const char* kAmbientIntensity = "ambient_intensity";
#endif

        static const char* kVersion = "version";
        static const char* kCameraSpeed = "camera_speed";
        static const char* kActiveCamera = "active_camera";
        static const char* kLightingScale = "lighting_scale";

        static const char* kName = "name";

        static const char* kModels = "models";
        static const char* kFilename = "file";
        static const char* kModelInstances = "instances";
        static const char* kTranslationVec = "translation";
        static const char* kRotationVec = "rotation";
        static const char* kScalingVec = "scaling";
        static const char* kActiveAnimation = "active_animation";

        static const char* kCameras = "cameras";
        static const char* kCamPosition = "pos";
        static const char* kCamTarget = "target";
        static const char* kCamUp = "up";
        static const char* kCamFocalLength = "focal_length";
        static const char* kCamDepthRange = "depth_range";
        static const char* kCamAspectRatio = "aspect_ratio";
        static const char* kPathLoop = "loop";
        static const char* kPathFrames = "frames";
        static const char* kFrameTime = "time";

        static const char* kLights = "lights";
        static const char* kType = "type";
        static const char* kDirLight = "dir_light";
        static const char* kPointLight = "point_light";
        static const char* kLightIntensity = "intensity";
        static const char* kLightShadow = "shadow";
        static const char* kLightOpeningAngle = "opening_angle";
        static const char* kLightPenumbraAngle = "penumbra_angle";
        static const char* kLightPos = "pos";
        static const char* kLightDirection = "direction";

        static const char* kPaths = "paths";
        static const char* kAttachedObjects = "attached_objects";
        static const char* kModelInstance = "model_instance";
        static const char* kLight = "light";
        static const char* kCamera = "camera";

        static const char* kUserDefined = "user_defined";
    };
}