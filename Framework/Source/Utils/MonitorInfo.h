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

// #TODO Implement MonitorInfo cross-platform. GLFW?
#ifdef _WIN32
#include <vector>
#include "glm/gtc/type_ptr.hpp"
#include "glm/gtx/polar_coordinates.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Falcor
{
    /** A class to extract information about displays
    */
    class MonitorInfo
    {
    public:
        /** Description data structure
        */
        struct MonitorDesc
        {
            std::string mIdentifier;
            glm::vec2 mResolution;
            glm::vec2 mPhysicalSize;
            float mPpi;
            bool mIsPrimary;
        };

        /** Function to enumerate all monitors.
            This function is _not_ thread-safe
            \param monitorDescs An empty vector to receive the output list of monitor configurations.
        */
        static void getMonitorDescs(std::vector<MonitorDesc>& monitorDescs);

        /** Display information on currently connected monitors
        */
        static void displayMonitorInfo();
    };
}
#endif // _WIN32