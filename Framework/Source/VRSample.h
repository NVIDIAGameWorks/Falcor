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
#include "Sample.h"
#include "VRRenderer.h"

namespace Falcor
{
    
    /** Bootstrapper class for Falcor
        Call Sample::run() to start the sample.
        The render loop will then call the user's Renderer object
    */
    class VRSample : public Sample
    {
    public:
        /** Entry-point to Sample. User should call this to start processing.
            On Windows, command line args will be retrieved and parsed even if not passed through this function.
            On Linux, this function is the only way to feed the sample command line args.

            \param[in] config Requested sample configuration
            \param[in] pRenderer The user's renderer
            \param[in] argc Optional. Number of command line arguments
            \param[in] argv Optional. Array of command line arguments
        */
        static void run(const SampleConfig& config, VRRenderer::UniquePtr& pRenderer);

        virtual ~VRSample() {}
    protected:
        void renderFrame() override;

        VRSample(Renderer::UniquePtr& pRenderer) : Sample(pRenderer) {}
        VRSample(const VRSample&) = delete;
        VRSample& operator=(const VRSample&) = delete;
    };
};
