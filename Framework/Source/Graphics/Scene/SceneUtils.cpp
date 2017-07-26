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
#include "Framework.h"
#include "SceneUtils.h"
#include "Scene.h"
#include "API/ConstantBuffer.h"

namespace Falcor
{
    void getSceneLightString(const Scene* pScene, std::string& lights)
    {
        lights.clear();
        for(uint32_t i = 0; i < pScene->getLightCount(); i++)
        {
            auto pLight = pScene->getLight(i);
            lights += " " + pLight->getName() + ",";
        }

        if(lights.size() > 0)
        {
            // Remove the last ','
            lights = lights.erase(lights.length() - 1);
        }
    }

    void setSceneLightsIntoConstantBuffer(const Scene* pScene, ConstantBuffer* pBuffer)
    {
        // Set all the lights
        for(uint32_t i = 0; i < pScene->getLightCount(); i++)
        {
            auto pLight = pScene->getLight(i);
            pLight->setIntoConstantBuffer(pBuffer, pLight->getName());
        }
        pBuffer->setVariable("gAmbient", pScene->getAmbientIntensity());
    }
}