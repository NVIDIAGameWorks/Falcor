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
#include "ProjectTemplate.h"
#include "glm/gtc/epsilon.hpp"

void ProjectTemplate::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    pGui->addText("Hello from ProjectTemplate");
    if (pGui->addButton("Click Here"))
    {
        msgBox("Now why would you do that?");
    }
}

vec3 sphericalCrdToDir(vec2 uv)
{
    static const float PI = 3.1415926535f;
    static const float PI_2 = 2.0f * 3.1415926535f;

    float phi = PI * uv.y;
    float theta = PI_2 * uv.x - (PI / 2.0f);

    vec3 dir;
    dir.x = sin(phi) * sin(theta);
    dir.y = cos(phi);
    dir.z = sin(phi) * cos(theta);

    return -dir;
}

vec2 dirToSphericalCrd(vec3 direction)
{
    static const float PI = 3.14159265f;
    vec3 p = normalize(direction);
    vec2 uv;
    uv.x = (1.0f + atan2(-p.z, p.x) / PI) * 0.5f;
    uv.y = 1.0f - (-acos(p.y) / PI);
    return uv;
}

bool intersectRaySphere(vec3 rayPosW, vec3 rayDir, vec3 spherePosW, float sphereRadius,
    vec3& intersectionPosW)
{
    float3 m = rayPosW - spherePosW;
    float b = dot(m, rayDir);
    float c = dot(m, m) - (sphereRadius * sphereRadius);

    // If ray origin is outside sphere (c > 0) and ray is pointing away from sphere (b > 0)
    // For now assume input always produces valid intersection
    // if(c > 0.0f && b > 0.0f) return false;

    float discr = b * b - c;
    // Negative discriminant means ray missed sphere
    // if(discr < 0.0f) return false;

    float t = -b - sqrt(discr);

    // t will be negative if origin is inside sphere, 
    // take the abs since we want the position on the sphere's surface
    intersectionPosW = normalize(rayPosW + abs(t) * rayDir);

    return true;
}

#include "glm/gtc/random.hpp"

void ProjectTemplate::onLoad(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext)
{
    vec3 output0;
    intersectRaySphere(vec3(2.5f, 0, 0), normalize(vec3(.707f, .707f, 0)), vec3(), 5.0f, output0);
    
    vec3 output1;
    intersectRaySphere(vec3(2.5f, 0, 0), normalize(vec3(0, 1, 0)), vec3(), 5.0f, output1);
    
    vec3 output2;
    intersectRaySphere(vec3(2.5f, 0, 0), normalize(vec3(1, 0, 0)), vec3(), 5.0f, output2);



    //for(uint32_t i = 0; i < 1000000; i++)
    //{
    //    vec3 dir = normalize(linearRand(vec3(-10), vec3(10)));
    //    vec2 uv = dirToSphericalCrd(dir);
    //    vec3 outDir = sphericalCrdToDir(uv);
    //    vec2 outUV = dirToSphericalCrd(outDir);

    //    vec2 uvError = abs(outUV - uv);
    //    vec3 dirError = abs(dir - outDir);

    //    if (any(epsilonNotEqual(uvError, vec2(0), vec2(0.0001f))) || 
    //        any(epsilonNotEqual(dirError, vec3(0), vec3(0.0001f))))
    //    {
    //        __debugbreak();
    //    }
    //}

}

void ProjectTemplate::onFrameRender(SampleCallbacks* pSample, RenderContext::SharedPtr pRenderContext, Fbo::SharedPtr pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
}

void ProjectTemplate::onShutdown(SampleCallbacks* pSample)
{
}

bool ProjectTemplate::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    return false;
}

bool ProjectTemplate::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return false;
}

void ProjectTemplate::onDataReload(SampleCallbacks* pSample)
{

}

void ProjectTemplate::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    ProjectTemplate::UniquePtr pRenderer = std::make_unique<ProjectTemplate>();
    SampleConfig config;
    config.windowDesc.title = "Falcor Project Template";
    config.windowDesc.resizableWindow = true;
    Sample::run(config, pRenderer);
    return 0;
}
