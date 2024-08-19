/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#include "OverlaySamplePass.h"
#include "RenderGraph/RenderPassStandardFlags.h"

// We'll be directly using ImGui to draw the overlay UI.
#include "imgui.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, OverlaySamplePass>();
    ScriptBindings::registerBinding(OverlaySamplePass::registerBindings);
}

namespace
{
const ChannelList kInputChannels = {
    {"input", "", "Input buffer", true, ResourceFormat::RGBA32Float},
};

const ChannelList kOutputChannels = {
    {"output", "", "Output buffer of the solution", false, ResourceFormat::RGBA32Float},
};
} // namespace

OverlaySamplePass::OverlaySamplePass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {}

Properties OverlaySamplePass::getProperties() const
{
    Properties props;
    return props;
}

RenderPassReflection OverlaySamplePass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void OverlaySamplePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Just copy the input to the output.
    auto src = renderData.getTexture(kInputChannels[0].name);
    auto dst = renderData.getTexture(kOutputChannels[0].name);
    mFrameDim.x = dst->getWidth();
    mFrameDim.y = dst->getHeight();
    if (src)
    {
        pRenderContext->blit(src->getSRV(), dst->getRTV());
    }
    mFrameCount++;
}

void OverlaySamplePass::renderOverlayUI(RenderContext* pRenderContext)
{
    // This callback occurs after the "renderUI" callback, and will be triggered even when the dropdown is closed.
    // Rather than drawing to a widget, we work directly with the ImGui draw list.

    float margin = 50.f;

    float2 frameMin = float2(0.0) + margin;
    float2 frameMax = float2(mFrameDim) - margin;
    float2 frameSize = frameMax - frameMin;

    // Get the background draw list
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();

    // Create a rectangular frame with a red border.
    drawList->AddRect(frameMin, frameMax, ImColor(float4(1.0, 0.0, 0.0, 0.0)));

    int primType = 0;

    // Inside that retangle, create a 5x3 grid of rectangles.
    for (int j = 0; j < 3; j++)
    {
        for (int i = 0; i < 5; i++)
        {
            float2 rectMin = frameMin + frameSize * float2(float(i + 0) / 5.0f, float(j + 0) / 3.0f);
            float2 rectMax = frameMin + frameSize * float2(float(i + 1) / 5.0f, float(j + 1) / 3.0f);
            rectMin += margin;
            rectMax -= margin;
            drawList->AddRect(rectMin, rectMax, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));

            // Draw one of the primitive types in this rectangle.

            // draw a line
            if (primType == 0)
            {
                float2 p0 = rectMin;
                float2 p1 = rectMax;
                drawList->AddLine(p0 + margin, p1 - margin, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // draw a filled rectangle
            if (primType == 1)
            {
                drawList->AddRect(rectMin + margin, rectMax - margin, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Draw a filled multicolor rectangle
            if (primType == 2)
            {
                drawList->AddRectFilledMultiColor(
                    rectMin + margin,
                    rectMax - margin,
                    ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)),
                    ImColor(float4(0.0f, 0.0f, 1.0f, 1.0f)),
                    ImColor(float4(0.0f, 1.0f, 0.0f, 1.0f)),
                    ImColor(float4(1.0f, 0.0f, 0.0f, 1.0f))
                );
            }

            // Draw a diamond using the quad function
            if (primType == 3)
            {
                float2 center = (rectMin + rectMax) / 2.0f;
                float2 size = ((rectMax - rectMin) / 2.0f) - 2.f * float2(margin, margin);
                float2 p1 = center + float2(0.0f, size.y);
                float2 p2 = center + float2(size.x, 0.0f);
                float2 p3 = center + float2(0.0f, -size.y);
                float2 p4 = center + float2(-size.x, 0.0f);
                drawList->AddQuad(p1, p2, p3, p4, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Draw a filled diamond using the quad function
            if (primType == 4)
            {
                float2 center = (rectMin + rectMax) / 2.0f;
                float2 size = ((rectMax - rectMin) / 2.0f) - 2.f * float2(margin, margin);
                float2 p1 = center + float2(0.0f, size.y);
                float2 p2 = center + float2(size.x, 0.0f);
                float2 p3 = center + float2(0.0f, -size.y);
                float2 p4 = center + float2(-size.x, 0.0f);
                drawList->AddQuadFilled(p1, p2, p3, p4, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Draw a triangle within the rectangle
            if (primType == 5)
            {
                // the three tri points between 0 and 1
                float2 p1 = (rectMin + margin) + float2(0.5f, 0.0f) * ((rectMax - margin) - (rectMin + margin));
                float2 p2 = (rectMin + margin) + float2(1.0f, 1.0f) * ((rectMax - margin) - (rectMin + margin));
                float2 p3 = (rectMin + margin) + float2(0.0f, 1.0f) * ((rectMax - margin) - (rectMin + margin));
                drawList->AddTriangle(p1, p2, p3, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Draw a filled triangle within the rectangle
            if (primType == 6)
            {
                float2 p1 = (rectMin + margin) + float2(0.5f, 0.0f) * ((rectMax - margin) - (rectMin + margin));
                float2 p2 = (rectMin + margin) + float2(1.0f, 1.0f) * ((rectMax - margin) - (rectMin + margin));
                float2 p3 = (rectMin + margin) + float2(0.0f, 1.0f) * ((rectMax - margin) - (rectMin + margin));
                drawList->AddTriangleFilled(p1, p2, p3, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Draw a circle within the rectangle
            if (primType == 7)
            {
                float2 center = ((rectMin + margin) + (rectMax - margin)) / 2.0f;
                float2 size = (rectMax - margin) - (rectMin + margin);
                float radius = std::min(size.x, size.y) / 2.0f;
                drawList->AddCircle(center, radius, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Draw a filled circle within the rectangle
            if (primType == 8)
            {
                float2 center = ((rectMin + margin) + (rectMax - margin)) / 2.0f;
                float2 size = (rectMax - margin) - (rectMin + margin);
                float radius = std::min(size.x, size.y) / 2.0f;
                drawList->AddCircleFilled(center, radius, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Draw an ngon
            if (primType == 9)
            {
                float2 center = ((rectMin + margin) + (rectMax - margin)) / 2.0f;
                float2 size = (rectMax - margin) - (rectMin + margin);
                float radius = std::min(size.x, size.y) / 2.0f;
                drawList->AddNgon(center, radius, 5, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Draw a filled ngon
            if (primType == 10)
            {
                float2 center = ((rectMin + margin) + (rectMax - margin)) / 2.0f;
                float2 size = (rectMax - margin) - (rectMin + margin);
                float radius = std::min(size.x, size.y) / 2.0f;
                drawList->AddNgonFilled(center, radius, 5, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Draw a text string
            if (primType == 11)
            {
                std::string text = "Hello, world!";
                drawList->AddText(rectMin + margin, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)), text.c_str());
            }

            // Draw an example polyline inside the area of the rectangle in the form of a star
            if (primType == 12)
            {
                std::vector<ImVec2> points;
                float2 center = ((rectMin + margin) + (rectMax - margin)) / 2.0f;
                float2 size = (rectMax - margin) - (rectMin + margin);
                float radius = std::min(size.x, size.y) / 2.0f;
                for (int k = 0; k < 12; k++)
                {
                    float angle = 2.0f * 3.14159f * float(k) / 12.0f;
                    float2 point = center + radius * (((k % 2) == 0) ? .5f : 1.f) * float2(std::cos(angle), std::sin(angle));
                    points.push_back(point);
                }
                drawList->AddPolyline(points.data(), points.size(), ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)), ImDrawFlags_None, 1.0f);
            }

            // Same as above, but makes a filled convex polygon
            if (primType == 13)
            {
                std::vector<ImVec2> points;
                float2 center = ((rectMin + margin) + (rectMax - margin)) / 2.0f;
                float2 size = (rectMax - margin) - (rectMin + margin);
                float radius = std::min(size.x, size.y) / 2.0f;
                for (int k = 0; k < 12; k++)
                {
                    float angle = 2.0f * 3.14159f * float(k) / 12.0f;
                    float2 point = center + radius * (((k % 2) == 0) ? .5f : 1.f) * float2(std::cos(angle), std::sin(angle));
                    points.push_back(point);
                }
                drawList->AddConvexPolyFilled(points.data(), points.size(), ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)));
            }

            // Now draw a bezier cubic curve in the next rectangle
            if (primType == 14)
            {
                float2 p0 = rectMin + margin;
                float2 p1 = rectMin + float2(0.25f, 0.75f) * (rectMax - rectMin);
                float2 p2 = rectMin + float2(0.75f, 0.25f) * (rectMax - rectMin);
                float2 p3 = rectMax - margin;
                drawList->AddBezierCubic(p0, p1, p2, p3, ImColor(float4(1.0f, 1.0f, 1.0f, 1.0f)), 1.0f);
            }

            primType++;
        }
    }
}

void OverlaySamplePass::registerBindings(pybind11::module& m) {}
