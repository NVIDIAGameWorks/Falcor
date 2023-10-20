/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Visualization2D.h"

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
const char kMarkerShaderFile[] = "Samples/Visualization2D/Visualization2d.ps.slang";
const char kNormalsShaderFile[] = "Samples/Visualization2D/VoxelNormals.ps.slang";

const Gui::DropdownList kModeList = {
    {(uint32_t)Visualization2D::Scene::MarkerDemo, "Marker demo"},
    {(uint32_t)Visualization2D::Scene::VoxelNormals, "Voxel normals"},
};
} // namespace

Visualization2D::Visualization2D(const SampleAppConfig& config) : SampleApp(config) {}

Visualization2D::~Visualization2D() {}

void Visualization2D::onLoad(RenderContext* pRenderContext)
{
    createRenderPass();
}

void Visualization2D::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    auto var = mpMainPass->getRootVar();
    var["Visual2DCB"]["iResolution"] = float2(width, height);
    var["Visual2DCB"]["iGlobalTime"] = (float)getGlobalClock().getTime();
    var["Visual2DCB"]["iMousePosition"] = mMousePosition;

    switch (mSelectedScene)
    {
    case Scene::MarkerDemo:
        break;
    case Scene::VoxelNormals:
        var["VoxelNormalsCB"]["iShowNormalField"] = mVoxelNormalsGUI.showNormalField;
        var["VoxelNormalsCB"]["iShowBoxes"] = mVoxelNormalsGUI.showBoxes;
        var["VoxelNormalsCB"]["iShowBoxDiagonals"] = mVoxelNormalsGUI.showBoxDiagonals;
        var["VoxelNormalsCB"]["iShowBorderLines"] = mVoxelNormalsGUI.showBorderLines;
        var["VoxelNormalsCB"]["iShowBoxAroundPoint"] = mVoxelNormalsGUI.showBoxAroundPoint;
        break;
    default:
        break;
    }

    // Run main pass.
    mpMainPass->execute(pRenderContext, pTargetFbo);
}

void Visualization2D::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Visualization 2D", {700, 900}, {10, 10});
    bool changed = w.dropdown("Scene selection", kModeList, reinterpret_cast<uint32_t&>(mSelectedScene));
    if (changed)
    {
        createRenderPass();
    }
    bool paused = getGlobalClock().isPaused();
    changed = w.checkbox("Pause time", paused);
    if (changed)
    {
        if (paused)
        {
            getGlobalClock().pause();
        }
        else
        {
            getGlobalClock().play();
        }
    }

    renderGlobalUI(pGui);
    if (mSelectedScene == Scene::MarkerDemo)
    {
        w.text("Left-click and move mouse...");
    }
    else if (mSelectedScene == Scene::VoxelNormals)
    {
        w.text("Left-click and move mouse in the left boxes to display the normal there.");
        w.checkbox("Show normal field", mVoxelNormalsGUI.showNormalField, false);
        w.checkbox("Show boxes", mVoxelNormalsGUI.showBoxes, false);
        w.checkbox("Show box diagonals", mVoxelNormalsGUI.showBoxDiagonals, false);
        w.checkbox("Show border lines", mVoxelNormalsGUI.showBorderLines, false);
        w.checkbox("Show box around point", mVoxelNormalsGUI.showBoxAroundPoint, false);
    }
}

bool Visualization2D::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return false;
}

bool Visualization2D::onMouseEvent(const MouseEvent& mouseEvent)
{
    bool bHandled = false;
    switch (mouseEvent.type)
    {
    case MouseEvent::Type::ButtonDown:
    case MouseEvent::Type::ButtonUp:
        if (mouseEvent.button == Input::MouseButton::Left)
        {
            mLeftButtonDown = mouseEvent.type == MouseEvent::Type::ButtonDown;
            bHandled = true;
        }
        break;
    case MouseEvent::Type::Move:
        if (mLeftButtonDown)
        {
            mMousePosition = mouseEvent.screenPos;
            bHandled = true;
        }
        break;
    default:
        break;
    }
    return bHandled;
}

void Visualization2D::createRenderPass()
{
    switch (mSelectedScene)
    {
    case Scene::MarkerDemo:
        mpMainPass = FullScreenPass::create(getDevice(), kMarkerShaderFile);
        break;
    case Scene::VoxelNormals:
        mpMainPass = FullScreenPass::create(getDevice(), kNormalsShaderFile);
        break;
    default:
        FALCOR_UNREACHABLE();
        break;
    }
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "Falcor 2D Visualization";
    config.windowDesc.resizableWindow = true;
    config.windowDesc.width = 1400;
    config.windowDesc.height = 1000;
    config.windowDesc.enableVSync = true;

    Visualization2D visualization2D(config);
    return visualization2D.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
