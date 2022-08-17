/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Rendering/RTXGI/RTXGIVolume.h"
#include "VisualizerDefines.slang"

using namespace Falcor;

/** Diffuse indirect illumination using RTXGI.

    The default output is a fullscreen buffer containing the indirect illumination.
    The RTXGI probe volume can be configured from the UI. Most changes require the
    probe volume to be re-created, which means the illumination is reset.
    It converges relatively quickly to a stable solution for a static scene.

    The render pass also includes a probe visualization mode enabled from the UI.
    In visualization mode, the individual probes are rasterized as spheres.
    The background shows the direct illumination solution that feeds the probe updates.
*/
class RTXGIPass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<RTXGIPass>;

    static const Info kInfo;

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    RTXGIPass(const Dictionary& dict);

    void init();
    void parseDictionary(const Dictionary& dict);
    void renderVisualizerUI(Gui::Widgets& widget);
    void probeVisualizerPass(RenderContext* pRenderContext, const RenderData& renderData);
    void computeIndirectPass(RenderContext* pRenderContext, const RenderData& renderData);
    void bindPassIO(ShaderVar var, const RenderData& renderData) const;

    // Internal state
    Scene::SharedPtr mpScene;
    RTXGIVolume::SharedPtr mpVolume;
    RTXGIVolume::Options mVolumeOptions;

    uint2 mFrameDim = { 0, 0 };
    uint32_t mFrameCount = 0;                       ///< Frames rendered. This is used as random seed.

    bool mEnablePass = true;                        ///< On/off switch for the pass.
    bool mUseVBuffer = false;                       ///< Use a V-buffer as input. Use compile-time constant USE_VBUFFER in shader.
    bool mGBufferAdjustShadingNormals = false;      ///< True if GBuffer/VBuffer has adjusted shading normals enabled.

    ChannelList mInputChannels;                     ///< Render pass inputs.

    // Indirect pass
    ComputePass::SharedPtr mpComputeIndirectPass;

    // Probe visualizer
    Scene::SharedPtr mpSphereScene;                 ///< Mesh for probe visualization (single sphere).
    Texture::SharedPtr mpDepthStencil;
    ComputePass::SharedPtr mpVisualizeBackground;   ///< Program drawing the scene/background in visualization mode.

    struct
    {
        RtProgram::SharedPtr pProgram;              ///< Ray tracing program for visualizing the scene with direct illumination.
        RtBindingTable::SharedPtr pBindingTable;
        RtProgramVars::SharedPtr pVars;
    } mVisualizeDirect;

    struct
    {
        GraphicsState::SharedPtr pState;
        GraphicsProgram::SharedPtr pProgram;        ///< Raster program drawing the probes in visualization mode.
        GraphicsVars::SharedPtr pVars;
    } mVisualizeProbes;

    struct
    {
        bool enableVisualizer = true;               ///< Main on/off switch for the visualization pass.
        bool showProbeStates = false;               ///< Enables visualization of the probe states as colored outline on the probes.
        bool highlightProbe = false;                ///< Enables highlighting of a specific probe.
        uint32_t probeIndex = 0;                    ///< Index of the currently selected probe. This is only used for highlighting a particular probe at the moment.
        float probeRadius = 1.f;                    ///< Probe radius. This is set based on scene bbox and/or the UI.
        VisualizerSceneMode sceneMode = VisualizerSceneMode::IndirectDiffuse;
        VisualizerProbeMode probeMode = VisualizerProbeMode::Irradiance;
    } mVisualizerOptions;
};
