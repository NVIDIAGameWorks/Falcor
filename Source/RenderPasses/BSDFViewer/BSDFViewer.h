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
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Utils/Debug/PixelDebug.h"
#include "Scene/Lights/EnvMap.h"
#include "BSDFViewerParams.slang"

using namespace Falcor;

class BSDFViewer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(BSDFViewer, "BSDFViewer", "BSDF inspection utility.");

    static ref<BSDFViewer> create(ref<Device> pDevice, const Properties& props) { return make_ref<BSDFViewer>(pDevice, props); }

    BSDFViewer(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override;

private:
    void parseProperties(const Properties& props);
    bool loadEnvMap(const std::filesystem::path& path);
    void readPixelData();

    // Internal state

    /// Loaded scene if any, nullptr otherwise.
    ref<Scene> mpScene;
    /// Environment map if loaded, nullptr otherwise.
    ref<EnvMap> mpEnvMap;
    /// Use environment map if available.
    bool mUseEnvMap = true;

    /// Parameters shared with the shaders.
    BSDFViewerParams mParams;
    /// Random number generator for the integrator.
    ref<SampleGenerator> mpSampleGenerator;
    bool mOptionsChanged = false;

    /// GPU fence for synchronizing readback.
    ref<Fence> mpFence;
    /// Buffer for data for the selected pixel.
    ref<Buffer> mpPixelDataBuffer;
    /// Staging buffer for readback of pixel data.
    ref<Buffer> mpPixelStagingBuffer;
    /// Pixel data for the selected pixel (if valid).
    PixelData mPixelData;
    bool mPixelDataValid = false;
    bool mPixelDataAvailable = false;

    /// Utility class for pixel debugging (print in shaders).
    std::unique_ptr<PixelDebug> mpPixelDebug;

    ref<ComputePass> mpViewerPass;

    // UI variables
    Gui::DropdownList mMaterialList;
};
