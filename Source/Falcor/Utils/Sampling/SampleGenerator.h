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
#include "SampleGeneratorType.slangh"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/Program/DefineList.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/UI/Gui.h"
#include <functional>
#include <memory>

namespace Falcor
{
class RenderContext;

/**
 * Utility class for sample generators on the GPU.
 *
 * This class has functions for configuring the shader program and
 * uploading the necessary lookup tables (if needed).
 * On the GPU, import SampleGenerator.slang in your shader program.
 */
class FALCOR_API SampleGenerator : public Object
{
    FALCOR_OBJECT(SampleGenerator)
public:
    virtual ~SampleGenerator() = default;

    /**
     * Factory function for creating a sample generator of the specified type.
     * @param[in] pDevice GPU device.
     * @param[in] type The type of sample generator. See SampleGeneratorType.slangh.
     * @return New object, or throws an exception on error.
     */
    static ref<SampleGenerator> create(ref<Device> pDevice, uint32_t type);

    /**
     * Get macro definitions for this sample generator.
     * @return Macro definitions that must be set on the shader program that uses this sampler.
     */
    virtual DefineList getDefines() const;

    /**
     * Binds the data to a program vars object.
     * @param[in] pVars ProgramVars of the program to set data into.
     */
    virtual void bindShaderData(const ShaderVar& var) const {}

    /**
     * Render the sampler's UI.
     */
    virtual void renderUI(Gui::Widgets& widget) {}

    /**
     * Begin a frame.
     * This should be called at the beginning of each frame for samplers that do extra setup for each frame.
     * @param[in] pRenderContext Render context.
     * @param[in] frameDim Current frame dimension.
     * @return Returns true if internal state has changed and bindShaderData() should be called before using the sampler.
     */
    virtual bool beginFrame(RenderContext* pRenderContext, const uint2& frameDim) { return false; }

    /**
     * End a frame.
     * This should be called at the end of each frame for samplers that do extra setup for each frame.
     * @param[in] pRenderContext Render context.
     * @param[in] pRenderOutput Rendered output.
     */
    virtual void endFrame(RenderContext* pRenderContext, const ref<Texture>& pRenderOutput) {}

    /**
     * Returns a GUI dropdown list of all available sample generators.
     */
    static const Gui::DropdownList& getGuiDropdownList();

    /**
     * Register a sample generator type.
     * @param[in] type The type of sample generator. See SampleGeneratorType.slangh.
     * @param[in] name Descriptive name used in the UI.
     * @param[in] createFunc Function to create an instance of the sample generator.
     */
    static void registerType(uint32_t type, const std::string& name, std::function<ref<SampleGenerator>(ref<Device>)> createFunc);

protected:
    SampleGenerator(ref<Device> pDevice, uint32_t type) : mpDevice(pDevice), mType(type) {}

    ref<Device> mpDevice;
    const uint32_t mType; ///< Type of sample generator. See SampleGeneratorType.slangh.

private:
    /**
     * Register all basic sample generator types.
     */
    static void registerAll();

    friend struct RegisterSampleGenerators;
};
} // namespace Falcor
