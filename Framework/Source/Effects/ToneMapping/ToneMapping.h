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
#include "Graphics/FullScreenPass.h"
#include "API/ConstantBuffer.h"
#include "API/FBO.h"
#include "API/Sampler.h"
#include "Utils/Gui.h"
#include "Graphics/RenderGraph/RenderPass.h"

namespace Falcor
{
    /** Tone-mapping effect
    */
    class ToneMapping : public RenderPass, public inherit_shared_from_this<RenderPass, ToneMapping>
    {
    public:
        using SharedPtr = std::shared_ptr<ToneMapping>;

        /** Destructor
        */
        ~ToneMapping();

        /** The tone-mapping operator to use
        */
        enum class Operator
        {
            Clamp,              ///< Clamp to [0, 1]. Just like LDR
            Linear,             ///< Linear mapping
            Reinhard,           ///< Reinhard operator
            ReinhardModified,   ///< Reinhard operator with maximum white intensity
            HejiHableAlu,       ///< John Hable's ALU approximation of Jim Heji's filmic operator
            HableUc2,           ///< John Hable's filmic tone-mapping used in Uncharted 2
            Aces,               ///< Aces Filmic Tone-Mapping
        };

        /** Create a new object
        */
        static SharedPtr create(Operator op = Operator::Aces);       
        static SharedPtr create(const Dictionary& dict);

        /** Render UI elements
            \param[in] pGui GUI instance to render UI with
            \param[in] uiGroup Name for the group to render UI elements within
        */
        void renderUI(Gui* pGui, const char* uiGroup) override;

        /** Run the tone-mapping program
        \param pRenderContext Render-context to use
        \param pSrc The source FBO. Only color-texture 0 will be tone-mapped
        \param pDst The destination FBO
        */
        deprecate("3.2", "Use the other execute() method, which accepts a single texture as the source")
        void execute(RenderContext* pRenderContext, const Fbo::SharedPtr& pSrc, const Fbo::SharedPtr& pDst);

        /** Run the tone-mapping program
            \param pRenderContext Render-context to use
            \param pSrc The source texture
            \param pDst The destination FBO
        */
        void execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrc, const Fbo::SharedPtr& pDst);

        /** Set a new operator. Triggers shader recompilation if operator has not been set on this instance before.
        */
        void setOperator(Operator op);

        /** Sets the middle-gray luminance used for normalizing each pixel's luminance. 
            Middle gray is usually in the range of [0.045, 0.72].
            Lower values maximize contrast. Useful for night scenes.
            Higher values minimize contrast, resulting in brightly lit objects.
        */
        void setExposureKey(float exposureKey);

        /** Sets the maximal luminance to be consider as pure white.
            Only valid if the operator is ReinhardModified
        */
        void setWhiteMaxLuminance(float maxLuminance);

        /** Sets the luminance texture LOD to use when fetching average luminance values.
            Lower values will result in a more localized effect
        */
        void setLuminanceLod(float lod);

        /** Sets the white-scale used in Uncharted 2 tone mapping.
        */
        void setWhiteScale(float whiteScale);

        /** Called once before compilation. Describes I/O requirements of the pass.
        The requirements can't change after the graph is compiled. If the IO requests are dynamic, you'll need to trigger compilation of the render-graph yourself.
        */
        virtual RenderPassReflection reflect() const override;

        /** Executes the pass.
        */
        virtual void execute(RenderContext* pRenderContext, const RenderData* pData) override;
        
        /** Get the tonemapping operator type.
		*/
        Operator getOperator() const { return mOperator; }
        
        /** Get tonemapper exposure key value.
		*/
        float getExposureKey() const { return mConstBufferData.exposureKey; }
        
        /** Gets the maximal luminance to be consider as pure white. 
		*/
        float getWhiteMaxLuminance() const { return mConstBufferData.whiteMaxLuminance; }
        
        /** Gets the luminance texture LOD to use when fetching average luminance values. 
		*/
        float getLuminanceLod() const { return mConstBufferData.luminanceLod; }
        
        /** Gets the white-scale used in Uncharted 2 tone mapping. 
		*/
        float getWhiteScale() const { return mConstBufferData.whiteScale; }

        /** Get the scripting dictionary
        */
        Dictionary getScriptingDictionary() const override;
    private:
        ToneMapping(Operator op);
        void createLuminanceFbo(const Texture::SharedPtr& pSrc);

        Operator mOperator;
        FullScreenPass::UniquePtr mpToneMapPass;
        FullScreenPass::UniquePtr mpLuminancePass;
        Fbo::SharedPtr mpLuminanceFbo;
        GraphicsVars::SharedPtr mpToneMapVars;
        GraphicsVars::SharedPtr mpLuminanceVars;
        ConstantBuffer::SharedPtr mpToneMapCBuffer;
        Sampler::SharedPtr mpPointSampler;
        Sampler::SharedPtr mpLinearSampler;

        struct PassBindLocations
        {
            ParameterBlockReflection::BindLocation luminanceSampler;
            ParameterBlockReflection::BindLocation colorSampler;
            ParameterBlockReflection::BindLocation colorTex;
            ParameterBlockReflection::BindLocation luminanceTex;
        } mBindLocations;

        struct
        {
            float exposureKey = 0.042f;
            float whiteMaxLuminance = 1.0f;
            float luminanceLod = 16; // Max possible LOD, will result in global operation
            float whiteScale = 11.2f;
        } mConstBufferData;

        void createToneMapPass(Operator op);
        void createLuminancePass();
    };

#define tonemap_op(a) case ToneMapping::Operator::a: return #a
    inline std::string to_string(ToneMapping::Operator op)
    {
        switch (op)
        {
            tonemap_op(Clamp);
            tonemap_op(Linear);
            tonemap_op(Reinhard);
            tonemap_op(ReinhardModified);
            tonemap_op(HejiHableAlu);
            tonemap_op(HableUc2);
            tonemap_op(Aces);
        default:
            should_not_get_here();
            return "";
        }
    }
#undef tonemap_op
}