/***************************************************************************
* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
***************************************************************************/
#pragma once
#include "Falcor.h"

namespace Falcor
{
    class Gui;

    /** Temporal AA class
    */
    class TemporalAA
    {
    public:
        using UniquePtr = std::unique_ptr<TemporalAA>;
        /** Destructor
        */
        ~TemporalAA();

        /** Create a new instance
        */
        static UniquePtr create();

        /** Set UI elements
        */
        void renderUI(Gui* pGui);

        /** Run the effect
        \param pRenderContext The Render Context - with the Destination Fbo already applied.
        \param pCurColor The Current Color.
        \param pPrevColor The Previous Color.
        \param pMotionVec The Motion Vector.
        */
        void execute(RenderContext* pRenderContext, const Texture::SharedPtr & pCurColor, const Texture::SharedPtr & pPrevColor, const Texture::SharedPtr & pMotionVec);

        /** Sets the alpha value used to blend the previous frame with the current frame. Lower values means previous frame has more weight
        \param alpha The New Alpha.
        */
        void setAlphaValue(float alpha) { mControls.alpha = alpha; }

        /** Sets the sigma value
        \param sigma The Sigma.
        */
        void setColorBoxSigma(float sigma) { mControls.colorBoxSigma = sigma; }


    private:
        TemporalAA();

        //  Create the Program.
        void createProgram();

        //  Set the Variable Data needed for Rendering.
        void setVarsData(const Texture::SharedPtr & pCurColor, const Texture::SharedPtr & pPrevColor, const Texture::SharedPtr & pMotionVec);

        FullScreenPass::UniquePtr mpProgram;
        GraphicsVars::SharedPtr mpProgVars;
        ConstantBuffer::SharedPtr mpCB;
        Sampler::SharedPtr mpLinearSampler;

        struct
        {
            uint32_t colorTex;
            uint32_t prevColorTex;
            uint32_t motionVecTex;
            uint32_t sampler;
            size_t alpha;
            size_t colorBoxSigma;
        } mVarOffsets;

        struct Controls
        {
            float alpha = 0.1f;
            float colorBoxSigma = 1.0f;
        };

        //  
        Controls mControls;
    };
}