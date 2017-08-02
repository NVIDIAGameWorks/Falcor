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
#include "TestBase.h"


class GraphicsStateTest : public TestBase
{
private:

    //  Add the Tests.
    void addTests() override;
    
    //  
    void onInit() override {};

    //  Test the Blend State Changes.
    register_testing_func(TestBlendStateSimple);                    //   
    register_testing_func(TestBlendStateNullptr);                   //  
    register_testing_func(TestBlendStateChanges);                   //
    register_testing_func(TestBlendStateMultipleRTChanges);         //

    //  Test the Depth Stencil Changes.
    register_testing_func(TestDepthSimple);                         //  
    register_testing_func(TestDepthNullptr);                        //  
    register_testing_func(TestDepthChanges);                        //  

    //  Test the Stencil State Change
    register_testing_func(TestStencilSimple);                       //
    register_testing_func(TestStencilNullptr);                      //
    register_testing_func(TestStencilChanges);                      //

    //  Test the VAO Changes.
    register_testing_func(TestVAOSimple);                           //
    register_testing_func(TestVAONullptr);                          //
    register_testing_func(TestVAOChanges);                          //

    //  Test the Fbo State Changes.
    register_testing_func(TestFboSimple);                           //
    register_testing_func(TestFboNullptr);                          //  
    register_testing_func(TestFboChanges);                          //


    //  Test the Rasterizer State Changes.
    register_testing_func(TestRasterizerBasicChanges);              //
    register_testing_func(TestRasterizerDepthBiasChanges);          //  
    register_testing_func(TestRasterizerFillModeChanges);           //
    register_testing_func(TestRasterizerSampleCountChanges);        //
    register_testing_func(TestRasterizerConservativeRaster);        // 
    register_testing_func(TestRasterizerScissorChanges);            //
    register_testing_func(TestRasterizerLineAntiAliasing);          //  


    // Test the Graphics State Changes.
    register_testing_func(TestGraphicsProgramBasic);             //
    register_testing_func(TestGraphicsProgramChanges)            //


    //  Set the Rasterizer State to Null and Render.
    static bool renderDefaultRasterizerState(RenderContext::SharedPtr pCtx, GraphicsState::SharedPtr pGS, GraphicsVars::SharedPtr pGV, Vao::SharedPtr pVAO);




    //  
    static const uint32_t kRTCount = 4;


};
