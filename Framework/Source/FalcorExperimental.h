/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

// RenderGraph
#include "Experimental/RenderGraph/RenderGraph.h"
#include "Experimental/RenderGraph/RenderPass.h"
#include "Experimental/RenderGraph/RenderGraphIR.h"
#include "Experimental/RenderGraph/RenderGraphImportExport.h"
#include "Experimental/RenderGraph/RenderGraphUI.h"

// Render Passes
#include "Experimental/RenderPasses/ForwardLightingPass.h"
#include "Experimental/RenderPasses/BlitPass.h"
#include "Experimental/RenderPasses/DepthPass.h"
#include "Experimental/RenderGraph/RenderPassLibrary.h"
#include "Experimental/RenderGraph/RenderGraphImportExport.h"

// Raytracing
#ifdef FALCOR_D3D12
#include "Experimental/Raytracing/RtModel.h"
#include "Experimental/Raytracing/RtScene.h"
#include "Experimental/Raytracing/RtShader.h"
#include "Experimental/Raytracing/RtProgram/RtProgram.h"
#include "Experimental/Raytracing/RtProgram/RtProgramVersion.h"
#include "Experimental/Raytracing/RtProgram/SingleShaderProgram.h"
#include "Experimental/Raytracing/RtProgram/HitProgram.h"
#include "Experimental/Raytracing/RtProgramVars.h"
#include "Experimental/Raytracing/RtState.h"
#include "Experimental/Raytracing/RtStateObject.h"
#include "Experimental/Raytracing/RtSceneRenderer.h"
#endif
