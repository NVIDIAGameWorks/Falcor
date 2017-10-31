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
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "FalcorConfig.h"
#include "Framework.h"
#include "Sample.h"
#define _USE_MATH_DEFINES
#include <math.h>

// Core
#include "API/BlendState.h"
#include "API/Buffer.h"
#include "API/DepthStencilState.h"
#include "API/Device.h"
#include "API/FBO.h"
#include "API/Formats.h"
#include "API/GpuTimer.h"
#include "API/GraphicsStateObject.h"
#include "API/ProgramReflection.h"
#include "API/ProgramVars.h"
#include "API/ProgramVersion.h"
#include "API/RasterizerState.h"
#include "API/RenderContext.h"
#include "API/Sampler.h"
#include "API/Shader.h"
#include "API/StructuredBuffer.h"
#include "API/Texture.h"
#include "API/ConstantBuffer.h"
#include "API/VAO.h"
#include "API/VertexLayout.h"
#include "API/Window.h"
#include "API/TypedBuffer.h"
#include "API/CopyContext.h"
#include "API/ComputeContext.h"
#include "API/QueryHeap.h"

#if defined FALCOR_D3D12 || defined FALCOR_VK
#include "API/DescriptorSet.h"
#include "API/LowLevel/DescriptorPool.h"
#include "API/LowLevel/FencedPool.h"
#include "API/LowLevel/GpuFence.h"
#include "API/LowLevel/RootSignature.h"
#endif //FALCOR_D3D12 || defined FALCOR_VK

// Graphics
#include "Graphics/Camera/Camera.h"
#include "Graphics/Camera/CameraController.h"
#include "Graphics/GraphicsState.h"
#include "Graphics/FullScreenPass.h"
#include "Graphics/TextureHelper.h"
#include "Graphics/Light.h"
#include "Graphics/Program.h"
#include "Graphics/GraphicsProgram.h"
#include "Graphics/FboHelper.h"
#include "Graphics/ComputeProgram.h"
#include "Graphics/ComputeState.h"

// Material
#include "Graphics/Material/Material.h"
#include "Graphics/Material/BasicMaterial.h"
#include "Graphics/Material/MaterialSystem.h"
#include "Graphics/Material/MaterialEditor.h"

// Model
#include "Graphics/Model/Mesh.h"
#include "Graphics/Model/Model.h"
#include "Graphics/Model/ModelRenderer.h"

// Scene
#include "Graphics/Scene/Scene.h"
#include "Graphics/Scene/SceneRenderer.h"
#include "Graphics/Scene/Editor/SceneEditor.h"
#include "Graphics/Scene/SceneUtils.h"


// Math
#include "Utils/Math/FalcorMath.h"
#include "Utils/Math/CubicSpline.h"
#include "Utils/Math/ParallelReduction.h"

// Utils
#include "Utils/Bitmap.h"
#include "Utils/DDSHeader.h"
#include "Utils/Font.h"
#include "Utils/Gui.h"
#include "Utils/Logger.h"
#include "Utils/OS.h"
#include "Utils/TextRenderer.h"
#include "Utils/CpuTimer.h"
#include "Utils/UserInput.h"
#include "Utils/Profiler.h"
#include "Utils/StringUtils.h"
#include "Utils/BinaryFileStream.h"
#include "Utils/Video/VideoEncoder.h"
#include "Utils/Video/VideoEncoderUI.h"
#include "Utils/Video/VideoDecoder.h"
#include "Utils/ProgressBar.h"
#include "Utils/ThreadPool.h"

// VR
#include "VR/OpenVR/VRSystem.h"
#include "VR/VrFbo.h"

// Effects
#include "Effects/NormalMap/LeanMap.h"
#include "Effects/Shadows/CSM.h"
#include "Effects/Utils/GaussianBlur.h"
#include "Effects/SkyBox/SkyBox.h"
#include "Effects/ToneMapping/ToneMapping.h"
#include "Effects/AmbientOcclusion/SSAO.h"
#include "Effects/ParticleSystem/ParticleSystem.h"
#include "Effects/TAA/TAA.h"

#define FALCOR_MAJOR_VERSION 2
#define FALCOR_MINOR_VERSION 0
#define FALCOR_DEV_STAGE "beta"
#define FALCOR_DEV_REVISION 8
#define FALCOR_VERSION_STRING "2.0b8"