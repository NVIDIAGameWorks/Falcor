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

// Core
#include "Core/Macros.h"
#include "Core/FalcorConfig.h"
#include "Core/Assert.h"
#include "Core/ErrorHandling.h"
#include "Core/Errors.h"
#include "Core/HotReloadFlags.h"
#include "Core/Sample.h"

// Core/Platform
#include "Core/Platform/OS.h"

// Core/API
#include "Core/API/Common.h"
#include "Core/API/BlendState.h"
#include "Core/API/Buffer.h"
#include "Core/API/ComputeContext.h"
#include "Core/API/ComputeStateObject.h"
#include "Core/API/CopyContext.h"
#include "Core/API/DepthStencilState.h"
#include "Core/API/Device.h"
#include "Core/API/FBO.h"
#include "Core/API/FencedPool.h"
#include "Core/API/Formats.h"
#include "Core/API/GpuFence.h"
#include "Core/API/GpuTimer.h"
#include "Core/API/GraphicsStateObject.h"
#include "Core/API/IndirectCommands.h"
#include "Core/API/LowLevelContextData.h"
#include "Core/API/ParameterBlock.h"
#include "Core/API/QueryHeap.h"
#include "Core/API/RasterizerState.h"
#include "Core/API/Raytracing.h"
#include "Core/API/RenderContext.h"
#include "Core/API/Resource.h"
#include "Core/API/GpuMemoryHeap.h"
#include "Core/API/ResourceViews.h"
#include "Core/API/RtStateObject.h"
#include "Core/API/Sampler.h"
#include "Core/API/Texture.h"
#include "Core/API/VAO.h"
#include "Core/API/VertexLayout.h"

// Core/BufferTypes
#include "Core/BufferTypes/VariablesBufferUI.h"

// Core/Platform
#include "Core/Platform/OS.h"
#include "Core/Platform/ProgressBar.h"

// Core/Program
#include "Core/Program/ComputeProgram.h"
#include "Core/Program/GraphicsProgram.h"
#include "Core/Program/CUDAProgram.h"
#include "Core/Program/Program.h"
#include "Core/Program/ProgramReflection.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Program/ProgramVersion.h"
#include "Core/Program/RtProgram.h"

// Core/State
#include "Core/State/ComputeState.h"
#include "Core/State/GraphicsState.h"

// Scene
#include "Scene/Scene.h"
#include "Scene/Importer.h"
#include "Scene/Camera/Camera.h"
#include "Scene/Camera/CameraController.h"
#include "Scene/Lights/Light.h"
#include "Scene/Material/MaterialSystem.h"
#include "Scene/Material/StandardMaterial.h"
#include "Scene/Material/HairMaterial.h"
#include "Scene/Material/ClothMaterial.h"
#include "Scene/Animation/Animation.h"
#include "Scene/Animation/AnimationController.h"

// @skallweit: This is temporary to allow renderpasses to be compiled unmodified. Needs to be removed.
#include "RenderGraph/RenderPass.h"

// Utils
#include "Utils/StringFormatters.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/Vector.h"
#include "Utils/Math/Float16.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/Scripting.h"

#include <fmt/format.h> // TODO C++20: Replace with <format>
#include <fstd/span.h> // TODO C++20: Replace with <span>

#include <memory>
#include <iostream>
#include <locale>
#include <codecvt>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <variant>

#include <cstdint>
#include <cmath>
