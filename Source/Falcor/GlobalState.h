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

#include "Core/Macros.h"
#include "Core/AssetResolver.h"
#include "Core/API/Device.h"
#include "Scene/SceneBuilder.h"

namespace Falcor
{

/// This file is a temporary workaround to give access to global state in deprecated python bindings.
/// Some of the Python API was originally designed to allow creation of objects "out of thin air".
/// Two places are affected:
/// - Loading `.pyscene` files: Here many of the scene objects can just be created without a factory.
/// - Creating/loading render graphs and passes
/// The C++ side is currently being refactored to get rid of all that global state (for example, the GPU device).
/// In order to not break the existing Python API, we use global state in very specific contexts only.
/// All of the affected python bindings are marked with PYTHONDEPRECATED. Once these bindings are removed,
/// this file can also be removed as well.

FALCOR_API void setActivePythonSceneBuilder(SceneBuilder* pSceneBuilder);
FALCOR_API SceneBuilder& accessActivePythonSceneBuilder();
FALCOR_API AssetResolver& getActiveAssetResolver();

FALCOR_API void setActivePythonRenderGraphDevice(ref<Device> pDevice);
FALCOR_API ref<Device> getActivePythonRenderGraphDevice();
FALCOR_API ref<Device> accessActivePythonRenderGraphDevice();

} // namespace Falcor
