/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************/
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

static const float kInvalidCosConeAngle = -1.0f;

// IMPORTANT: these constants must be kept in sync with the definitions of
// InternalNode and LeafNode in Experimental/Scene/Lights/LightBVH.h.

// Common internal and leaf nodes' members offset
static const uint32_t kNodeTypeOffset            =  0u;
static const uint32_t kNodeAABBMinOffset         =  4u;
static const uint32_t kNodeFluxOffset            = 16u;
static const uint32_t kNodeAABBMaxOffset         = 20u;
static const uint32_t kNodeCosConeAngleOffset    = 32u;
static const uint32_t kNodeConeDirectionOffset   = 36u;

// Internal nodes' specific members offset
static const uint32_t kNodeLeftByteOffsetOffset  = 48u;
static const uint32_t kNodeRightByteOffsetOffset = 52u;

// Leaf nodes' specific members offset
static const uint32_t kNodeTriangleCountOffset   = 48u;
static const uint32_t kNodeTriangleIndicesOffset = 52u;
static const uint32_t kNodeTriangleIndexByteSize =  4u;

END_NAMESPACE_FALCOR
