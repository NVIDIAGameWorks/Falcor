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

#define VERTEX_POSITION_LOC         0
#define VERTEX_NORMAL_LOC           1
#define VERTEX_BITANGENT_LOC        2
#define VERTEX_TEXCOORD_LOC         3
#define VERTEX_LIGHTMAP_UV_LOC      4
#define VERTEX_BONE_WEIGHT_LOC      5
#define VERTEX_BONE_ID_LOC          6
#define VERTEX_DIFFUSE_COLOR_LOC    7
#define VERTEX_PREV_POSITION_LOC    8

#define VERTEX_LOCATION_COUNT       9

#define VERTEX_USER_ELEM_COUNT      4
#define VERTEX_USER0_LOC            (VERTEX_LOCATION_COUNT)

#define VERTEX_POSITION_NAME        "POSITION"
#define VERTEX_NORMAL_NAME          "NORMAL"
#define VERTEX_BITANGENT_NAME       "BITANGENT"
#define VERTEX_TEXCOORD_NAME        "TEXCOORD"
#define VERTEX_LIGHTMAP_UV_NAME     "LIGHTMAP_UV"
#define VERTEX_BONE_WEIGHT_NAME     "BONE_WEIGHTS"
#define VERTEX_BONE_ID_NAME         "BONE_IDS"
#define VERTEX_DIFFUSE_COLOR_NAME   "DIFFUSE_COLOR"
#define VERTEX_PREV_POSITION_NAME   "PREV_POSITION"
