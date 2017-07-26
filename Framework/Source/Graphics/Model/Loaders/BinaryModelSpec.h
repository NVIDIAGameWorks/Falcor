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

//------------------------------------------------------------------------
/*

Binary scene file format v8
---------------------------

- The basic units of data are 32-bit little-endian ints and floats.
- In addition to the latest version, the below specification also describes previous versions of the file format.
- Each individual field is marked with the version number where it was introduced.
- Legacy structs are postfixed with the highest version number for which they are still valid.
- Each line describes: <ofs_dwords> <size_dwords> <Type> <version> <name> (<comments>)

File
0       2       string8 v6  formatID            ("BinScene")
2       1       int     v6  formatVersion       (6)
3       1       int     v6  numTextures
4       1       int     v6  numMeshes
5       1       int     v6  numInstances
6       n*?     array   v6  Texture             (numTextures)
?       n*?     array   v6  Mesh                (numMeshes)
?       n*?     array   v6  Instance            (numInstances)
?

File_v5
0       2       string8 v1  formatID            ("BinMesh ")
2       1       int     v1  formatVersion       (1 .. 5)
3       1       int     v1  numAttribs
4       1       int     v1  numVertices
5       1       int     v2  numTextures
6       1       int     v1  numSubmeshes
7       n*3     array   v1  AttribSpec          (numAttribs)
?       n*?     array   v1  Vertex              (numVertices)
?       n*?     array   v2  Texture             (numTextures)
?       n*?     array   v1  Submesh             (numSubmeshes)
?

Texture
0       1       int     v2  idLength
1       ?       string  v2  idString
?       ?       struct  v2  BinaryImage         (see ImageBinaryIO.hpp)
?

Mesh
0       1       int     v6  numAttribs
1       1       int     v6  numVertices
2       1       int     v6  numSubmeshes
3       n*3     array   v6  AttribSpec          (numAttribs)
?       n*?     array   v6  Vertex              (numVertices)
?       n*?     array   v6  Submesh             (numSubmeshes)
?

AttribSpec
0       1       int     v1  Type                (see MeshBase::AttribType)
1       1       int     v1  format              (see MeshBase::AttribFormat)
2       1       int     v1  length
3

Vertex
0       ?       bytes   v1  vertex data         (dictated by the AttribSpecs)
?

Submesh
0       3       float   v1  ambient             (ignored)
3       4       float   v1  diffuse
7       3       float   v1  specular
10      1       float   v1  glossiness
11      1       float   v3  displacementCoef
12      1       float   v3  displacementBias
13      1       int     v2  diffuseTexture      (-1 if none)
14      1       int     v2  alphaTexture        (-1 if none)
15      1       int     v3  displacementTexture (-1 if none)
16      1       int     v4  normalTexture       (-1 if none)
17      1       int     v4  environmentTexture  (-1 if none)
18      1       int     v5  specularTexture     (-1 if none)
19      1       int     v1  numTriangles
20      n*3     int     v1  indices             (numTriangles * 3)
?

Instance
0       1       int     v6  meshIdx             (-1 if none)
1       1       bool    v6  enabled
2       16      float   v6  meshToWorld         (column-major 4x4 matrix)
18      1       int     v6  nameLength
19      ?       string  v6  nameString
?       1       int     v6  metadataLength
?       ?       string  v6  metadataString
?

*/
//------------------------------------------------------------------------

enum AttribType // allows arbitrary values
{
    AttribType_Position = 0,    // (x, y, z) or (x, y, z, w)
    AttribType_Normal,          // (x, y, z)
    AttribType_Color,           // (r, g, b) or (r, g, b, a)
    AttribType_TexCoord,        // (u, v) or (u, v, w)

    AttribType_AORadius,        // (min, max)


    AttribType_Tangent,        // (x, y, z)
    AttribType_Bitangent,      // (x, y, z)

    AttribType_Max
};

enum AttribFormat
{
    AttribFormat_U8 = 0,
    AttribFormat_S32,
    AttribFormat_F32,

    AttribFormat_Max
};

enum TextureType
{
    TextureType_Diffuse = 0,    // Diffuse color map.
    TextureType_Alpha,          // Alpha map (green = opacity).
    TextureType_Displacement,   // Displacement map (green = height).
    TextureType_Normal,         // Tangent-space normal map.
    TextureType_Environment,    // Environment map (spherical coordinates).
    TextureType_Specular,       // Specular color map.
    TextureType_Glossiness,     // Glossiness map.
    TextureType_Max
};
