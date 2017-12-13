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
#include <string>
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "Scene.h"
#include "rapidjson/document.h"
#include <sstream>

namespace Falcor
{

    class SceneExporter
    {
    public:
        friend class Scene;

        enum : uint32_t
        {
            ExportGlobalSettings = 0x1,
            ExportModels         = 0x2,
            ExportLights         = 0x4,
            ExportCameras        = 0x8,
            ExportPaths          = 0x10,
            ExportUserDefined    = 0x20,
            ExportMaterials      = 0x40,
            ExportAll = 0xFFFFFFFF
        };

        static bool saveScene(const std::string& filename, const Scene::SharedPtr& pScene, uint32_t exportOptions = ExportAll);

        static const uint32_t kVersion = 2;

    private:

        SceneExporter(const std::string& filename, const Scene::SharedPtr& pScene)
            : mpScene(pScene), mFilename(filename) {}

        bool save(uint32_t exportOptions);

        void writeModels();
        void writeLights();
        void writeCameras();
        void writeGlobalSettings(bool writeActivePath);
        void writePaths();
        void writeUserDefinedSection();
        void writeMaterials();

        rapidjson::Document mJDoc;
        Scene::SharedPtr mpScene = nullptr;
        std::string mFilename;
        uint32_t mExportOptions = 0;
    };
}