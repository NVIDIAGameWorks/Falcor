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
#include "Framework.h"
#include "MaterialSystem.h"
#include "Material.h"
#include "Graphics/Program/Program.h"
#include <map>

namespace Falcor
{
    namespace MaterialSystem
    {
        using ProgramVersionMap = std::map<const ProgramVersion*, ProgramVersion::SharedConstPtr>;
        using MaterialProgramMap = std::map<uint64_t, ProgramVersionMap>;

        static MaterialProgramMap gMaterialProgramMap;

        void reset()
        {
            gMaterialProgramMap.clear();
        }

        void removeMaterial(uint64_t descIdentifier)
        {
            gMaterialProgramMap.erase(descIdentifier);
        }

        void removeProgramVersion(const ProgramVersion* pProgramVersion)
        {
            if(gMaterialProgramMap.size())
            {
                for(auto& it : gMaterialProgramMap)
                {
                    it.second.erase(pProgramVersion);
                }
            }
        }

        static ProgramVersion::SharedConstPtr findProgramInMap(ProgramVersionMap& programMap, const ProgramVersion* pVersion)
        {
            auto it = programMap.find(pVersion);
            return (it == programMap.end()) ? nullptr : it->second;
        }

        static ProgramVersionMap& getMaterialProgramMap(const Material* pMaterial)
        {
            uint64_t descId = pMaterial->getDescIdentifier();
            auto it = gMaterialProgramMap.find(descId);
            if(it == gMaterialProgramMap.end())
            {
                gMaterialProgramMap[descId] = ProgramVersionMap();
            }

            return gMaterialProgramMap[descId];
        }

        void patchProgram(Program* pProgram, const Material* pMaterial)
        {
//             // Get the active program version
//             const ProgramVersion* pProgVersion = pProgram->getActiveVersion().get();
// 
//             // Get the material's program map
//             ProgramVersionMap& programMap = getMaterialProgramMap(pMaterial);
// 
//             // Check if it we have data for it
//             ProgramVersion::SharedConstPtr pMaterialProg = findProgramInMap(programMap, pProgVersion);
//            if(pMaterialProg == nullptr)
            {
                // Add the material desc
                pProgram->addDefine("_MS_STATIC_MATERIAL_DESC", pMaterial->getMaterialDescStr());

               
                // Get the program version and set it into the map
//                 pMaterialProg = pProgram->getActiveVersion();
//                 programMap[pProgVersion] = pMaterialProg;

                // Restore the previous define string
//                pProgram->removeDefine("_MS_STATIC_MATERIAL_DESC");
            }

//            return pMaterialProg;
        }
    }
}