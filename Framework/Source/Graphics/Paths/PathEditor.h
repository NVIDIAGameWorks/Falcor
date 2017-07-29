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
#include "Utils/Gui.h"
#include "Graphics/Paths/ObjectPath.h"
#include "Graphics/Camera/Camera.h"
#include "Graphics/Scene/Scene.h"

namespace Falcor
{
    class PathEditor
    {
    public:
        using PathEditorCallback = std::function<void(void)>;
        using UniquePtr = std::unique_ptr<PathEditor>;
        using UniqueConstPtr = std::unique_ptr<const PathEditor>;

        /** Create a path editor instance
            \param[in] pPath Path to edit
            \param[in] pCamera Camera being used in the scene. Allows user to update key frame orientation based on a camera
            \param[in] frameChangedCB Function that is called when a key frame's data has been updated
            \param[in] addRemoveKeyframeCB Function that is called when a key frame has been added or removed
            \param[in] editCompleteCB Function that is called when the path editor is closed
        */
        static UniquePtr create(const ObjectPath::SharedPtr& pPath, const Camera::SharedPtr& pCamera, PathEditorCallback frameChangedCB, PathEditorCallback addRemoveKeyframeCB, PathEditorCallback editCompleteCB);
        ~PathEditor();

        /** Render the editor UI elements.
            \param[in] pGui GUI object to render the editor UI with
        */
        void render(Gui* pGui);

        /** Set the active key frame to edit.
            \param[in] id Key frame id
        */
        void setActiveFrame(uint32_t id);

        /** Get the active key frame
        */
        uint32_t getActiveFrame() const { return mActiveFrame; }

        /** Get the path being edited.
        */
        const ObjectPath::SharedPtr& getPath() const { return mpPath; }

    private:
        PathEditor(const ObjectPath::SharedPtr& pPath, const Camera::SharedPtr& pCamera, PathEditorCallback frameChangedCB, PathEditorCallback addRemoveKeyframeCB, PathEditorCallback editCompleteCB);

        bool closeEditor(Gui* pGui);
        void editPathName(Gui* pGui);
        void editPathLoop(Gui* pGui);
        void editActiveFrameID(Gui* pGui);
        void addFrame(Gui* pGui);
        void deleteFrame(Gui* pGui);
        void editFrameTime(Gui* pGui);
        void editKeyframeProperties(Gui* pGui);

        void updateFrameTime(Gui* pGui);
        void moveToCamera(Gui* pGui);

        ObjectPath::SharedPtr mpPath;
        Camera::SharedPtr mpCamera;

        PathEditorCallback mEditCompleteCB;
        PathEditorCallback mFrameChangedCB;
        PathEditorCallback mAddRemoveKeyframeCB;

        int32_t mActiveFrame = 0;
        float mFrameTime = 0;
    };
}