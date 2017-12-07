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
#include "ObjectPath.h"
#include "PathEditor.h"
#include "Graphics/Camera/Camera.h"
#include "Utils/StringUtils.h"
#include <cstring>

namespace Falcor
{
    bool PathEditor::closeEditor(Gui* pGui)
    {
        if (pGui->addButton("Close Editor"))
        {
            pGui->popWindow();
            if (mEditCompleteCB)
            {
                mEditCompleteCB();
            }
            return true;
        }
        return false;
    }

    void PathEditor::editKeyframeProperties(Gui* pGui)
    {
        if (mpPath->getKeyFrameCount() > 0)
        {
            const auto& keyframe = mpPath->getKeyFrame(mActiveFrame);

            vec3 p = keyframe.position;
            vec3 t = keyframe.target;
            vec3 u = keyframe.up;

            bool changed = false;

            if (pGui->addFloat3Var("Keyframe Position", p, -FLT_MAX, FLT_MAX))
            {
                mpPath->setFramePosition(mActiveFrame, p);
                changed = true;
            }

            if (pGui->addFloat3Var("Keyframe Target", t, -FLT_MAX, FLT_MAX))
            {
                mpPath->setFrameTarget(mActiveFrame, t);
                changed = true;
            }

            if (pGui->addFloat3Var("Keyframe Up", u, -FLT_MAX, FLT_MAX))
            {
                mpPath->setFrameUp(mActiveFrame, u);
                changed = true;
            }

            if (changed)
            {
                mFrameChangedCB();
            }
        }
    }

    void PathEditor::editActiveFrameID(Gui* pGui)
    {
        if(mpPath->getKeyFrameCount() > 0)
        {
            if (pGui->addIntVar("Selected Frame", mActiveFrame, 0, mpPath->getKeyFrameCount() - 1))
            {
                setActiveFrame(mActiveFrame);
            }
        }
    }

    void PathEditor::setActiveFrame(uint32_t id)
    {
        mActiveFrame = id;
        mFrameTime = mpPath->getKeyFrame(mActiveFrame).time;
        mFrameChangedCB();
    }
    
    void PathEditor::editPathLoop(Gui* pGui)
    {
        bool loop = mpPath->isRepeatOn();
        if (pGui->addCheckBox("Loop Path", loop))
        {
            mpPath->setAnimationRepeat(loop);
        }
    }

    void PathEditor::editPathName(Gui* pGui)
    {
        char buffer[1024];
        copyStringToBuffer(buffer, arraysize(buffer), mpPath->getName());

        if (pGui->addTextBox("Path Name", buffer, arraysize(buffer)))
        {
            mpPath->setName(buffer);
        }
    }

    PathEditor::UniquePtr PathEditor::create(const ObjectPath::SharedPtr& pPath, const Camera::SharedPtr& pCamera, PathEditorCallback frameChangedCB, PathEditorCallback addRemoveKeyframeCB, PathEditorCallback editCompleteCB)
    {
        return UniquePtr(new PathEditor(pPath, pCamera, frameChangedCB, addRemoveKeyframeCB, editCompleteCB));
    }

    PathEditor::PathEditor(const ObjectPath::SharedPtr& pPath, const Camera::SharedPtr& pCamera, PathEditorCallback frameChangedCB, PathEditorCallback addRemoveKeyframeCB, PathEditorCallback editCompleteCB)
        : mpPath(pPath)
        , mpCamera(pCamera)
        , mFrameChangedCB(frameChangedCB)
        , mAddRemoveKeyframeCB(addRemoveKeyframeCB)
        , mEditCompleteCB(editCompleteCB)
    {
        if(mpPath->getKeyFrameCount() > 0)
        {
            mFrameTime = mpPath->getKeyFrame(0).time;
        }
    }

    PathEditor::~PathEditor()
    {
    }

    void PathEditor::render(Gui* pGui)
    {
        pGui->pushWindow("Path Editor", 350, 400, 440, 400);
        if (closeEditor(pGui)) return;
        pGui->addSeparator();
        editPathName(pGui);
        editPathLoop(pGui);
        editActiveFrameID(pGui);

        addFrame(pGui);
        deleteFrame(pGui);
        pGui->addSeparator();
        editFrameTime(pGui);
        updateFrameTime(pGui);

        pGui->addSeparator();
        editKeyframeProperties(pGui);
        moveToCamera(pGui);
        pGui->popWindow();
    }

    void PathEditor::editFrameTime(Gui* pGui)
    {
        pGui->addFloatVar("Frame Time", mFrameTime, 0, FLT_MAX);
    }

    void PathEditor::addFrame(Gui* pGui)
    {
        if(pGui->addButton("Add Frame"))
        {
            // If path has keyframes, create new keyframe at the location of the selected keyframe
            if (mpPath->getKeyFrameCount() > 0)
            {
                const auto& currFrame = mpPath->getKeyFrame(mActiveFrame);
                mActiveFrame = mpPath->addKeyFrame(mFrameTime, currFrame.position, currFrame.target, currFrame.up);
            }
            else
            {
                mActiveFrame = mpPath->addKeyFrame(mFrameTime, glm::vec3(), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            }

            mAddRemoveKeyframeCB();

            setActiveFrame(mActiveFrame);
        }
    }

    void PathEditor::deleteFrame(Gui* pGui)
    {
        if (mpPath->getKeyFrameCount() > 0)
        {
            if(pGui->addButton("Remove Frame", true))
            {
                mpPath->removeKeyFrame(mActiveFrame);
                mAddRemoveKeyframeCB();

                mActiveFrame = min(mpPath->getKeyFrameCount() - 1, (uint32_t)mActiveFrame);

                if (mpPath->getKeyFrameCount() > 0)
                {
                   setActiveFrame(mActiveFrame);
                }
            }
        }
    }

    void PathEditor::updateFrameTime(Gui* pGui)
    {
        if (mpPath->getKeyFrameCount() && pGui->addButton("Update Current Frame Time"))
        {
            mActiveFrame = mpPath->setFrameTime(mActiveFrame, mFrameTime);
            mAddRemoveKeyframeCB();

            setActiveFrame(mActiveFrame);
        }
    }

    void PathEditor::moveToCamera(Gui* pGui)
    {
        if (mpPath->getKeyFrameCount() > 0)
        {
            if (pGui->addButton("Move Frame to Camera"))
            {
                mpPath->setFramePosition(mActiveFrame, mpCamera->getPosition());
                mpPath->setFrameTarget(mActiveFrame, mpCamera->getTarget());
                mpPath->setFrameUp(mActiveFrame, mpCamera->getUpVector());
                mFrameChangedCB();
            }
        }
    }

}