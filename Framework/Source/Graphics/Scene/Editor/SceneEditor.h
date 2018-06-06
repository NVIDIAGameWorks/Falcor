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
#include <vector>
#include <set>
#include "Graphics/Paths/PathEditor.h"
#include "Utils/DebugDrawer.h"
#include "Utils/Picking/Picking.h"
#include "Graphics/Scene/Editor/Gizmo.h"
#include "Graphics/Scene/Editor/SceneEditorRenderer.h"

namespace Falcor
{
    class Scene;
    class Gui;

    /** Used by the Scene Editor utility app to edit scenes. Contains a separate internal Scene instance to manage
        editor-specific objects such as light bulbs, camera models, and gizmos.
    */
    class SceneEditor
    {
    public:
        using UniquePtr = std::unique_ptr<SceneEditor>;
        using UniqueConstPtr = std::unique_ptr<const SceneEditor>;

        /** Create a Scene Editor instance
            \param[in] pScene Scene to edit
            \param[in] modelLoadFlags Flags to use when adding new models to the scene
        */
        static UniquePtr create(const Scene::SharedPtr& pScene, Model::LoadFlags modelLoadFlags = Model::LoadFlags::None);
        ~SceneEditor();

        /** Get the internal camera used for rendering editor-specific objects. This is also typically the
            camera used to render the application in the SceneEditor utility app.
        */
        const Camera::SharedPtr getEditorCamera() const { return mpEditorScene->getActiveCamera(); }

        /** Updates the internal scene containing editor-specific objects
        */
        void update(double currentTime);

        /** Render the scene editor
            \param[in] pContext Render context
        */
        void render(RenderContext* pContext);

        /** Render the editor's UI elements
            \param[in] pGui GUI instance to render the editor UI with
        */
        void renderGui(Gui* pGui);

        /** Handles mouse events including object picking.
            \param[in] pContext Render context. Used for object picking
            \param[in] mouseEvent Mouse event
        */
        bool onMouseEvent(RenderContext* pContext, const MouseEvent& mouseEvent);

        /** Handles keyboard events
        */
        bool onKeyEvent(const KeyboardEvent& keyEvent);

        /** Updates the internal graphics objects used by the editor. Call this whenever Sample::onResizeSwapChain() is called.
        */
        void onResizeSwapChain();

    private:

        SceneEditor(const Scene::SharedPtr& pScene, Model::LoadFlags modelLoadFlags);
        Scene::SharedPtr mpScene;

        bool mSceneDirty = false;

        // Main GUI functions
        void renderModelElements(Gui* pGui);
        void renderCameraElements(Gui* pGui);
        void renderLightElements(Gui* pGui);
        void renderGlobalElements(Gui* pGui);
        void renderPathElements(Gui* pGui);

        // Model functions
        void addModel(Gui* pGui);
        void deleteModel(Gui* pGui);
        void deleteModel();
        void setModelName(Gui* pGui);
        void setShadingModel(Gui* pGui);
        void setModelVisible(Gui* pGui);
        void selectActiveModel(Gui* pGui);

        // Model instance
        void addModelInstance(Gui* pGui);
        void addModelInstanceRange(Gui* pGui);
        void deleteModelInstance(Gui* pGui);
        void setInstanceTranslation(Gui* pGui);
        void setInstanceScaling(Gui* pGui);
        void setInstanceRotation(Gui* pGui);

        // Camera functions
        void setCameraFocalLength(Gui* pGui);
        void setCameraDepthRange(Gui* pGui);
        void setCameraAspectRatio(Gui* pGui);
        void setCameraName(Gui* pGui);
        void setCameraSpeed(Gui* pGui);
        void addCamera(Gui* pGui);
        void deleteCamera(Gui* pGui);
        void setActiveCamera(Gui* pGui);
        void setCameraPosition(Gui* pGui);
        void setCameraTarget(Gui* pGui);
        void setCameraUp(Gui* pGui);

        // Light functions
        void deleteLight(uint32_t id);
        void addPointLight(Gui* pGui);
        void addDirectionalLight(Gui* pGui);

        // Paths
        void addPath(Gui* pGui);
        void selectPath(Gui* pGui);
        void deletePath(Gui* pGui);
        void startPathEditor(Gui* pGui);
        void startPathEditor();
        void setObjectPath(Gui* pGui, const IMovableObject::SharedPtr& pMovable, const std::string& objType);

        // Global functions
        void saveScene();

        void renderModelAnimation(Gui* pGui);

        Model::LoadFlags mModelLoadFlags = Model::LoadFlags::None;
        Scene::LoadFlags mSceneLoadFlags = Scene::LoadFlags::None;

        //
        // Initialization
        //

        // Initializes Editor helper-scenes, Picking, and Rendering
        void initializeEditorRendering();

        // Initializes Editor's representation of the scene being edited
        void initializeEditorObjects();

        // Model Instance Rotation Angle Helpers
        const glm::vec3& getActiveInstanceRotationAngles();
        void setActiveInstanceRotationAngles(const glm::vec3& rotation);
        std::vector<std::vector<glm::vec3>> mInstanceRotationAngles;

        //
        // Editor Objects
        //

        static const float kCameraModelScale;
        static const float kLightModelScale;
        static const float kKeyframeModelScale;

        // Update transform of gizmos and camera models
        void updateEditorObjectTransforms();

        // Helper function to update the transform of a single camera model from it's respective camera in the master scene
        void updateCameraModelTransform(uint32_t cameraID);

        // Rebuilds the lookup map between light model instances and master scene point lights
        void rebuildLightIDMap();

        // Helper to apply gizmo transforms to the object being edited
        void applyGizmoTransform();

        // Types of objects selectable in the editor
        enum class ObjectType
        {
            None,
            Model,
            Camera,
            Light,
            Keyframe
        };

        std::string getUniqueNumberedName(const std::string& baseName, uint32_t idSuffix, const std::set<std::string>& nameMap) const;

        std::set<std::string> mInstanceNames;
        std::set<std::string> mCameraNames;
        std::set<std::string> mLightNames;

        //
        // Picking
        //

        void select(const Scene::ModelInstance::SharedPtr& pModelInstance, const Model::MeshInstance::SharedPtr& pMeshInstance = nullptr);
        void deselect();

        void setActiveModelInstance(const Scene::ModelInstance::SharedPtr& pModelInstance);

        // ID's in master scene
        uint32_t mSelectedModel = 0;
        int32_t mSelectedModelInstance = 0;
        uint32_t mSelectedCamera = 0;
        uint32_t mSelectedLight = 0;
        uint32_t mSelectedPath = 0;

        Picking::UniquePtr mpScenePicker;

        std::set<Scene::ModelInstance*> mSelectedInstances;
        ObjectType mSelectedObjectType = ObjectType::None;

        CpuTimer mMouseHoldTimer;

        //
        // Gizmos
        //

        static const Gui::RadioButtonGroup kGizmoSelectionButtons;

        void setActiveGizmo(Gizmo::Type type, bool show);

        bool mGizmoBeingDragged = false;
        Gizmo::Type mActiveGizmoType = Gizmo::Type::Translate;
        Gizmo::Gizmos mGizmos;

        //
        // Editor
        //

        // Find instance ID of a model instance in the editor scene. Returns uint -1 if not found.
        uint32_t findEditorModelInstanceID(uint32_t modelID, const Scene::ModelInstance::SharedPtr& pInstance) const;
        // Update Camera, Lightbulb, and Keyframe model ID's in the Editor Objects Scene
        void updateEditorModelIDs();

        // Wireframe Rendering
        bool mHideWireframe = false;
        GraphicsState::SharedPtr mpSelectionGraphicsState;
        GraphicsProgram::SharedPtr mpColorProgram;
        GraphicsVars::SharedPtr mpColorProgramVars;

        // Separate scene for rendering selected model wireframe
        Scene::SharedPtr mpSelectionScene;
        SceneRenderer::SharedPtr mpSelectionSceneRenderer;

        // Separate scene for editor gizmos and objects
        Scene::SharedPtr mpEditorScene;
        SceneEditorRenderer::UniquePtr mpEditorSceneRenderer;
        Picking::UniquePtr mpEditorPicker;

        Model::SharedPtr mpCameraModel;
        Model::SharedPtr mpLightModel;
        Model::SharedPtr mpKeyframeModel;

        uint32_t mEditorCameraModelID = (uint32_t)-1;
        uint32_t mEditorLightModelID = (uint32_t)-1;
        uint32_t mEditorKeyframeModelID = (uint32_t)-1;

        // Maps between light models and master scene light ID
        std::unordered_map<uint32_t, uint32_t> mLightIDEditorToScene;
        std::unordered_map<uint32_t, uint32_t> mLightIDSceneToEditor;

        std::string mSelectedMeshString;
        Mesh::SharedPtr mpSelectedMesh;

        const static Gui::DropdownList kShadingModelList;

        //
        // Paths
        //
        void renderPath(RenderContext* pContext);

        void detachObjectFromPaths(const IMovableObject::SharedPtr& pMovable);

        void addSelectedPathKeyframeModels();
        void removeSelectedPathKeyframeModels();

        // When path editor closes
        void pathEditorFinishedCB();

        // When switching to a new active frame, or if active frame properties are changed
        void pathEditorFrameChangedCB();

        // When frames are added or removed
        void pathEditorFrameAddRemoveCB();

        bool mRenderAllPaths = false;

        PathEditor::UniquePtr mpPathEditor;
        std::unordered_map<const IMovableObject*, ObjectPath::SharedPtr> mObjToPathMap;

        DebugDrawer::SharedPtr mpDebugDrawer;

        GraphicsState::SharedPtr mpPathGraphicsState;
        GraphicsProgram::SharedPtr mpDebugDrawProgram;
        GraphicsVars::SharedPtr mpDebugDrawProgramVars;
    };
}