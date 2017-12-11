/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "MultiRendererSample.h"
#include "Renderers/ShaderToy.h"
#include "Renderers/LiveTrainingDemo.h"

// We're need to override our scene loader to provide good default lights (if there are none),
//    since our LiveTrainRenderer assumes there's (at least) one light in the scene.
class DemoApp : public MultiRendererSample
{
    virtual Scene::SharedPtr loadScene(const std::string& filename) override;
};

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    // Our application program
    MultiRendererSample sample;
    
    // Setup our demo config
    SampleConfig config;
    config.windowDesc.title = "Live Training & Python Integration Demo";
    config.windowDesc.resizableWindow = false;
    config.windowDesc.width = 1800;
    config.windowDesc.height = 980;
    config.freezeTimeOnStartup = true;

    // Create our Python integration renderer
#if FALCOR_USE_PYTHON
    Renderer::SharedPtr liveTrain   = LiveTrainRenderer::create();     // The renderer that does Python-based live training (default; program starts showing this one)
    sample.addRenderer(liveTrain);
#endif

    // Create another, fallback renderer to run if there's no Python enabled.
    Renderer::SharedPtr toyDemo     = ShaderToyRenderer::create();     // Another, super-simple renderer as an example of how to use the MultiRendererSample
    sample.addRenderer(toyDemo);

    // Run the program
#ifdef _WIN32
    sample.run(config);
#else
    sample.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}

Scene::SharedPtr DemoApp::loadScene(const std::string& filename)
{
    Scene::SharedPtr pScene = Scene::loadFromFile(filename);
    if (pScene != nullptr)
    {
        if (pScene->getCameraCount() == 0)
        {
            const Model* pModel = pScene->getModel(0).get();
            Camera::SharedPtr pCamera = Camera::create();
            vec3 position = pModel->getCenter();
            float radius = pModel->getRadius();
            position.y += 0.1f * radius;
            pScene->setCameraSpeed(radius * 0.03f);
            pCamera->setPosition(position);
            pCamera->setTarget(position + vec3(0, -0.3f, -radius));
            pCamera->setDepthRange(0.1f, radius * 10);
            pScene->addCamera(pCamera);
        }

        if (pScene->getLightCount() == 0)
        {
            DirectionalLight::SharedPtr pDirLight = DirectionalLight::create();
            pDirLight->setWorldDirection(vec3(-0.189f, -0.861f, -0.471f));
            pDirLight->setIntensity(vec3(1, 1, 0.985f) * 10.0f);
            pDirLight->setName("DirLight");
            pScene->addLight(pDirLight);
            pScene->setAmbientIntensity(vec3(0.1f));
        }
    }
    return pScene;
}

