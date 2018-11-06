v3.1.1
------
- D3D Feature Level is now automatically selected by default
- Check Vulkan device's max supported API version if user requested a specific version
- Enable RGB32 format on AMD GPU's now that drivers support it correctly
- Added macro for suppressing deprecation warnings
- Add option to Gui::pushWindow() to choose if it should be have focus or not
- Add RenderContext::getBindFlags() getter
- Add Sampler::getDesc() getter
- Add Material::isEmissive() getter
- Add `alphaTest()` that uses Slang generics to select Sample method
- Add scene_unit key to fscene
- Add error check for missing file in Program creation
- Store scene bounding box in Scene, add getter
- Change OBJ/MTL to use SpecGloss by default, overrideable with metal_rough key in fscene
- Add frame reset option to video encoder UI
- Make video encoder retain options (do not delete the UI class between exports)
- Renamed getTriNormalsAndEdges to getTriNormalsAndEdgesInObjectSpace to clarify it's in object space
- Renamed getGeometricNormal to getGeometricNormalW to clarify it's in world space
- Improved accuracy of Light Probe integration

Bug Fixes:
- Fixed reflection data to use row major flag from Slang
- Fixed bug in RtProgram::addDefine() methods
- Fixed bug in scene exporter that it aborted on nan/inf
- Fixed Vulkan image tiling flag selection for textures

Dependencies:
- Updated packman to 5.7.1
- Updated Slang to 0.11.7
- Updated Falcor Media to 2.2.1

v3.1
------
- Falcor now requires Windows 10 SDK version 1809 (10.0.17763.0)
- `DebugDXR` and `ReleaseDXR` build configs have been removed, raytracing features are officially a part of DirectX 12 on Windows 10 RS5
- `RtSceneRenderer::renderScene()` and `RenderContext::raytrace()` have been updated as DXR ray dispatch now takes 3 parameters (width, height, depth) instead of two
- Added deprecation system
- Added a ProgramBase base class for better Program/RtProgram abstraction
- Added the option to pass CompilerFlags to many objects
- Added Logger::Level::Fatal
- Added `env_map` attribute and environment texture to scenes
- Added `Scene::setCamerasAspectRatio()` which will set all the scene's cameras aspect ratio at once
- Gui class supports multiple fonts. Default font is trebuchet, profiler font is consolas-bold
- Added support for high-DPI displays (GUI only)
- Added a way to query for device feature support (see `Device::isFeatureSupported()`)

Bug Fixes and Improvements:
- Added debug checks when binding compute/graphics vars
- Added debug checks in resourceBarrier() to make sure the resource has the correct flags 
- Added support for StructuredBuffers with VariableBuffer::renderUI
- Better abstraction of the alpha-test
- Use a priority_queue to help optimize descriptor-heap allocations
- Fixed VideoCapture UI
- Fixed debug visualization in Shadows sample
- No longer call SetSamplePositions when device doesn't support it

Dependencies:
- Added pybind11 2.2.4
- Updated GLM to 0.9.9.2
- Updated Vulkan SDK to 1.1.82.1
- Updated Slang to 0.11.4
- Updated imgui to 1.65

v3.0.7
------
- Updated Slang to 0.10.31

Bug Fixes:
- Fixed a crash when rendering a VariablesBuffer/ConstantBuffer UI without specifying a group name

v3.0.6
------
- Changed max bones to 256

Bug Fixes:
- Updated Slang to 0.10.30. Fixes SceneEditor shaders in Vulkan configs
- Apply scaling transforms in animations
- Fixed interpolation issues at the end of animations

v3.0.5
------
- Added support for exporting BMP and TGA images.
- Added `ConstantBuffer::renderUI()` to automatically render UI for editing a constant buffer's values.

Bug Fixes:
- Fixed crash when setting ForwardRenderer sample to MSAA with sample count 1
- std::string version of Gui::addTextBox() now correctly updates the user's string
- Fixed row-pitch calculation when copying texture subresources in DX12

v3.0.4
------
- Updated Slang to 0.10.24
- Added an option to create a `Program` from a string
- Added `CopyContext::updateSubresourceData()` which allows updating a region of a subresource
- Added `Program::Desc` has a new function - `setShaderModel()`. It allows the user to request shader-model 6.x, which will use dxcompiler instead of FXC
- Added support for double-quotes when parsing command line arguments. Text surrounded by double-quotes will be considered a single argument.

v3.0.3
------
- Added FXAA as an effect
- Support programmable sample position - `Fbo::setSamplePositions()` (DX only)
- Added RenderContext::resolveResource() and RenderContext::resolveSubresource() MSAA resolve functions
- Added support for setting shading model through fscene files and load flags. Also editable in Scene Editor

v3.0.2
------
- Various bug fixes
- Fixed Vulkan error spam seen when running Falcor's included samples
- Updated API abstraction interfaces to return const-ref where applicable
- Fixed crash when handling mouse/keyboard messages after the renderer has shut down

v3.0.1
------
- Added RenderContext::StateBindFlags, which allows the user to control which part of the `GraphicsState` will be bound to the pipeline
- Added helper functions to initialize D3D12 state-objects descs (root-signature, GSO-desc)
- Added a function that creates a texture from an resource API handle
- Added a new sample: LightProbeViewer. Allows you to see how light probe images look after pre-integration with various sample counts and materials.
