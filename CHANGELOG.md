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
