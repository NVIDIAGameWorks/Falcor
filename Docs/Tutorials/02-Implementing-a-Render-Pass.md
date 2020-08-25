### [Index](../index.md) | [Tutorials](./index.md) | Implementing a Render Pass

--------

# Implementing a Render Pass

Now that you've successfully loaded a render graph into Mogwai through a script, let's create one. A render graph is formed from a number of component render passes. This tutorial will focus on creating render passes through an example pass that blits a source texture into a destination texture; graph creation and editing will be the subject of the next.

## Creating a Render Pass Project
All render passes are contained within projects in the solution. Render pass projects can contain any number of passes, though we suggest keeping a limit on the number of passes per project. Follow these steps to set up a fresh project:

1. Navigate to `Source/RenderPasses`.
2. Execute `make_new_pass_project.bat <PassName>` with the name of your pass. This will create a new directory with a new DLL project for your pass and add both the `Falcor.props` property sheet and reference to the Falcor project.
3. Open `Falcor.sln` and add (`Add -> Existing Project`) the newly created project to the `RenderPasses` folder.
4. Open the Configuration Manager. Under `Active Solution Configurations`, click `<Edit...>`, and remove the `Debug` and `Release` configurations, which were automatically added by Visual Studio.

You do not need to do anything else as part of setup. For any pass that contains shaders, the shader files are by default placed in `Data/RenderPasses/$(ProjectName)` under the output folder as part of the build process. This can be changed in the project properties `Shader Source`->`Destination Subfolder`, but the recommendation is to leave it at the default to avoid name clashes.

For our example, we will execute `make_new_pass_project.bat ExampleBlitPass`. (Name is intended to avoid conflicting with an already existing `BlitPass`, which is a bit more complex than the one we will write.)

## Implementing a Render Pass
If you open the header and source files, you'll notice your pass already implements some functions inherited from `RenderPass`. There are other optional functions you can inherit, which you can find in `RenderPass.h`. The following four are required for all render passes:

### `getDesc()`
This function (found in the header) returns a description of what the pass does.
```c++
virtual std::string getDesc() override { return "Blits a texture into another texture"; }
```
`ExampleBlitPass` will simply copy a texture from a source to a destination.

### `create()`
This function is used to create a pass and can optionally take a dictionary containing values to initialize the pass with.

Our example pass contains nothing that needs to be initialized, so we only need to call the constructor and return the object wrapped inside a shared pointer, which is what the template does by default.
```c++
ExampleBlitPass::SharedPtr ExampleBlitPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pPass = SharedPtr(new ExampleBlitPass);
    return pPass;
}
```

### `reflect()`
This function describes what resources the pass needs and sets a name to refer to them by using the `RenderPassReflection` class. These can be marked as any of the following types:
- `Input` and `Output` are self-explanatory.
- Marking both `Input` and `Output` declares a pass-through resource: A resource is required as an input, the pass will update the resource, then that resource can also be referred to as an output with the same name.
- `Internal` tells the render graph system to allocate a resource for use within the pass. This is currently identical to declaring resources directly as a member of your RenderPass class, but this API allows future versions to alias and reuse resource memory automatically behind the scenes.

All of these variations have corresponding helper functions to simplify usage: `addInput()`, `addOutput()`, `addInputOutput()`, `addInternal()`.

`ExampleBlitPass` only requires a single input and output and no internal resources. We create the `RenderPassReflection`, add an input and an output, and return it. If the pass required any internal resources, they would be added here as well.

```c++
RenderPassReflection ExampleBlitPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput("input", "the source texture");
    reflector.addOutput("output", "the destination texture");
    return reflector;
}
```

### `execute()`
This function runs the pass and contains all required render and/or compute operations to produce the desired output. All requested resources are available through `renderData` under the same names assigned to them in `reflect()`.

`ExampleBlitPass` copies the source texture to the destination texture, which are accessed through `renderData["input"]->asTexture()` and `renderData["output"]->asTexture()`, respectively. `RenderContext` already implements the blit operation which takes a `ShaderResourceView` as a source and a `RenderTargetView` as a destination. We will use it like so:
```c++
void ExampleBlitPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pSrcTex = renderData["input"]->asTexture();
    const auto& pDstTex = renderData["output"]->asTexture();

    if (pSrcTex && pDstTex)
    {
        pRenderContext->blit(pSrcTex->getSRV(), pDstTex->getRTV());
    }
    else
    {
        logWarning("ExampleBlitPass::execute() - missing an input or output resource");
    }
}
```

## Registering Render Passes

Every render pass library project contains a `getPasses()` function which registers all render passes implemented in the project. Let's update the pass' description in this function as well.
```c++
extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("ExampleBlitPass", "Blits a texture into another texture", ExampleBlitPass::create);
}
```

We will ignore further details regarding render passes and their implementation for the purposes of this tutorial. Additional information can be found [here](../Usage/Render-Passes.md).
