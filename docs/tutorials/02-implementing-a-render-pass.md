### [Index](../index.md) | [Tutorials](./index.md) | Implementing a Render Pass

--------

# Implementing a Render Pass
Now that you've successfully loaded a render graph into Mogwai through a script, let's create one. A render graph is formed from a number of component render passes. This tutorial will focus on creating render passes through an example pass that blits a source texture into a destination texture; graph creation and editing will be the subject of the next.

## Creating a Render Pass Library
All render passes are contained in `Source/RenderPasses` and are shared libraries. Render pass libraries can contain any number of passes, though we suggest keeping a limit on the number of passes per library.

Run `tools/make_new_render_pass_library.bat <LibraryName>` to create a new render pass library. This will create a new subfolder containing the new render pass library and adjust the build scripts.

## Implementing a Render Pass
If you open the header and source files, you'll notice your pass already implements some functions inherited from `RenderPass`. There are other optional functions you can inherit, which you can find in `RenderPass.h`. The following are required for all render passes:

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

`ExampleBlitPass` copies the source texture to the destination texture, which are accessed through `renderData.getTexture("input")` and `renderData.getTexture("output")`, respectively. `RenderContext` already implements the blit operation which takes a `ShaderResourceView` as a source and a `RenderTargetView` as a destination. We will use it like so:
```c++
void ExampleBlitPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pSrcTex = renderData.getTexture("input");
    const auto& pDstTex = renderData.getTexture("output");

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

Every render pass library project contains a `getPasses()` function which registers all render passes implemented in the project.
```c++
extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(ExampleBlitPass::kInfo, ExampleBlitPass::create);
}
```

You can adjust the description of the render pass library by adjusting the following line:

```c++
const RenderPass::Info ExampleBlitPass::kInfo { "ExampleBlitPass", "Blits a texture into another texture." };
```

We will ignore further details regarding render passes and their implementation for the purposes of this tutorial. Additional information can be found [here](../Usage/Render-Passes.md).
