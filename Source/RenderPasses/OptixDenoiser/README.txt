Optix Denoiser Render Pass
--------------------------

See the main README on how to install OptiX.

-----------------------------------------------------------------------------------------------------

When using this pass in a Python render graph:

* Make sure to add to your Python script:
    - loadRenderPassLibrary("OptixDenoiser.dll")

* Create a pass something like this:  (Default settings are pretty good, see config options below.)
    - graph.addPass(createPass("OptixDenoiser", {}), "Denoiser")

* Connect the denoiser into your graph by inputting the noisy result.
  I did denoising post-tonemap here, but that's not required:
    - graph.addEdge("ToneMapping.dst", "Denoiser.color")

* (Optionally) Include guide images for denoising.  Not connecting these
  will disable some GUI configuration options.
    - graph.addEdge("GBuffer.diffuseOpacity", "Denoiser.albedo")
    - graph.addEdge("GBuffer.normW",          "Denoiser.normal")
    - graph.addEdge("GBuffer.mvec",           "Denoiser.mvec")

* Consume the denoised result, either by marking it as an output or
  passing the result to another pass (via addEdge())
    - graph.markOutput("Denoiser.output")

-------------------------------------------------------------------------------------------------------

You can configure the denoiser from Python by added paremters between the {} when creating the pass.
These controls (and a few more) are all available as part of the GUI, though some are not shown when
the render pass is not given all of the optional inputs textures.

By default, when connecting the OptiX denoiser to a render graph, it is:
  (a) Enabled.    (control with 'enabled' : False or True)

  (b) Showing the denoised result only.  (You can blend between the noisy input and the denoised result
      using the `blend' : 0.0 parameter.  0.0 = denoised only, 1.0 = noisy only, values between are
      interpolated between the endpoints.)

  (c) Not denoising the alpha channel of the noisy color (control with 'denoiseAlpha: True or False)
      Denoising alpha is somewhat more expensive, and often denoising occurs as a last step, where
      alpha is irrelevant.

  (d) Uses either the HDR or Temporal OptiX denoiser (depending if motion vectors are provided)
      - Can be overridden in the Python file via 'model' : OptixDenoiserModel.LDR, OptixDenoiserModel.HDR,
        or OptixDenoiserModel.Temporal

  (e) Using the most guides possible.
      - If you pass in noisy color only, no guides are used.
      - If you pass in an albedo channel, then albedo will be used for denoising.
      - If you pass in a normal channel, the normal guids will be used for denoising.
      - If you pass in a motion vector, temporal denoising will be used

      This is not currently fully configurable via Python.

-------------------------------------------------------------------------------------------------------

Debugging:

1) If the DLL fails with an error message about optixInit() failing, please update your driver.

2) If Mogwai crashes when loading any render script containing the OptixDenoiser DLL, please double check
   that all dependencies of OptixDenoiser.dll got copied to the `Bin` directory where you are running from.
   (Unfortunately when loading Windows DLL, loading fails if either (a) the DLL does not exist or (b) one of
   the DLL's dependencies doesn't exist.  We ensure (a), but since we do not distribute OptiX and CUDA,
   sometimes these files are not in the search path.  Unfortunately, some Falcor versions fail cryptically
   on DLL load when dependencies do not exist.)

3) Double check that all passes using OptiX and/or CUDA in your renderer link and depend on the *same*
   versions of OptiX and CUDA.  Mixing and matching will cause extremly hard-to-debug null dereferences.
