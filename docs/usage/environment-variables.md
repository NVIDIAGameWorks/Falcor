### [Index](../index.md) | [Usage](./index.md) | Environment Variables

--------

# Environment Variables

The following environment variables are used in Falcor:

| Variable Name | Description |
|-----|-----|
| `FALCOR_DEVMODE` | Set to `1` to enable development mode. In development mode, shader and data files are picked up from the `Source` folder instead of the binary output directory allowing for shader hot reloading (`F5`). Note that this environment variable is set by default when launching any of the Falcor projects from Visual Studio. |
| `FALCOR_MEDIA_FOLDERS` | Specifies a semi-colon (`;`) separated list of absolute path names containing Falcor scenes. Falcor will search in these paths when loading a scene from a relative path name. |
