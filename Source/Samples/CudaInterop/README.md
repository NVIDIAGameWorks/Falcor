# Using the CUDA interop with a project

In order to use the CUDA interop, the following steps will need to be completed:
1. In Visual Studio, make a new CUDA Runtime project and add it to Falcor. (You must have the CUDA Toolkit installed to do so.)
2. Right-click on References under the new project in the Solution Explorer, select Add Reference, and add Falcor.
3. Open the Property Manager and add the Falcor and FalcorCUDA property sheets to both Debug and Release.
4. Open the project properties. If the project will produce a Windows application, go to General -> Configuration Type and change the setting to **Application (.exe)**, and go to Linker -> System -> SubSystem and change the setting to **Windows**. If the project is a DLL, then only the Configuration Type will need to be changed to **Dynamic Library (.dll)**.