add_plugin(TestPasses)

target_sources(TestPasses PRIVATE
    TestPasses.cpp
    TestPyTorchPass.cpp
    TestPyTorchPass.h
    TestPyTorchPass.cs.slang
    TestRtProgram.cpp
    TestRtProgram.h
    TestRtProgram.rt.slang
)

target_copy_shaders(TestPasses RenderPasses/TestPasses)

target_source_group(TestPasses "RenderPasses")
