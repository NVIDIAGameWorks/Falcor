### [Index](../index.md) | [Development](./index.md) | Unit Testing

--------

# Unit Testing

Falcor has a custom unit testing system. Unit tests are implemented in the `Tools/FalcorTest` project.

When `FalcorTest.exe` runs, you should see it output something like:

```
Running 5 tests
  unittest.cpp/TestCPUTest (CPU)                              : PASSED (0 ms)
  unittest.cpp/TestGPUTest (GPU)                              : PASSED (22 ms)
  shadingutilstests.cpp/RadicalInverse (GPU)                  : PASSED (65 ms)
  shadingutilstests.cpp/Random (GPU)                          : PASSED (1111 ms)
  shadingutilstests.cpp/SphericalCoordinates (GPU)            : PASSED (571 ms)
```

The return code is `0` if tests pass or a positive integer indicating the number tests that failed.

## Running Unit Tests

### From Script

Run the batch file `tests/run_unit_tests.bat` with appropriate settings.

```
$ ./run_unit_tests.bat --help
usage: run_unit_tests.py [-h] [-c CONFIG] [-e ENVIRONMENT] [-f FILTER]
                         [-x XML_REPORT] [-r REPEAT] [--skip-build]
                         [--list-configs]

Utility for running unit tests.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Build configuration
  -e ENVIRONMENT, --environment ENVIRONMENT
                        Environment
  -f FILTER, --filter FILTER
                        Regular expression for filtering tests to run
  -x XML_REPORT, --xml-report XML_REPORT
                        XML report output file
  -r REPEAT, --repeat REPEAT
                        Number of times to repeat the test.
  --skip-build          Skip building project before running tests
  --list-configs        List available build configurations.
```

### From Visual Studio or the Command Line

Run the executable `build/<preset name>/bin/[Debug|Release]/FalcorTest.exe`

```
  FalcorTest {OPTIONS}

    Falcor unit tests.

  OPTIONS:

      -h, --help                        Display this help menu.
      -f[filter], --filter=[filter]     Regular expression for filtering tests
                                        to run.
      -r[N], --repeat=[N]               Number of times to repeat the test.
      --enable-debug-layer              Enable debug layer (enabled by default
                                        in Debug build).
```

## Add a New Unit Test

To add a new test, either edit an appropriate `.cpp` file in `Source/Tools/FalcorTest/Tests/` or create a new `.cpp` file and add it there. Add the newly created file to the `FalcorTest` project (matching the directory structure).

Note that unit test code should live in `.cpp` files seperate from the tested implementation. Shader code for the tests should be placed *alongside* the test `.cpp` file and use the `ShaderSource` type in the project to make sure the shader file is copied to the output directory upon build.

Make sure you've got `#include "UnitTest.h"` in your `.cpp` test file and you're ready to go. We'll walk through simple examples for both CPU and GPU unit tests.

### CPU Tests

As a first example, let's write a simple test for `std::sqrt` on the CPU:

```c++
CPU_TEST(Sqrt)
{
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(i, std::sqrt(i * i));
    }
}
```

The `CPU_TEST` macro handles the details of registering the test with the test system and generating the appropriate function signature; with that, one adds an opening brace and starts writing code.

The `EXPECT_EQ` macro is also defined in `UnitTest.h`; it takes two values and the test fails if they aren't equal (there's similarly `EXPECT_NE` that tests for inequality, and `EXPECT_{GE,GT,LE,LT}`, which cover the remaining ordering relations). Finally, there's `EXPECT()`, which takes a single boolean value.

In general, it's best to use one of the `EXPECT_*` macros if rather than `EXPECT`. Upon failure, they include the two values in the test output.  If `std::sqrt(2 * 2)` returned `3` in the above test, `EXPECT_EQ` would give us the output:

```
Test failed: i == std::sqrt(i * i) (2 vs. 3)
```

### GPU Tests

As a second example, let's consider testing that multiplication on the GPU is working correctly.

First, we have a short compute shader, which we might name `Square.cs.slang`. A `RWStructuredBuffer` is used to return the result, and the value to be multiplied is provided in a constant buffer:

```hlsl
RWStructuredBuffer<float> result;
cbuffer Inputs { float value; };

[numthreads(1, 1, 1)]
void main()
{
    result[0] = value * value;
}
```

Next, we add the following straightforward code in the C++ file to define the test.

```c++
GPU_TEST(Square)
{
    ctx.createProgram("Square.slang.hlsl");
    ctx.allocateStructuredBuffer("result", 1);
    const float value = 3.f;
    ctx["Inputs"]["value"] = value;
    ctx.runProgram();

    const float *result = ctx.mapBuffer<const float>("result");
    EXPECT_EQ(result[0], value * value);
    ctx.unmapBuffer("result");
}
```

Within a `GPU_TEST` function, an instance of the `GPUUnitTestContext` is available via a parameter named `ctx`. `GPUUnitTestContext` provides a variety of helpful methods that make it possible to run GPU-side compute programs, allocate buffers, set parameters and check results with a minimal amount of code.

## Output

One can add additional output all of the `EXPECT*` macros just by using `operator<<` to print more values, like like C++ `std::ostream`. This additional output is only printed if a test fails. Thus, if we instead wrote `EXPECT_EQ` like this:

```c++
EXPECT_EQ(result[0], value * value) << "value = " << value;
```

Then upon failure, we'd see:

```
Test failed: result[0] == value * value (4 vs. 9) value = 3
```

This additional information can be helpful in understanding what went wrong.

## Skipping Tests

Broken tests can temporarily be skipped by changing `CPU_TEST(SomeTest)` to `CPU_TEST(SomeTest, "Skipped due to ...")`. The message will be printed when running the test and the test will finish with status `SKIPPED`, which is not considered a failure. The same principle applies to `GPU_TEST` as well.
