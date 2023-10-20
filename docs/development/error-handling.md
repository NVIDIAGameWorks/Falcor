### [Index](../index.md) | [Development](./index.md) | Error Handling

--------

# Error Handling

**NOTE: These guidelines represent the recommended way of handling errors going forward. Pre-existing code uses a mix of error handling methods, including not checking for errors at all.**

Falcor and its applications must be able to handle errors that occur during development and execution in a consistent manner.
Errors are broadly classified into three main categories, which are handled differently:

1. *Logic errors* are conditions that should never occur; they are the result of programmer mistakes. The main mechanism for handling them is *assertions*.
2. *Runtime errors* are failures that are not expected, but can happen due to adverse runtime conditions or invalid input. The main mechanism for handling them is to throw *exceptions*.
3. *Errors that are likely to happen* and where the common case is to check for and recover from the failure. These are handled by *return values* or *return codes*.

Exceptions is the primary error handling method because it offers several advantages over other methods, such as return codes:

* Failures don't go unnoticed as a result of missing to check a return code. An exception forces the calling code to recognize an error and handle it.
* The exception is passed up the stack until the application handles it or the program terminates. Intermediate layers automatically let the error propagate.
* The exception stack-unwinding destroys all objects in scope according to well-defined rules.
* There is a clean separation between the code that detects the error and the code that handles the error.
* Exceptions can pass the Python API boundary in a well defined way.

Note that error reporting is a separate topic from error handling and is discussed further down. All failure conditions that occur must be appropriately detected and handled.

## Assertions

* Use asserts to check for logic errors in C++ code; errors that should never occur and are a result of programmer mistakes rather than runtime conditions.
* Use `static_assert()` to check for logic errors at compile time if possible, for example, type checking of template arguments, or checking struct sizes.
* Use `FALCOR_ASSERT()` to check for logic errors at runtime.
* Use `FALCOR_ASSERT_EQ`, `FALCOR_ASSERT_NE`, `FALCOR_ASSERT_GE`, `FALCOR_ASSERT_GT`, `FALCOR_ASSERT_LE`, `FALCOR_ASSERT_LT` to do binary comparison between two values. These assert macros will print messages that not only show the tested condition but also the values that lead to failure.
* Use asserts generously. Even trivially correct code might be affected by changes elsewhere.
* Assertions are only enabled in Debug builds by default. Make a habit of running the application in Debug mode regularly to make sure that no asserts trigger.

## Exceptions

* Use exceptions to check for runtime errors that might occur. For example, errors due to bad user input, missing or corrupt files, running out of memory, etc.
* Use exceptions to check for shader errors. Shader programs are dynamically loaded and errors in the code are considered runtime errors.
* Use exceptions to check arguments to public API functions. Even if your code is correct, you might not have control over what arguments a user might pass in.
* Throw exceptions with descriptive messages. The built-in exception classes take format strings to make this easier.
* Throw exceptions by value, catch them by reference.
* Do not catch exceptions in user code, other than in rare cases. Falcor's default error handler logs the error and terminates the application, which is often the most reasonable action.

Falcor provides its own exception classes (inherited from `std::exception`) and some utility macros for improved development experience. The main exception class is `Exception`. For simplicity, we try to avoid using many different categories of exceptions. The main exception should always be thrown using the `FALCOR_THROW(fmt, ...)` macro which supports format strings for formatting the error message. This macro has the advantage of being able to break into the debugger if one is attached. This is convenient during development as we automatically break into the debugger before throwning the exception, making it easy to find the code/conditions responsible for the exception. Alternatively one can use the debugger _break on exception_ functionality but this can be cumbersome to use as it breaks on **all** exceptions, both local and external code. The `FALCOR_THROW` macro also logs a stack trace of where the exception is thrown in case no debugger is attached, which helps post-mortem debugging.

In addition to `FALCOR_THROW` there is also a `FALCOR_CHECK(cond, fmt, ...)` macro which can be used to check conditions, including checking for valid arguments as well as for runtime invariants. The macro is simply defined as:

```c++
#define FALCOR_CHECK(cond, fmt, ...)        \
    if (!(cond))                            \
        FALCOR_THROW(fmt, ##__VA_ARGS__)
```

## Assertions vs. Exceptions

Sometimes the line between using _assertions_ and _exceptions_ can be blurry. In general, _exceptions_ should be used rigorously on all public facing API, that is especially true for functions exposed through the Python API. Internal code can rely more heavily on assertions. Here is a simple example:

```c++
// This is the public facing function.
// We throw an exception if the data is not in a valid format.
void processTriplets(std::span<float> data)
{
    FALCOR_CHECK(
        data.size() % 3 == 0,
        "'data' needs to contain multiples of 3 values but got size={}.",
        data.size()
    );
    processTripletsInternal(data)
}

// This is the internal function.
// We assume the data to be in the correct format, but we still check our
// precondition to help finding logical errors during development/refactoring.
void processTripletsInternal(std::span<float> data)
{
    FALCOR_ASSERT(data.size() % 3 == 0);
}
```

## Return values

* Operations where failure due to non-programmer errors is likely should use null return value, error codes, etc., *if* the expected common case is to actually check for and handle the failures.
* Examples of such cases may be user dialogs, bad input in some config file, etc.
* Return values or error codes may also be used in tight performance critical sections, where both detection and handling a failure is done locally. For example, when iterating over a vertex buffer to check for invalid vertices.

## Error reporting

Falcor provides some helpers for reporting errors to the user. Note that apart from warnings, application code should generally prefer throwing exceptions over reporting errors. The framework will catch these exceptions and report them accordingly. In some cases though, applications might want to wait for user input upon an error condition, for example if a script failed to execute, allowing users to fix the script and retry. But these cases are generally rare.

### Warnings

* Use `logWarning(msg)` to report a non-critical, but unexpected conditions.
* For example, warning might be appropriate for a condition that negatively affects performance, but otherwise does not affect execution.
* Warnings are printed to the console, debugger output window and the log file (depending on enabled log outputs) but are easily missed. Do not rely on a user seeing a warning message unless explicitly looking for it.

### Errors

* Use `reportError(msg)` to report critical errors where the user has the option to continue execution.
  * The message is logged with level `Error`.
  * A message box is shown (unless disabled) and the user has options to abort (terminate the application), enter the debugger (if attached) or continue.
* Use `reportErrorAndAllowRetry(msg)` to critical report errors where the user has the option to retry the operation. This is for example used when shaders fail to compile.
  * The message is logged with level `Error`.
  * A message box is shown (unless disabled) and the user has options to abort (terminate the application), enter the debugger (if attached) or retry.
* Use `reportFatalError(msg)` to report fatal errors where the application should be terminated immediately to avoid any undefined behavior.
  * The message is logged with level `Fatal`.
  * A message box is shown (unless disabled) and the user has options to abort (terminate the application) or enter the debugger (if attached).

If message boxes are disabled using `setShowMessageBoxOnError(false)`, all the above `reportError` functions terminate the application immediately after logging the error.

Do not call any of the `reportError` functions before throwing an exception, as the default exception handler will already report the error.

## Logging

Falcor provides a logging infrastructure. Messages are logged using one of the following global functions: `logDebug`, `logInfo`, `logWarning`, `logError` and `logFatal`.

### Levels

Falcor uses the following guidelines for using different log levels:

| Level | Description |
| --- | --- |
| `Debug` | Messages that may be needed for diagnosing issues and troubleshooting. |
| `Info` | Messages that are purely informative. This level should not be used to indicate any unexpected conditions. |
| `Warning` | Messages that indicate that something unexpected happened, but that the application is able to continue running. |
| `Error` | Messages that indicate that an error occured, and the application might not be able to continue running correctly. |
| `Fatal` | Messages that indicate that a fatal error occured and the application needs to terminate immediately. |

### Verbosity

The verbosity of the logger can be configured by setting the level up to which messages are being logged. By default it is set to `Logger::Level::Info`, meaning that all levels other than `Debug` are logged. The verbosity can be changed using `Logger::setVerbosity`.

### Output streams

The logging system can log to the following outputs streams:

- Console (stdout)
- Visual Studio Debug Window (if debugger is attached)
- File

By default all three output streams are enabled. This can be changed using `Logger::setOutputs`.

When logging to a file, the logger automatically chooses the filename based on the executed process's name and an number incremented every time the process is launched. For `Mogwai.exe` this results in log files named `Mogwai.exe.0.log`, `Mogwai.exe.1.log` etc.

**Note**: Falcor 4.4 and below used the logger to pop up dialog boxes on error conditions or when allowing users to retry an operation. In current versions, the logger is soley used for logging messages and has no other logic attached to it.

## Guidelines for Falcor Users

### Resource allocation

* Failing to allocate a resource is a runtime error, for which Falcor will throw an exception internally.
* User code can assume resource allocation succeeds and does not have to check for `nullptr`.

### Shader program creation

* Shader compilation errors are handled internally in Falcor.
* The default behavior is to ask the user how to proceed, giving the user a chance to correct the shader and retry.
* If compilation fails and the user selects abort/cancel, it is a runtime error and an exception is thrown inside Falcor.
* User code can assume shader program creation succeeds and does not have to check for `nullptr`.

### Shader bindings

* Shader binding code should assume that the variable exists in the shader program.
* If it does not, it is a runtime error and an exception is thrown inside Falcor.
* User code does not have to check return values since the error has already been handled.
* If you are unsure if a variable exists, the shader reflection API should be used to query it first.

### Shader reflection

* Failure is an expected outcome of shader reflection, as it is used to query for the existence of shader variables.
* Therefore, the shader reflection API uses return values to indicate success or failure, e.g., returning `nullptr` instead of a type reflection object if the type cannot be found.
* User code must check the return values from shader reflection calls.
