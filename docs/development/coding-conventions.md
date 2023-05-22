### [Index](../index.md) | [Development](./index.md) | Coding Conventions

--------

# Coding Conventions

This covers the basic coding conventions and guidelines for all C++ and Slang code that is submitted to this repository.

- It's expected that you will not love every convention that we've adopted.
- These conventions establish a modern C++17 style.
- Please keep in mind that it's impossible to make everybody happy all the time.
- Instead appreciate the consistency that these guidelines will bring to our code and thus improve the readability for others.

## General Guidelines

- Code is a medium for communication. When in doubt, favor code that is readable and maintainable.
- When in doubt, make your additions/edits fit in with the style of the code around it. It should not be possible to see where code switches from one programmer to another.
- In particular, resist the urge to _clean up_ code from other developers when doing an edit/merge, *even if* such cleanup brings the code more in line with the official style in this document. Seemingly innocent code cleanup can complicate subsequent merges by making diffs more complicated. Cleanup changes should be handled as separate edits and should take into account potential conflict with in-progress feature work.
- Follow the [error handling policy](./error-handling.md) described separately.

## Documentation and Comments

- When writing comments/documentation, remember that code could be read by a wide variety of people with different languages and cultures.
- Favor a simple and plain-spoken style over fancy words, langue-/culture-specific idioms, or pop-culture references.
- However, don't avoid technical terms when they are the correct and appropriate terms for what code is doing (e.g., *radiant flux* is a complex term, but it might be the right one).

## Files

- All files must start with the legal header (run `tools/update_legal_headers.bat` to do this automatically).
- All files must end in blank line (use `.editorconfig` to enforce this).
- Folders, header and source files should be named with **PascalCase**. If applicable, the name should represent the type of the class defined in the file.

### Rules for C++ files

- Header files should have the extension `.h`
- Source files should have the extension `.cpp`
- Header files must be self-sufficient, i.e. they need to compile if being the only include in a translation unit.
- Header files must include the preprocessor directive to only include a header file once:

```c++
#pragma once
```

- Source files should include the associated header in the first line of code after the commented license banner.
- Includes in source files should be arranged into groups ordered by:
  1. Matching header for the cpp file is first, if it exists.
  2. Other local includes from the same project.
  3. Other local includes from other projects (for example, includes from Falcor from a render pass).
  4. Other 3rd party includes (using `#include <...>` syntax).
  5. STL headers.
  6. C-library headers (prefer the new C-library headers, e.g. `#include <cmath>` instead of `#include <math.h>`)

### Rules for Slang files

- Slang files that are intended to be imported as modules should be named `.slang`
- Slang files that for some reason cannot be imported and must be included should be named `.slangh`
- Slang files that contain program entry points should include a suffix: `.rt.slang` for ray tracing, `.cs.slang` for compute, and `.3d.slang` for graphics programs (if the file has only a single shader stage, it is fine to name it according to that stage, for example `.ps.slang`). These suffices make it easier to find the entry points and prevent accidental `import` of a module that cannot be imported.
- Slang files that contain program entry points should never `__export import` another Slang file, since they are not to be imported.
- Slang files should be placed alongside the host source files in the directory tree.
- Slang files that are included as headers (on the host or device) must include the preprocessor directive `#pragma once` to only include a header file once.
- Slang files that are imported as modules should _not_ include the preprocessor directive `#pragma once`.


## Namespaces

- Namespaces should be named with **PascalCase**.
- All code in the folder `Falcor` should be in namespace `Falcor`.
- `using namespace` is only allowed inside source files. It should never be used in header files.
- `using namespace std` is disallowed even in source files. If you want to save some work, alias the type you need from the std namespace, or use `auto`.
- Prefer anonymous namespaces to `static` functions and variables in `.cpp` files (`static` should be omitted inside anonymous namespaces).
- Do not indent code inside namespaces.

## Naming Guidelines

- When there are multiple choices, favor US English spellings (e.g., `Color` instead of `Colour`, `Gray` instead of `Grey`). This isn't a value judgement, but we need a way for programmers to be able to predict how a function/type name will be spelled.
- Names that involve acronyms (e.g. NASA) or initialisms (e.g. BRDF, AABB) and which are conventionally written in all capitals in English, should be written either in all caps or all lowercase, depending on where they appear in a name, rather than force awkward capitalizations like `Brdf`.
- In extreme cases where multiple initialisms/acronyms must appear consecutively, it is okay to break with this convention for overall readability (so `XmlHttpRpc` instead of `XMLHTTPRPC`), but you probably shouldn't be giving things names like that anyway.
- Markers for dimensionality (e.g. `2D`, `3D`) are effectively initialisms (they are pronounced as such in English).
- The term `ID` is also pronounced as initialism, and is used as part of other all caps types such as `UID`, `UUID`, and `GUID`, which have a long standard all caps style. Therefore, the `ID` itself shall also be all caps.
- Abbreviations should be applied consistently across type/function names: either a term is *always* abbreviated, or it *never* is.
    - As a general rule, acronyms should be used only if they're understandable to people who are well versed in the general topic of graphs on the GPU, but not with Falcor itself.
- Specific terms that should always be abbreviated in type/function names:
    - `ID` for "identifier"
    - `Desc` for "description" or "descriptor" (in the sense of a value used to describe a thing to be constructed, but *not* in the case of a resource/binding "descriptor" in Vulkan/D3D12).
    - `Ptr` for "pointer"
- The following names cannot be used according to the [C++ standard](https://en.cppreference.com/w/cpp/language/identifiers):
    - Names that are already keywords
    - Names with a double underscore anywhere are reserved
    - Names that begin with an underscore followed by an uppercase letter are reserved
    - Names that begin with an underscore are reserved in the global namespace.
- Method names must always begin with a verb, this avoids confusion about what a method actually does:

```c++
someVector.getLength();
someObject.applyForce(x, y, z);
someObject.isDynamic();
someTexture.getFormat();
```

- The terms get/set or is/set (*bool*) should be used where an attribute is accessed directly. This indicates there is no significant computation overhead and only access:

```c++
mesh.getName();
mesh.setName("Bunny");
light.isEnabled();
light.setEnabled(true);
```

- Use stateful names for all boolean variables (`enabled`, `initialized`) and leave questions for methods (`isEnabled()` and `isInitialized()`)

```c++
bool isEnabled() const;
void setEnabled(bool enabled);

void doSomething()
{
    bool initialized = mSomeSystem.isInitialized();
    ...
}
```

- Please consult the [antonym list](https://gist.github.com/maxtruxa/b2ca551e42d3aead2b3d) if naming symmetric functions.
- Avoid redundancy in naming methods and functions. The name of the object is implicit, and must be avoided in method names:

```c++
line.getLength(); // NOT: line.getLineLength();
```

- Prefer function names that indicate when a method does significant work:

```c++
float waveHeight = wave.computeHeight(); // NOT: wave.getHeight();
```

- Avoid public method, arguments and member names that are likely to have been defined in the preprocessor, when in doubt, use another name or prefix it:

```c++
size_t malloc; // BAD
size_t bufferMalloc; // GOOD
int min, max; // BAD
int boundsMin, boundsMax; // GOOD
```

- Avoid conjunctions and sentences in names as much as possible:

```c++
bool skipIfDataIsCached; // BAD
bool skipCachedData; // GOOD
```

- Use `Count` at the end of a name for the number of items:

```c++
size_t numberOfShaders; // BAD
size_t shaderCount; // GOOD
```

## Name Prefixing and Casing

The following table outlines the naming prefixing and casing used:

|Construct| Prefixing / Casing|
| --- | --- |
| class, struct, enum class and typedef | `PascalCase` |
| constants | `kCamelCase` |
| enum class values | `PascalCase` |
| functions | `camelCase` |
| public member variables | `camelCase` |
| private/protected member variables | `mCamelCase` |
| private/protected static member variables | `sCamelCase` |
| global - static variable at file or project scope | `gCamelCase` |
| local variables | `camelCase` |
| macros | `UPPER_SNAKE_CASE` |

In the past a `p` prefix was used for pointers. This applies to all kinds of variables, e.g. `mpSomePtr`, `spSomePointer` etc.
The prefix `p` should not be used for self-contained new code, but might still be used to conform to old code.

### Variables

- Names should in general be descriptive of what a variable/parameter/etc. represents.
- The shorter the scope of a variable, the less important it is for it to have a verbose and descriptive name. Single-letter names are okay for extremely short-lived variables and loop counters, so long as their role is clear in context.

### Name shadowing

- The compiler is configured to error on name shadowing to prevent unexpected behavior and make name replacement more robust.
- It's good practice to introduce local scopes for reusing the same variable name:

```c++
void foo()
{
    {
        Program::Desc desc;
        ...
        mProgramA = Program::create(desc);
    }

    {
        Program::Desc desc;
        ...
        mProgramB = Program::create(desc);
    }
}
```

- If function argument names are shadowing struct members, use a trailing underscore (`_`) to avoid shadowing:

```c++
struct Data
{
    int foo;
    int bar;
    void set(int foo_, int bar_)
    {
        foo = foo_;
        bar = bar_;
    }
}
```

### Macros

- The use of macros should be minimized.
- Macros in header files should be prefixed with `FALCOR_` to avoid collisions.

## Language

- Use `using` declaration instead of `typedef` (e.g. `using IndexVector = std::vector<uint32_t>;`).
- Use sized types such as `int32_t`, `uint32_t`, `int16_t`. Conceptually, `bool` has unknown size, so no size equivalent. `char` is special and can be used only for C strings (use `int8_t` otherwise).
    - Avoid using `std::vector<bool>`. It behaves differently from all other `std::vector` types, and only should be used when the memory pressure makes it absolutely necessary.
- Use `nullptr` instead of `NULL` or `0`.
- Use `enum class` instead of `enum`.

### Rules for C++ classes/interfaces

- Classes with shared ownership:
    - Are generally expected to live on heap and owned via a `std::shared_ptr<MyClass>`.
    - Should be used sparingly and only in justified cases, as shared ownership often has other implied assumptions on lifetimes of either objects or data in those objects.
    - If classes define an alias for the `std::shared_ptr`, it shall be in the form:
        - `using MyClassSharedPtr = std::shared_ptr<MyClass>;` outside the class.
            - This is preferred as it can be easily forward declared
        - `using SharedPtr = std::shared_ptr<MyClass>;` inside the class.
            - This is only permitted for compatibility with older code that assumes the in-class definition.
        - When in doubt whether a header should use the `SharedPtr` alias and bring in the full `MyClass.h` header, or it should use `std::shared_ptr<MyClass>` instead, always do the latter.
    - Other types of smart pointers use aliases `ConstSharedPtr = std::shared_ptr<const MyClass>` and `WeakPtr = std::weak_ptr` follows the same patterns as `SharedPtr`
    - These aliases must be only for `std` versions of smart pointers, it is not permissible to use `SharedPtr` for `boost::shared_ptr` or any other non-standard smart pointer.
    - Are recommended to expose convenience `SharedPtr create(...);` factory functions.
        - The former ban on public constructors has been lifted, to allow using `std::make_shared`.
- Passing arguments:
    - Use a (const) reference or (const) raw-pointer if the argument is used only during the function's lifetime (i.e. the function doesn't store it in a member variable for future use).
    - Passing shared pointers to function that can take ownership can be done by either by value `SomeObject::SharedPtr` or by const reference `const SomeObject::SharedPtr&`.
        - The first is preferred if the function takes ownership, as it allows efficient use of `std::move` (which should be used to take the actual ownership of the shared pointer)
        - The latter is preferred if the function takes ownership only conditionally, as it avoid incrementing reference count in case the ownership is not taken.
- Prefer using `std::shared_ptr` and `std::unique_ptr` over raw pointers to denote ownership. Raw pointers should generally only be used either in classes that explicitly implement some form of smart pointer (e.g., pointer aliased with data), or when the code cannot reasonably use references (e.g., the class needs to be copied, or the underlying APIs require pointers).
    - Use `std::unique_ptr` unless the ownership of an object is truly shared, it is the more efficient smart pointer.
- Use the `override` specifier on all overridden virtual methods. It helps catch bugs.
- Avoid using `friend` unless absolutely needed.
- Use `static constexpr` for class/struct scoped constant values. Previous to C++17, `static const` members have to be initialized out-of-class, with C++17 they can be initialized inline using either `inline static const` or `static constexpr` (which is implicitly `inline`). C++ has an exception to allow `static const` integral types to be initialized in the declaration, but this will result in no global symbol for the constant (unless also defined out-of-class), which in turn leads to hard to understand behavior when referencing such a constant in a different library for example. Some more information can be found here: https://fekir.info/post/linker-error-with-static-member-variable

### Rules for C++ structs

- Structs are primarily used as data containers without member functions.
- All fields must be public.
- Variable initializers are allowed.

### Rules for Slang code

- The formatting and naming guidelines are identical to C++.
- Structs are used as data and code containers (there are no classes).
- Prefer member functions over global functions.
- Prefer enums instead of using macros or global constants.
- All entry points for a ray tracing or graphics program should be defined in the same file.
- Use the `1.f, 1.5f` format for floating point numbers.
    - Slang accepts both `1.f` and `1.0` as float, but some slang files are actually included in C++ files, where the latter becomes a double.
    - Using the form that's common between Slang and C++ avoids any possible hidden performance issues.
    - Slang uses `1.l, 1.L` for double. In C++, `1.l` (lowercase L) is `long double`, so using lowercase L for double literals is also safe.
    - Please beware of the performance implications of using doubles.

## Code Formatting

- You should set your Editor/IDE to follow the formatting guidelines.
- This repository uses `.editorconfig`, take advantage of it.

### Indentation

- Use **4 spaces**, no tabs.
- All braces `{` `}` are on a new line (except empty `{}` blocks which can be put on a single line).
- Indent next line after all braces `{` `}`.
- Move code after braces `{` `}` to the next line.
- Always indent the next line of any condition statement line:

```c++
if (box.isEmpty())
    return;
```
- `if`/`while`/`for` with a single statement do not have to use braces, if they are highly unlikely to be expanded to multiple statements.
```c++
for (size_t i = 0; i < count; ++i)
{
    if (distance(sphere, points[i]) > sphere.radius)
    {
        return false; //< OK, prefer when odds are the block will be expanded
    }

    if (std::isnan(distance(sphere, points[i])))
        return false; //< Also OK, prefer when the block is a trivial statement, e.g., error checking.
}
```
- It is OK to have single line getters and setters with trivial (single statement) bodies.

```c++
int getValue() const { return mValue; }
```

### Spacing

- Function call spacing:
    - No space before bracket.
    - No space just inside brackets.
    - One space after each comma separating parameters.

```c++
serializer->writeFloat("range", range);
```

- Conditional statement spacing:
    - One space after conditional keywords.
    - No space just inside the brackets.
    - One space separating commas, colons and condition comparison operators.

```c++
if (light.getIntensity() > 0.f)
{
    switch (light.getType())
    {
        case Light::Type::Directional:
            ...
```

- Align indentation space for parameters when wrapping lines to match the initial bracket:

```c++
Matrix::Matrix(float m11, float m12, float m13, float m14,
               float m21, float m22, float m23, float m24,
               float m31, float m32, float m33, float m34,
               float m41, float m42, float m43, float m44)
```

```c++
return sqrt((point.x - sphere.center.x) * (point.x - sphere.center.x) +
            (point.y - sphere.center.y) * (point.y - sphere.center.x) +
            (point.z - sphere.center.z) * (point.z - sphere.center.x));
```

### Comments

- Commenting style has been changed for better utilization of available tools.
    - Some of the existing code still uses the previous guidelines, refer to "Old style" section below.
- First reason is that the auto-reflow of comments in clang-format (which is needed to avoid awkward breaks in the actual code) cannot handle the current style, and requiring manual touchups after automated reformat is just waste of human time.
- Second reason is that both Visual Studio and Visual Studio Code have mechanisms to generate doxygen-style function comments, but the style is different from the old comment style. And again, given the choice between automating the work and keeping the current standard, we opt to save everyone's time and just change the standard.

#### New style
- Use the following syntax for adding comments to functions:

```c++
/**
 * @brief Colorize a string for writing to a terminal.
 * Detailed description. Can be an empty line.
 * @tparam TParam1 First template argument
 * @tparam TParam2 Second template argument
 * @param Param1 First normal argument
 * @param Param2 Second normal argument
 * @return Only when non-void
 */
```

- The `@brief` is optional. Doxygen will automatically end the brief section when it encounters `.`. As such, it is ok to use `.` even when the brief isn't a well formed sentence.
- The parameters can have `@param[in]` and `@param[out]`, but it is not mandatory and not necessary when the direction is obvious, e.g., parameters passed by value or const references and pointers.

- You can use `///` if the comment is short, i.e., if most of the lines of the comment were taken by the `/**` and `*/`
    - Examples are variable comments that do not fit after the declaration (`///<` is still the preferred way).
    - Short enum descriptions.
- Trivial functions that are self-commenting (e.g., getters and setters) do not require comments.
- Use `//` for inline comments:
```c++
// Find next empty slot.
auto it = find_if(list.begin(), list.end(), [] (const Slot& slot) { return slot.isEmpty(); });
```

- Use `///<` for commenting struct fields:

```c++
struct Info
{
    float distance; ///< Distance to nearest point.
    uint32_t index; ///< Index of nearest point.
};
```

#### Old style
- This style is now obsolete for new files and libraries.
- This style shall still be used when doing small touch-ups in files that already use this style, to maintain a consistent look.
- Use the following syntax for adding comments to functions:

```c++
/** Colorize a string for writing to a terminal.
    \param[in] str String to colorize
    \param[in] color Color
    \return Returns string wrapped in color codes.
*/
```

- Use `\param[in]`, `\param[out]` and `\param[in,out]` to specify how the function uses the arguments.
- Trivial functions that are self-commenting (e.g., getters and setters) do not require comments.
- Use `//` for inline comments:

```c++
// Find next empty slot.
auto it = find_if(list.begin(), list.end(), [] (const Slot& slot) { return slot.isEmpty(); });
```

- Use `///<` for commenting struct fields:

```c++
struct Info
{
    float distance; ///< Distance to nearest point.
    uint32_t index; ///< Index of nearest point.
};
```

- Avoid block comments `/* */` inside implementation code (`.cpp`).

### Logging

- Use single quotes for quoting names in log output:

```c++
logWarning("'{}' is not a valid name.", name);
```
