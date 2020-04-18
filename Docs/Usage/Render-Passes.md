### [Index](../index.md) | [Usage](./index.md) | Render Passes

--------

Note: This page will cover more complex elements of render passes. Basic elements are covered in the tutorial [here](../Tutorials/03-Implementing-a-Render-Pass.md).

## Exporting as a DLL
Falcor will look for a function called `getPasses()` in your DLL. The signature is as follows:
```c++
extern "C" __declspec(dllexport) void getPasses(RenderPassLibrary& lib);
```
This function should register all the passes that the DLL exports:
```c++
extern "C" __declspec(dllexport) void getPasses(RenderPassLibrary& lib)
{
    lib.registerClass("MyBlitPass", "My Blit Class", MyBlitPass::create);
}
```
Generally, `getPasses()` will reside in the same source file as the render pass it exports. For DLLs that export multiple passes, this function as well as `getProjDir()` should be located in a separate source file that shares a name with the project (e.g. a DLL named `Antialiasing` that contains some number of render passes should contain these functions within the source file `Antialiasing.cpp`).

If using scripting, make sure to include the appropriate DLLs through calls to `loadRenderPassLibrary()` at the beginning of your graph script. These are generally DLLs that do not share a name with the render passes that they contain; all others will be found automatically.

## Render Pass Resource Allocation and Lifetime
By default, every output and temporary resource is allocated by the render-graph. Input resources are never allocated specifically for the pass. 
Input resources must be satisfied by either an edge in the graph or an resource bound directly to the graph by the user.

A resource can be marked with the `Field::Flags::Optional` bit. This flag only makes sense for input and output resources. Using this field on an internal resource is a good way to ensure it will not be allocated.
For input resources, this flag means that the pass can execute even if this input is not satisfied.
If this flag is set on an output resource, it will only be allocated if it is required by a graph edge.

Using the `Field::Flags::Persistent` bit on a resource tells to graph system that the resource needs to retain it's data between calls to `RenderPass::execute()`. This effectively disables all resource-allocation optimizations the render-graph performs for the current resource.
* *Note that this flag doesn't ensure persistence across graph re-compilation. Re-compilation will most certainly reset the resources.*

As a final note, you should not cache resources inside your pass. This will interfere with the render-graph allocator and will probably result in rendering errors.

## Passing Data Between Passes

### Render Data

Each render pass specifies which input/outputs it needs. These are then available in the pass' `execute()` function via `pRenderData->getTexture()`.

### Other Data

Other data than render data can be passed between passes via a `Dictionary` accessible in `execute()` via `pRenderData->getDictionary()`.

The `Dictionary` is a shared resource, that all passes can read/write to by. The data stored in it is persistent between frames.

Pass A:
```
float x = 5.0f
Dictionary& dict = pRenderData->getDictionary();
dict["foo"] = x;    // stores float value for key 'foo'
```

Pass B:
```
Dictionary& dict = pRenderData->getDictionary();
float x = dict["foo"];    // load float value from key 'foo'
```

If a key doesn't exist when trying to load it, an error is logged. The function `keyExists()` can be used to check if a key exists before loading it.

Casting a value in the dictionary to another type upon loading it fails in some cases. This is also an error. For such types, load into a temporary variable before casting in a separate step:
```
float tmp = dict["foo"];
mytype x = (mytype)tmp;
```

## Serializing Passes
### Loading a pass

The `create()` method you are required to provide accepts a `Dictionary` object. This is essentially a map, where the key is a string and the value can be any object.

The render-graph importer will parse and pass that dictionary into the `create()` of the render-pass.

The pass is responsible for initializing its members based on key/value pairs found in the `Dictionary`.

### Saving a pass

The render-graph exporter will call the virtual `getScriptingDictionary()`. Use this function to serialize your pass, like so:

```c++
const char* kFilter = "filter"
Dictionary MyBlitPass::getScriptingDictionary() const
{
    Dictionary dict;
    dict[kFilter] = mFilter;
    return dict;
}
```

You will probably need to make some changes to the script bindings, even in cases where you did not declare new classes or enums. See below.

## Script Bindings
In order to use your pass with Python scripting, you will need to register it as well as any associated classes, functions, enums, and properties with PyBind11. This is done like so:
```c++
static void regExampleClass(ScriptBindings::Module& m)
{
    // register classes, properties, and enums here
}

extern "C" __declspec(dllexport) void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerClass("ExampleClass", "Example Class", ExampleClass::create);
    ScriptBindings::registerBinding(regExampleClass);
}
```

### Registering Classes
If you have classes you wish to register with the scripting module:
```c++
static void regExampleClass(ScriptBindings::Module& m)
{
    auto e = m.regClass(ExampleClass);
}
```
Note that you do not need to register your render pass unless you plan to also register properties, functions, or enums associated with your pass. 

### Registering Pass Properties
If you would like to access certain pass-specific properties from Python scripting:
```c++
static void regExampleClass(ScriptBindings::Module& m)
{
    auto c = m.regClass(SomeClass);
    c.property("property", &SomeClass::getProperty, &SomeClass::setProperty);
    c.roProperty("readOnlyProperty", &SomeClass::getReadOnlyProperty);
}
```
This allows you to access the bound properties of your pass the same way you would access properties for any given instance of a Python class.

### Registering Class Methods
If you have class-specific methods you wish to bind to Python:
```c++
static void regExampleClass(ScriptBindings::Module& m)
{
    auto c = m.regClass(SomeClass);
    c.func_("foo", &SomeClass::foo); // simple function registration
    c.func_("bar", &SomeClass::bar, "someArg"_a = 1); // function with a named/default argument
    c.func_("baz", ScriptBindings::overload_cast<uint32_t>(&SomeClass::baz));
    c.func_("baz", ScriptBindings::overload_cast<>(&SomeClass::baz); // overloaded functions
}
```
Some things to note:
- The `""_a` operator takes the provided string and converts it to a `pybind11::arg` with that name.
- `ScriptBindings::overload_cast<>()` takes as template parameters the types of the arguments to the overloaded function being bound. For example, the first binding for `SomeClass::baz` binds the overload that accepts a `uint32_t`, whereas the second binding binds the overload that takes no arguments.

### Registering Enums
If you have enums you wish to register:
```c++
static void regExampleClass(ScriptBindings::Module& m)
{
    auto enumClass = m.enum_<SomeClass::EnumClass>("SomeClassEnumClass");
    enumClass.regEnumVal(SomeClass::EnumClass::Foo);
}
```
The enum values need to be convertible to string format by a `to_string` function. Add that to the header where the enum is declared, if it does not already exist. For example:
```c++
#define enum_str(a) case SomeClass::EnumClass::a: return #a
    inline std::string to_string(SomeClass::EnumClass e)
    {
        switch (e)
        {
            enum_str(Foo);
        default:
            should_not_get_here();
            return "";
        }
    }
#undef enum_str
```
Note that enum values cannot be reserved keywords in Python. Names like `None`, `True`, `False` etc. are therefore not allowed. If your enum has values with those names, give them other names in the `to_string` function.

When registering the enum, include the class name as part of the enum name. This is a workaround for the fact that Python doesn't include the class name, and we risk getting name clashes if an enum type with the same name exists in different classes. 

The convention is that the C++ enum `MyClass::Type` should be registered as `MyClassType`. For example:

`auto e = m.enum_<MyClass::Type>("MyClassType");`

**Note:** 
You might be tempted to convert enums to to and from `uint`. Don't do that. *Passes that use this will be considered to have an incomplete implementation and will not be merged into Falcor*. This may result in a backward compatibility issue. Consider this a fast and dirty solution and only use it during pass bringup.

For more information on Python bindings check the [PyBind11 documentation](https://pybind11.readthedocs.io/en/stable/)

