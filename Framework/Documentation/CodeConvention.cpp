/***************************************************************************
Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.
***************************************************************************/
/*
Every file must start with the legal header

Curly braces:
  * Are always required, even for a single-line statement
  * Start on their own line (except for simple setters/getters defined inside the class definition)

- 'using namespace' is only allowed inside CPP/C files. It's disallowed in headers (except for very very rare cases).
- 'using namespace std' is disallowed even in CPP/C files. If you want to save some work, just typedef the type you need from the std namespace, or use 'auto'.

- For consistency reasons only - use 'using' declaration instead of 'typedef'
        using UintVector = std::vector<uint32_t>;

Use only sized types (int32_t, uint32_t, int16_t). Conceptually, bool has unknown size, so no size equivalent.
char is special and can be used only for C strings (use int8_t otherwise)
Don't use NULL or 0 to initialize pointers. 'nullptr' is part of the language now.

Tab is 4 spaces. Enable 'insert spaces instead of tabs'.

Function names should be descriptive
  * Function that perform an action should be named after the action it performs - Fbo::clear(), createTextureFromFile().
  * Getters/Setters should start with 'get' and 'set'
  * Functions names that return a bool should be phrased as a question - isWhite(), doesFileExist(), hasTexture()
*/

/////////////////////////////////   Variable prefixes ////////////////////////////////////
/*
We are using the following prefixes for variable names based on the scope 's', 'm', 'g' and 'k'. These are mutually exclusive prefixes, where 'k' takes precedence above the rest.
    'k' is used for compile-time const variables.
    'g' is used for global variables, including static global variables.
    's' is used for class static variables.
    'm' is used for member variables in classes (not in structs).
In addition 'p' is used for pointers.
*/

//Global Variables:
const uint32_t kConstGlobal;    // compile-time-const, so 'k' takes precedence
int32_t gSomeGlobal;            // global start with 'g'
static int gStaticGlobal;       // Static globals start with 'g'
void* gpSomePointer;            // Global variables which is a pointer is prefixed with 'gp'
const void* gpPointer2;         // Not compile-time constant.

// Use struct only as a data-container. All fields must be public. No member functions are allowed. In-class initialization is allowed.
struct UpperCamelStruct
{
    int32_t someVar;      // Struct members are lower-camel-case
    uint32_t* pVar;      // Pointer start with 'p'. Note that the '*' belongs to the 'int' and that with a pointer variable, “p” counts as the first word, so the next letter *is* capitalized
    int32_t** pDoublePointer;       // Double pointer is just a pointer
    smart_ptr<int> pSmartPtr;   // Smart pointer is a pointer
    char charArray[];           // Array is not a pointer
    std::string myString;       // String is a string
    bool isBoolean;              // bool name implies that it's a bool. 'enable', 'is*', 'has*', etc. Don't use negative meaning (use 'enable' instead of 'disable')
    uint32_t& refVal;           // Reference is not a pointer
};

#define SOME_DEFINE                 // Definitions without values are upper case, separated by a '_'
#define SOME_VALUE_DEFINITION (1)   // Value definitions are upper case, separated by a '_'
#define this_should_do_something(a_) {a} // definitions which accept arguments are lower case, separated by a '_'. To avoid collisions, use '_' as the argument suffix.


// Classes should hide their internal data as much as possible. Prefer to use in-class initialization instead of C'tor
class UpperCamelClass
{
public:
    bool isValid();  // Function names are lower-camel-case
    static uint32_t sInt;   // Static variables start with 's'
    static const uint32_t kValue;   // Const static is prefixed with 'k'

private:
    int32_t mMemberVar;           // Member variables start with 'm'
    int16_t* mpSomePointer;       // Note that with a pointer variable, “p” counts as the first word, so the next letter *is* capitalized
};

enum class SomeEnum     // enums are always strongly typed ('enum class')
{
    ValueOne,     // Enum values are upper-camel-case, without a prefix
    ValueTwo,
    ValueThree,
};

void someFunction() // Function names are lower-camel-case
{

}

int main()
{
    UpperCamelClass someClass; // Local variables are lower-camel-case
    int32_t newVariable;        // Declare a single variable per line
    int32_t anotherVariable;    // Declare a single variable per line

    if(someClass.memberFunction())
    {               // Curly braces start on a new line
        return 0;   // and are used even for a single line statement
    }

    // And put some new-lines and comments to explain your code, even if you believe it's straightforward
    return 1;
}

//End with newline, as required by C++ standard
