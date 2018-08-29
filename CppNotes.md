[Old notes](https://docs.google.com/document/d/1ZkNYxt-suCA4CFmzgXMdQojluI8TNiaJNhgTQcdqcOc/edit#)
    

[CppCon17](file:///Users/mpeaton/ME/CppCon17.md)

[Bryce Adelstein Lelback](bryce@cppcon.org)
[An Introduction to C++17 via Inspiring examples](https://youtu.be/fI2xiUqqH3Q)
 

Herb Sutter cpp 2017
- String_view(
- [Spaceship operator paper](http://open-std.org/JTC1/SC22/WG21/docs/papers/2017/p0515r0.pdf)
- strcmp() in C
https://wg21.link

[Modern C++](http://techbus.safaribooksonline.com/book/programming/cplusplus/9781491908419/firstchapter#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODE0OTE5MDg0MTklMkZ0ZXJtaW5vbG9neV9hbmRfY29udmVudGlvbnNfaHRtbCZxdWVyeT0=)

- Copies of rvalues are generally move constructed, while copies of lvalues are usually copy constructed.

#### Item 1: Understand template type deduction

- During template type deduction, arguments that are references are treated as non-references, i.e., their reference-ness is ignored.
- When deducing types for universal reference parameters, lvalue arguments get special treatment.
- When deducing types for by-value parameters, const and/or volatile arguments are treated as non-const and non-volatile.
- During template type deduction, arguments that are array or function names decay to pointers, unless they’re used to initialize references.

#### Item 2: Understand auto type deduction

- auto type deduction is essentiall the same as template type deduction
- The treatment of braced initializers is the only way in which auto type deduction and template type deduction differ.
```cpp 
auto x1 = 27;
auto x2(27);
auto x3 = { 27 };
auto x4{ 27 };
typeid(x4).name()
```
- The second two declare a variable of type std::initializer_list<int> containing a single element with value 27! 
- auto assumes that a braced initializer represents a std::initializer_list, but template type deduction doesn’t.
- auto in a function return type or a lambda parameter implies template type deduction, not auto type deduction.

#### Item 3: Understand decltype

- decltype almost always yields the type of a variable or expression without any modifications.
- For lvalue expressions of type T other than names, decltype always reports a type of T&.
- C++14 supports decltype(auto), which, like auto, deduces a type from its initializer, but it performs the type deduction using the decltype rules

#### Item 4: Know how to view deduced types
- Deduced types can often be seen using IDE editors, compiler error messages, and the Boost TypeIndex library.
- The results of some tools may be neither helpful nor accurate, so an understanding of C++’s type deduction rules remains essential.

#### Item 5: Prefer auto to explicit type definitions

- auto variables must be initialized, are generally immune to type mismatches that can lead to portability or efficiency problems, can ease the process of refactoring, and typically require less typing than variables with explicitly specified types.
- auto-typed variables are subject to the pitfalls described in Items 2 and 6.

#### Item 6: Use the explicitly typed initializer idiom when auto deduces undesired types.

- “Invisible” proxy types can cause auto to deduce the “wrong” type for an initializing expression.
- The explicitly typed initializer idiom forces auto to deduce the type you want it to have.

#### Item 7: Distinguish between () and {} when creating objects.





### Multithreading
- Introduced in c++11
- function pointer, callable or lambda

```c++
// CPP program to demonstrate multithreading
// using three different callables.
#include <iostream>
#include <thread>
using namespace std;

// A dummy function
void foo(int Z)
{
    for (int i = 0; i < Z; i++) {
        cout << "Thread using function"
               " pointer as callable\n";
    }
}

// A callable object
class thread_obj {
public:
    void operator()(int x)
    {
        for (int i = 0; i < x; i++)
            cout << "Thread using function"
                  " object as  callable\n";
    }
};

int main()
{
    cout << "Threads 1 and 2 and 3 "
         "operating independently" << endl;

    // This thread is launched by using 
    // function pointer as callable
    thread th1(foo, 3);

    // This thread is launched by using
    // function object as callable
    thread th2(thread_obj(), 3);

    // Define a Lambda Expression
    auto f = [](int x) {
        for (int i = 0; i < x; i++)
            cout << "Thread using lambda"
             " expression as callable\n";
    };

    // This thread is launched by using 
    // lamda expression as callable
    thread th3(f, 3);

    // Wait for the threads to finish
    // Wait for thread t1 to finish
    th1.join();

    // Wait for thread t2 to finish
    th2.join();

    // Wait for thread t3 to finish
    th3.join();

    return 0;
}
```
