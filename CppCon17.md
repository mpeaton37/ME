 

 Curiously Recurring C++ Bugs at Facebook](https://www.youtube.com/watch?v=3MB2iiCkGxg)
 [Louis Brandy](https://github.com/lbrandy)
 - ASAN
  - 1) [] "-fsanitize=address"
  - 2) std::map::operator[] : creates entry on missing key...
  - 3) "-fsanitize-address-use-after-scope"
  - 4) volatile is bad.... started using std::atomics for threads
  - 5) Are shared pointers thread safe ? ...
  - 6) std::string(foo) compiles as it is a declaration. unique_lock<mutex> g(m_mutex)  lock guard must be used... ?
  - 7) RAII types + default constructors -> DANGER .std::unique_lock, std::lock_guard  ( -Wshadow-compatible-local )
  - 
