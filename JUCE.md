# JUCE'17
### Julian Storer
- [Does your code actually matter](https://youtu.be/Yd0Ef6uzJb0?t=25m2s)
- [ Smule on me ](https://youtu.be/MQWKfs-qP7o?t=13m22s)
- [ MiMi and Elaine Chew ](https://www.amazon.com/Mathematical-Computational-Modeling-Tonality-International/dp/B00Z8EUIOG) 
- MusArt, Morpheus , Tristan chord, 

### Phoenix Perry
- [ David Kanaga ](http://www.davidkanaga.com/)
- BOTS

### Jon Latane  (JUCEish)
- [ topological ](https://topologica.co/)

### Andreas Gustafsson
- Space-Time transforms
  - [ Velasco et. al. Constructing Constant Q](https://www.univie.ac.at/nonstatgab/pdf_files/dohogve11_amsart.pdf)
  - [ Holighaus N et. al. "A Frameowrk for invertible..."](http://www.univie.ac.at/nonstatgab/pdf_files/dogrhove12_amsart.pdf)
  - *Gaborator library*

### [Angus Hewlett ](https://www.youtube.com/watch?v=cn-5k8fm_u0)
- [FXpansion](https://www.fxpansion.com/)
- [ SIMD library ](https://github.com/angushewlett/simd2voice)
- SIMD vector classes and branchless algorithms for audio
  - {SLP: [System, heterogeneous], TLP: [Thread, host/OS], ILP: [Instruction, 1ns], DLP: [Data, SIMD]}
  - Instruction Latency <- [ data dependencies, cache miss,etc.]
  - Optimal Interleaving... 
- Branchless conditional masks
  - result = _mm_and_ps(mask, [1,0,1.2,2.0,1.5])
  - naive oscillator using conditional masks

```cpp 
	phase += increment;
	phase -= ((phase >= 1.f) & 2.f);
	phase  = _mm_add_pas)phase,increment);
	mask   = _mm_cmp_gte_ps(phase,1.f);
	step   = _mm_and_ps(2.f,mask);
	phase  = _mm_sub_ps(phase,step);
```
[polyblep](https://pdfs.semanticscholar.org/3871/2d0f05e1904ec8d8eed0a5c872a4146ccf60.pdf)
 
How efficient is your code?

### Anastasia Kazakova (JetBrains)
```cpp 
//what does this evaluate to?
#ifdef MAGIC
template<int> 
struct x {
  x(int i) {}
}
#else
int x = 100;
#endif

void test(int y) {
  const int a = 100;

// expression or template?
  auto foo = x<a>(0);
}
```
 
#### Static Analysis Tools

 [CppCheck](https://sourceforge.net/p/cppcheck/wiki/ListOfChecks/)
 [Clang-analyzer](https://clang-analyzer.llvm.org/available_checks.html)
 [Clang-tidy](http://clang.llvm.org/extra/clang-tidy)
 [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines)

#### Dynamic Analysis Tools

[Valgrind = [ Memcheck, Cachegrind, Callgrind, Hellgrind, Massif ]](http://valgrind.org)
```python
{ 
Valgrind :[JIT/VM, any compiler, one thread AAT, 20-50 slow down],
 Sanitizers:[ Clang(3.1-3.2) GCC(4.8), require recompilation, 2-5 slow down]
}
```

#### Refactoring
[Clang-Rename]()
[LegalizeAdulthood](https://github.com/LegalizeAdulthood/refactor-test-suite)

#### Unit Testing
```Python 
  { Google Test:45, Boost.Test:26,CppUnit:11, CppUTest:5, Catch:5   }
```

#### Package Manager
[Blizzard](https://github.com/Blizzard/clang)
[Blizzard](https://github.com/berkus/clang-packaging-P0235R0)
[Hunter](https://github.com/ruslo/hunter)
[Conan](https://github.com/conan-io/conan)

##### References
[Timur Doumler, Readable Modern C++ Russia 2017](https://youtube.com/watch?v=6AoifPEOAXM)

### Andre Bergner
[Andre Bergner github](https://github.com/andre-bergner)
##### [Nonlinear oscillators](https://youtu.be/8G0EVWcysng)
[Stuart-Landau Equation](https://community.dur.ac.uk/suzanne.fielding/teaching/HST/sec8.pdf)
[Hopf Bifurcation](https://en.wikipedia.org/wiki/Hopf_bifurcation) [Arnold's tongue](https://en.wikipedia.org/wiki/Arnold_tongue)
[Duffing oscillator](https://en.wikipedia.org/wiki/Duffing_equation)
[Synchronization](https://www.amazon.com/dp/052153352X/_encoding=UTF8?coliid=I3F61YVR8LM95Y&colid=1Q7QLLP4GW2C0&psc=0)

### Dave Ramirez
[Developing Audio applications with JavaScript](https://youtu.be/WxRWgafjXSA)
[github](https://www.github.com/ramirezd42)

### Dave Rowland
[Using JUCE value trees and modern C++ to build large scale applicaitons](https://youtu.be/3IaMjH5lBEY)
[github](https://github.com/drowaudio/drowaudio)

### David Zicarelli
[Code generating Littlefoot](https://youtu.be/u0cmFmCT66A)
- "The longer I an stay in the problem space, the better"
- Real time code generation with printf for ROLI 

### Devendra Parakh
[Techniques for debugging realtime audio issues](https://youtu.be/MfWgNUsEleo)
no IO -> log to buffers 
[Firelog](https://developer.apple.com/documentation/iokit/iofirewiredeviceinterface/1555664-firelog?language=objc)
[tutorial](https://www.juce.com/doc/tutorial_audio_parameter)
Dry,Wet,Dry - e^i\pi * Wet := Audio Cancellation,  Look for artifacts

### Friedeman Schautz
[The Development of Ableton Live](https://youtu.be/_mnpp-Wuk3A)
Challenges: evolving UI, Language evolution, testing (coverage), CI complexity
Push Interface is Python

### Martin Shuppius
[Physical modelling of guitar strings](https://youtu.be/sxt5rxF_PdI)
Martin.Schuppius@gmail.com

#### References
- [S. Bilbao "Numerical Sound Synthesis" Wiley & Sons Ltd, 2009](http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470510463.html)
- [Julius O. Smith "Physical Audio Signal Processing"](https://ccrma.standofrd.edu/~/jos/pasp/) (Sept. 15, 2017)
- [H. Mansour "The bowed string and its playability: Theory, simulatino and analysis" PhD thesis, Dept. of Music Rsearch McGill University, Montreialm Quebec, Cancada, 2016.](http://www.music.mcgill.ca/caml/lib/exe/fetch.php?media=publications:phd_mansour_2016.pdf)
- [J. Woodhouse, "On the synthesis of guiltar plucks", Acta Acustica united with Acustica, vol. 90, pp. 928-944, 2004.](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=52A9E17345C828A1D403451496C49830?doi=10.1.1.701.1411&rep=rep1&type=pdf)


### Glenn Kasten
[Github](https://github.com/gkasten)
[Statistical and ML analysis of real time audio performance logs](https://youtu.be/3DK20e1-hzU)

#### Latency in Android audio loop, reduced latency -> dropouts,   
What to log:

- Task wakeup times
- Task execution times
- Available CPU bandwidth
- CPU core ID
- other...

#### Causes of lag:
- Fixed constant hardware cost
- USB
- interupt disabling by wifi, fingerprint reader etc.
- CPU clock frequency variation
- application related processing consistancy, SIMD, branchless etc.

### Sanna Wager
[homepage](http://homes.soic.indiana.edu/scwager/)

- Analysis of latency and glitches in app, airplane mode etc.
Fischer's exact Test, Welch's 2-sample test, Correspondence analysis, 
- ML based prediction using: device temperature, 
-References
  - [Estimation and inference of heterogeneous treatment effects](https://arxiv.org/pdf/1510.04342.pdf)
    - At a high level, trees and forests can be thought of as nearest neighbor methods with an adaptive neighborhood metric. Distance metric is leaf proximity.
  

[Stefan Wager](https://web.stanford.edu/~swager/research.html)



Instrumentation requirements:

- Lightweight, low overhead
- lock free , non-blocking algorithms
- Precise time stamps
- circular buffers.


#### Other techniques
lldb, gdb, ntsd, windbg
Mac OS[Dtrace](https://www.bignerdranch.com/blog/hooked-on-dtrace-part-1/)
Windows[ETW](https://msdn.microsoft.com/en-us/library/windows/desktop/bb968803(v=vs.85).aspx)

### Nikolas Borrel
[Harmonisation in modern rhythmic music using Hidden Markov models](https://youtu.be/tc22n7j7ixY)
[Livetake](http://www.livetakeconcert.com/watch/plst/roskilde_playlist/12525FC9-0264-45C3-B906-9123F0AC2FF0)
[Livetake](https://www.crunchbase.com/organization/livetake)




