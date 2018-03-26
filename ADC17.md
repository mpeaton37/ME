# ADC'17
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
- [Space-Time transforms](https://youtu.be/ONJVJBmFiuE)
  - [ Velasco et. al. Constructing Constant Q](https://www.univie.ac.at/nonstatgab/pdf_files/dohogve11_amsart.pdf)
  - [ Holighaus N et. al. "A Frameowrk for invertible..."](http://www.univie.ac.at/nonstatgab/pdf_files/dogrhove12_amsart.pdf)
  - [Gaborator library](https://gaborator.com)

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
 
[Euterpea](http://www.euterpea.com)

### Stefan Stenzel [email](stefan@ioptigan)

[The amazing usefulness of band limited impulse trains, shown for oscillator banks](https://youtu.be/lpM4Tawq-XU)

- blip... blip... blip... beeeeeeep

### Tim Adnitt, Carl Bussey
[Making Coputer music creation accessible to a wider audience](https://youtu.be/hQs-5-s7h98)

Understand, Observe, Synthesis, Ideate, Prototype, Test

[Spatial audio at Facebook](https://youtu.be/tJXT7yXcMbA)

[Facebook 360](https://facebook360.fb.com)
[Ambisonics](https://en.wikipedia.org/wiki/Ambisonics)
[binaural rendering]()


[Spacialized audio]()  [headlocked audio]() - both useful creatively

### Varun Nail, Hans Fugal
[Spatial Audio at Facebook](https://youtu.be/tJXT7yXcMbA)

- Heavily SIMD optimized spatializer plugins for production of Ambiasonic sound
- 360 video player
- Timecode sent over OSC
- JUCE + OpenGL
- Facebook internal library for lock-free queue
- std::atomic<>
- Headlocked and Ambiasonic mix

### Yvan Grabit
[VST3 history, advantages, and best practices](https://youtu.be/0QBWXC8KNz0)

- ASIO, VST, 
- VST3 has note expressions


### Matthieru Brucher
[Modeling and optimizaing a distortion circuit](https://youtu.be/HHiMD_QGRo0)

- Simulation of simple overdrive circuit using Newton Raphson based integration of nonlinear ODE.
- [AudioTK](https://github.com/mbrucher/AudioTK)
- [2017-ADC](https://github.com/mbrucher/2017-ADC)

### Ivan Cohen
[Fifty shades of distortion](https://youtu.be/oIChUOV_0w4)
[Musical Entropy](http://musicalentropy.github.io/downloads/)
[ Homework ](https://forum.juce.com/t/adc17-fifty-shades-of-distortion-extra-files-homework-part-1/25173﻿)

- Static Wave-shaper
  + Not bounded
  + Rectifier and asymmetrical
  + Power series and Chebyshev
  + Fractions of functions, polynomials
  + random, ramp
  + customizable

- Dynamics Processors
  + VCA, Compressions
  
- Digital Effects
  +   Aliasing, Bitcrusher
  + Slew Limiter
  + Stuttering
  + FFT Waveshaper
  + Ring modulator
  + Crossover Distortion
  + Binary operations
  
```cpp
auto N=4;
auto cpt=0;
for (auto i=0;i<numSamples;i++)
{
  samples[i] = (cpt==0?samples[i]:0.f);
  cpt = (cpt+1)%N;
}
```
  
- Black box analog modeling
  + Impulse response measurement and convolution for linear systems
  + Black box vs White box modeling
  + Dynamic Convolution from [Sintefex](http://www.sintefex.com)
  + Volterra and Hammerstein series
  + Gray box approach: generic modeling + estimatino of parameters
  + Machine learning

[U-he plugin users prefered Newton Raphson](https://www.sweetwater.com/store/detail/Repro1?gclid=CjwKCAiAx57RBRBkEiwA8yZdUEwXofFk7coBmyAKIlO1QYVdoFDc6Avd9nbKPnUuVg1OS7Jic9SethoC2lsQAvD_BwE)

### Martin Finke
[Reactive Extensions (RX) in JUCE](https://youtu.be/iynYX82N3RY)

- Basically publish and subscribe at the variable level through Observer Pattern
- [Reax on github](http://github.com/martinfinke/ReaX)
- ? Value tree vs Reactive extensions
- ? Reactive collections ? -> copy only, no reactive insert apparently

### Jan Koenig 
[Introduction to cross platform voice applications for Amazon Alexa and Google Home](https://youtu.be/sMJkjknFgKg)

- Alexa Skill <-> Google Action
- Alexa skills have doubled in the last 5 months
- AWS Lambda insertion
- "jovo new ADC" "jovo run"

##### Opportunities for Audio Developers

- Smart Speaker Composition
- Voice-enable Audio
- Tools for Pre-recorded Audio
  + "Alexa, tell my Studio to play a faster beat"
    * Odis Audio engineering assitant
    * Amazon, Sir-Mix Alot cooperation

### Ian Hobson 
[The Use of std variant in Realtime DSP](https://youtu.be/m7xYX8f8A7Y)

- Means of reducing memory footprint  
- std::variant := 
  + New in C++17
  + C++ Standard Library implementaiton of Discriminated Union
    * Tagged Union, Sum Type, Enum (Rust,Swift)
    * Algebraic Data type (Haskell)
    * Encapsulates the tagged union pattern
    * compiler optimizations strip away the abstractions

- Variants (match) in Rust are a language feature rather than a library
- Detailed discussion of variant exceptions and potential undefined behaviors


### Giulio Moro
[Assessing the stability of audio code for real time low latency embedded processing](https://youtu.be/e1D5vCBWhdk)

Real-Time := Output in bound amount of time

- [Bella](https://github.com/BelaPlatform/Bela.git) [Bella.io](http://bela.io)
- [c2dm](http://c4dm.eecs.qmul.ac.uk)
- [instrumentslab](http://instrumentslab.org)
- [ How Fast is fast enough](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwib5pTOvvjXAhWJ8CYKHSz7D5oQFggpMAA&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F9eb5%2F1dbe38fb23034e80b8664d8281996d2a5ef6.pdf&usg=AOvVaw3kgdLmZwDCgTHeQkHXP3vg)
- delay in response to external action
- computational time jitter
- thread wake-up latency
- clock jitter ns
- real-time (non-blocking) callback, constant CPU load, minimize synchronization are all good programming practice.

- Xenomai (Primary ) mode has best RT performance
- RTLinux has worse average wake-up time, but much shorter tail.

- Infrequent block computations (FFT) may drive ceiling for glitch
- splitting over call back in seperate thread
- context switching
- [GPIO based clock ~= 150ns](https://en.wikipedia.org/wiki/General-purpose_input/output)
- Aim for constant CPU load
- ? threadsafe, pre-emptable FFT ?


### Don Turner, Phil Burk
- [Don Turner github](https://github.com/dturner)
- [Phil Burk github](https://github.com/philburk)
- [SoftSynth](http://www.softsynth.com) [Mobileer](http://www.mobileer.com) 
- [Build a synth for Android](https://youtu.be/Pjgfje52Yv0)
- [Oboe open source library](https://github.com/google/oboe)

```cpp
builder.setPerformanceMode(OBOE_PERFORMANCE_MODE_LOW_LATENCY);
```
- 150ms -> 65ms of latency

```cpp
stream->setBufferSizeInfFrames(stream->getFramesPerBurst());
```

- 65ms -> 41ms of latency

```cpp
builder.setSharingMOde(OBOE_SHARING_MODE_EXLUCIVE)
```

- Exclusive mode MMAP noIRQ mode in ALSA driver will lead to lowest possible 41 -> 35ms 
- Round trip mode on pixel phones (just audio) as low as:
  + 10ms double buffer
  + < 10ms for single buffer
  
- ? Garbage collection now more incremental in java less latency

### Christof Mathies
[Opening the box: Whitebox Testing of Audio Software](https://youtu.be/Kvfhu0WDUM4)

- [behance](https://www.behance.net/gallery/58556847/Sticks)
- [ Apple Instruments profiler](https://developer.apple.com/library/content/documentation/DeveloperTools/Conceptual/InstrumentsUserGuide/index.html)
  + Can be done on production code
- Unit testing for signal processing, s1-s2, S1-S2, statistics, MFCC 
- [Adobe Audition](http://www.adobe.com/products/audition.html)

### Brecht De Man
[Rethinking the music production workflow](https://youtu.be/yJcqNR7by9o)

- [web site](http://www.brechtdeman.com)
- mixing assistants
- multitrack plugins
- [semantic audio](http://www.semanticaudio.co.uk)
- ["Semantic description of timbral transformations in music production"](http://dmtlab.bcu.ac.uk/ryanstables/ACMMM2016.pdf)

### Ben Supper
[Present and future improvements to MIDI](https://youtu.be/8xzA9sehSXc)
- MIDI community hoping to standardize Protocols, Profiles, and Properties

### Geert Bevin, Amos Gaynes
[Designing and implementing embedded synthesizer UIs with JUCE](https://youtu.be/jq7zUKEcyzI)

 [ MOOG SUB Phatty ](https://youtu.be/c0B8ftVkUac)
 [ Geert Bevin ](http://gbevin.com)

 - Projucer
   +  Consistent grammer
   +  Navigational widgets
     *  Conceptually similar moves to minimize re-learning effort
     *  Clarify, Simplify, Chunk, organize
     *  Never make the user feel like they are building a ship in a bottle
     *  Graphical prototyping of UI hardware

- Automated Tests
  + Sematic UI tests are relatively easy in JUCE using component identifiers
  + public UnitTest
  + [gitlab](https://about.gitlab.com)









