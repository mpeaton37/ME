# 08/28/2017
Lord Huron rules...
alsa

# 11/15/2017
## C++17 studies
[ update to c++notes](https://docs.google.com/document/d/1ZkNYxt-suCA4CFmzgXMdQojluI8TNiaJNhgTQcdqcOc/edit#)

## JUCE
[ ADC017 videos ](https://www.youtube.com/channel/UCaF6fKdDrSmPDmiZcl9KLnQ/videos)
iCloud says it's full... it isn't.  Time suck

# 11/16/2017
## ML review
[ scikit-learn ](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier)

## Enthought webinar
- enthought sells software and consulting
- virtual core sampling?
- Enterprise software services
- Training, coorporate onsite classes.
### ML for:
- forecast sales, failure, 
- classification etc.
- scikit-learn is composible
- sklearn.pipeline?
[spectral.py](https://github.com/scikit-learn/scikit-learn/blob/f3320a6f/sklearn/cluster/spectral.py#L273)
[ spectral clustering ](http://web.cse.ohio-state.edu/~belkin.8/papers/SC_AOS_07.pdf)
- second eigenvalue of the graph Laplacian  [sparsest cut](https://en.wikipedia.org/wiki/Cut_(graph_theory)#Sparsest_cut)


# 11/20/2017
# How to integrate Python and C++?
## Conda Boost Python doesn't seem to have the examples on 
[ boost python ](http://www.boost.org/doc/libs/1_65_1/libs/python/doc/html/tutorial/tutorial/hello.html)
##### Where are channels set?
[conda config ](https://conda.io/docs/commands/conda-config.html) 
'''bash conda config --show'''
#### IRC.. mumble mumble mumble
[chat.freenode.irc/#Python] (https://webchat.freenode.net)
#### Download boost 

# 11/21/2017
# Machine Learning
[ Hands on ML ](http://techbus.safaribooksonline.com/book/programming/9781491962282/firstchapter#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODE0OTE5NjIyODIlMkZpZG0xNDA1ODMwMjM0OTIzNjhfaHRtbCZxdWVyeT0=)
- The amount of regularization to apply during learning can be controlled by a hyperparameter. A hyperparameter is a parameter of a learning algorithm (not of the model).
- Recall / Precision tradeoff
- ROC := Sensitivity , 1 - Specificity
- Early Stopping Geoffry Hinton
# Audio Recording
[ Alan Parsons Video ]( )
- Vocal Comping
- Aritha Franklin tape distortion

# 11/25/2017

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

# 11/26/2017
[ Stone Aerospace ](http://stoneaerospace.com)

# 11/29/2017
### Continued review of JUCE 2017 (notes appended to 11/25/2017)
[Real-time IIR filter design introduction](http://www.eas.uccs.edu/~mwickert/ece5655/lecture_notes/ARM/ece5655_chap7.pdf)

# 11/30/2017
###Stephen Plaza
- [ Image segmentation in Spark](https://arxiv.org/pdf/1604.00385.pdf)
  +  Boundary Prediction -> Watershed -> Agglomation
    * ? Optimality of Lumped operations
    *  

### More JUCE

