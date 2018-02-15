    ## 08/28/2017
Lord Huron rules...
alsa

## 11/15/2017
### C++17 studies
[ update to c++notes](https://docs.google.com/document/d/1ZkNYxt-suCA4CFmzgXMdQojluI8TNiaJNhgTQcdqcOc/edit#)

## JUCE
[ ADC017 videos ](https://www.youtube.com/channel/UCaF6fKdDrSmPDmiZcl9KLnQ/videos)

[ADC17 notes](ADC17.md)

# 11/16/2017
### ML review
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
### How to integrate Python and C++?
#### Conda Boost Python doesn't seem to have the examples on 
[ boost python ](http://www.boost.org/doc/libs/1_65_1/libs/python/doc/html/tutorial/tutorial/hello.html)
##### Where are channels set?
[conda config ](https://conda.io/docs/commands/conda-config.html) 
'''bash conda config --show'''
#### IRC.. mumble mumble mumble
[chat.freenode.irc/#Python] (https://webchat.freenode.net)
#### Download boost 

## 11/21/2017
#### Review of Machine Learning
[ Hands on ML ](http://techbus.safaribooksonline.com/book/programming/9781491962282/firstchapter#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODE0OTE5NjIyODIlMkZpZG0xNDA1ODMwMjM0OTIzNjhfaHRtbCZxdWVyeT0=)
- The amount of regularization to apply during learning can be controlled by a hyperparameter. A hyperparameter is a parameter of a learning algorithm (not of the model).
- Recall / Precision tradeoff
- ROC := Sensitivity , 1 - Specificity
- Early Stopping Geoffry Hinton
Review of Audio Recording
[ Alan Parsons Video ]( )
- Vocal Comping
- Aritha Franklin tape distortion

## 11/25/2017
#### Review of JUCE '17 Videos

## 11/26/2017
#### Watched some TED talks
[ Stone Aerospace ](http://stoneaerospace.com)

## 11/29/2017
- Continued review of JUCE 2017 (notes appended to 11/25/2017)
- [Real-time IIR filter design introduction](http://www.eas.uccs.edu/~mwickert/ece5655/lecture_notes/ARM/ece5655_chap7.pdf)

## 11/30/2017

### Stephen Plaza
- [ Image segmentation in Spark](https://arxiv.org/pdf/1604.00385.pdf)
  +  Boundary Prediction -> Watershed -> Agglomation
    * ? Optimality of Lumped operations
    *  

### More JUCE

## 12/5/2017

- [distortion modeling](https://ccrma.stanford.edu/~dtyeh/papers/DavidYehThesissinglesided.pdf)
- [NGSpice](http://ngspice.sourceforge.net/screens.html)
- [Pyspice](https://pyspice.fabrice-salvaire.fr/examples/diode/voltage-multiplier.html)
- [AudioTK](https://github.com/mbrucher/AudioTK.git)
- [CCRMA Stanford](https://ccrma.stanford.edu/papers)
- [Numeric Integration in Structural Dynamics](http://people.duke.edu/~hpgavin/cee541/NumericalIntegration.pdf)


## 12/7/2017

- More JUCE notes
- [teensy](https://www.pjrc.com/teensy/td_libs_Audio.html)

## 12/11/2017

### Scott Sievert
- [ NEXT: Crowdsourcing, machine learning and cartoons](https://youtu.be/blPjDYCvppY)
- [scipy-next](tinyurl.com/scipy-next)
- Adaptive sampling 
  + quicksort nlog(n) analogy
  + [New Yorker Cartoons](https://www.newyorker.com/cartoons/vote)

### Drew Fustin
- [Interrupted Time Series Experiments in Python](https://www.newyorker.com/cartoons/vote)
- [Tom Augspurger](https://tomaugspurger.github.io)
- Homogeneous Poisson Point Process
  + $$ P(N(t)=n)=\frac{(\lambda t)^n }{n!} \exp^(-\lambda t) $$
  + $$ E[N(t)] = \lambda t $$
- Nonhomogenous allows lambda to vary in time
  + One approach: generate max rate and integrate define thinning function
  +  Seasonal ARIMA

## 12/12/2017

- [Deep Learning](https://docs.google.com/presentation/d/e/2PACX-1vQMZsWfjjLLz_wi8iaMxHKawuTkdqeA3Gw00wy5dBHLhAkuLEvhB7k-4LcO5RQEVFzZXfS6ByABaRr4/pub?start=false&loop=false&delayms=60000&slide=id.g2a19ddb012_0_490)
- ["Densely connected convolutional networks"](https://arxiv.org/abs/1608.06993)
- ["Unet: Convolutional networks for biomedical image segmentation"](https://arxiv.org/pdf/1505.04597.pdf)
- ["Parallel WaveNet: Fast High-Fidelity Speech Synthesis."](https://arxiv.org/abs/1711.10433)
- ["Learning Aligned Cross-Modal Representations from Weakly Aligned Data"](https://arxiv.org/abs/1607.07295)
- ["Unsupervised Cross-domain image generation"](https://arxiv.org/abs/1611.02200)
- ["Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"](https://junyanz.github.io/CycleGAN/)

## 12/14/2017
["Parallel Inversion of Huge Covariance Matrices"](https://arxiv.org/abs/1312.1869)
["Families of Algorithms Related to the Inversion of a Symmetric Positive Definate Matrix"](http://www.cs.utexas.edu/users/flame/pubs/toms_spd.pdf)
["Convex Optimization for Big Data"](https://arxiv.org/pdf/1411.0972.pdf)
["Iterative Methods for Computing Generalized inverses of Matrices"](http://pmf.ni.ac.rs/pmf/doktorati/doc/2012-04-27-ms.pdf)
[ Sunway TaihuLight System Applications](http://engine.scichina.com/downloadPdf/rdbMLbAMd6ZiKwDco)
["Undamentals of Linear Algebra and Optimization"](http://www.seas.upenn.edu/~cis515/linalg.pdf)
[""](http://www.siam.org/pdf/news/637.pdf)

## 12/18/2017

### Image/Video Matting
["Deep Image Matting"](https://arxiv.org/pdf/1703.03872.pdf)
["Automatically Removing Backgrounds from Images"](https://news.developer.nvidia.com/ai-software-automatically-removes-the-background-from-images/)
[ratings](http://videomatting.com/#rating)
### Python
[Inside Numpy: how it works and how we can make it better](https://youtu.be/fowHwlpGb34)
  - PyPy supporting Numpy


## 12/22/2017
[IMSL](https://www.roguewave.com/products-services/imsl-numerical-libraries/c-libraries/features)
[PLASMA](http://www.rce-cast.com/Podcast/rce-26-plasma-parallel-linear-algebra-software-for-multicore-architectures.html)
[High-performance Cholesky factorization for GPU-only execution](High-performance Cholesky factorization for GPU-only execution)
  
  - PLASMA tackles improving the expensive fork-join implementation in BLAS 3 by designing and using tile algorithms to achieve high performance.
  -The difficult-to-parallelize tasks are the panel factorizations
  - [MAGMA ](http://magma.maths.usyd.edu.au/magma/)
  - schedules the difficult-to-parallelize tasks on CPUs, and thus is not directly applicable for GPU-only execution)
  - [Level-3 Cholesky FActorization Routines](http://www.netlib.org/lapack/lawnspdf/lawn249.pdf)
  - [Matrix Algebra on GPU](https://www.olcf.ornl.gov/wp-content/training/electronic-structure-2012/ORNL-ESWorkshop.pdf)
    - Future Computer Systems: [ "AMD Fusion", "Nvidia Denver", "Intel MIC"]
    - Software Follows Hardware: [Linpack:BLAS1,LAPACK:BLAS3, ScaLAPACK:PBLAS,PLASMA:[DAG,blocks,kernls],MAGMA:[hybrid schuduler, hybrid kernels]
    - Compute-Communication gap is exponentially growing
      + Processor speed improves 59%, memory bandwidth 23%, latency 5.5%

## 12/27/2017
[cpp-annotated](https://blog.jetbrains.com/clion/2017/12/cpp-annotated-sep-dec-2017/)

#### Audio Studio Research
[muse box](http://www.museresearch.com/products/index.php)
[UAD Satellite](https://www.uaudio.com/uad-accelerators/uad2-satellite-usb.html)

## 12/28/2017
[python magic](https://rszalski.github.io/magicmethods/)

## 12/29/2017
####Machine Learning
##### People

[Emmanuel Candes](https://statweb.stanford.edu/~candes/)
  
  - [Stats 330 "And Introduction to Compressive Sensing](https://statweb.stanford.edu/~candes/stats330/index.shtml)
    + [Convex Optimization](http://techbus.safaribooksonline.com/book/math/9781107385924/firstchapter#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODExMDczODU5MjQlMkZwMjhfNV9odG1sJnF1ZXJ5PQ==)
    + 

[David Blei](http://www.cs.columbia.edu/~blei/)
[Tamara Broderick](http://www.tamarabroderick.com)
[Daniel Hsu](http://www.cs.columbia.edu/~djhsÂu/)
[Frank Wood](http://www.robots.ox.ac.uk/~fwood/)
[Marco Cuturi](http://marcocuturi.net)
[Finale Doshi-Velez](https://www.seas.harvard.edu/directory/finale)
[Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/)
[Neil Lawrence](http://inverseprobability.com)
[Ryan Adams](http://people.seas.harvard.edu/~rpa/)
[David Warde-Farley](http://www-etud.iro.umontreal.ca/~wardefar/)
[Hugo Larochelle](http://www.dmi.usherb.ca/~larocheh/index_en.html)


[Elad Hazan](https://www.cs.princeton.edu/~ehazan/)
  
  - [Optimization for Machine Learning](https://www.youtube.com/watch?v=f0qQsz4-o68&feature=youtu.be)
  - 

   [David Pfau](http://www.columbia.edu/~dbp2112/)
[Mu Li](http://www.cs.cmu.edu/~muli/)
[Martin Abadi](https://research.google.com/pubs/abadi.html)
[Derek Murray](https://research.google.com/pubs/DerekMurray.html)


####INFOSEC

[Blogs to Read](https://digitalguardian.com/blog/top-50-infosec-blogs-you-should-be-reading)

[INFOSEC](https://en.wikipedia.org/wiki/Information_security)


## 12/29/2017

[Accelerating Your Algorithms with Python and MKL](https://www.youtube.com/watch?time_continue=13&v=frOiCeljcsY)

- Intel Python performance much better than openBLAS on Xeon Phi processors
- Intel Vtune
- [NumExpr](http://numexpr.readthedocs.io/en/latest/user_guide.html)
-  Native C++ vs Python variants on Black Scholes { C++:4800,Cython:3400,NumExp:1200, Numpy:440 } MOPS 
  +  Compilation with Intel Compiler
-  [DAAL](https://software.intel.com/en-us/Intel-daal?cid=sem43700010409076506&intel_term=intel+daal&gclid=CjwKCAiAj53SBRBcEiwAT-3A2CoksapgyRmw31tHZeHRpCgxYifzrhYsp2mwBh4pzz6VdnAnN_GkeBoCfWcQAvD_BwE&gclsrc=aw.ds)
-  [DAAL4Py](https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library/topic/746317)
- [Daal on github](https://github.com/daaltces/pydaal-getting-started)

## 1/02/2018
[Audio Processing](https://towardsdatascience.com/the-promise-of-ai-in-audio-processing-a7e4996eb2ca)

- Selective Noise Cancelation
- Hi-fi audio reconstruction
- Analog audio emulation
- Speech processing
- Improved spatial simulations


[WaveNet](https://arxiv.org/pdf/1609.03499.pdf)
[Toward Data Science](https://towardsdatascience.com/a-deeper-understanding-of-nnets-part-1-cnns-263a6e3ac61)

[Le Cunn](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

[Urban Demographics](https://urbandemographics.blogspot.com.br/2017/12/the-most-visited-posts-of-2017.html)

## 1/03/2018
[Jim Cruchfield](http://csc.ucdavis.edu/~cmg/), [Complexisty Science Center](http://csc.ucdavis.edu/~chaos/)
[@SantaFe](https://www.santafe.edu/people/profile/james-p-crutchfield)
 
  - [Why Information](http://csc.ucdavis.edu/~chaos/Talks/COS_Singapore.pdf)
  - [Complexity of Simplicity slides James Crutchfield]()
  - [ Statistical Physics of Deep Learning](https://youtu.be/7KCWcx-YIRI)
  - [ Statistical Physics of Deep Learning.pdf]( https://www.msri.org/workshops/796/schedules/20463/documents/2705/assets/24735)
- [Entropy](http://www.cmp.caltech.edu/~mcc/Chaos_Course/Lesson8/Entropy.pdf)
- [Computational Mechanics of Input-Output Processes](https://escholarship.org/content/qt5p4164x0/qt5p4164x0.pdf)
- [Information Transfer](https://arxiv.org/pdf/nlin/0001042)
- [Partial Info (Williams Beer)](https://pdfs.semanticscholar.org/ea5d/983f480e1b2be06e52400e8721830b4ef15c.pdf)
    + Synergistic Informations
- [14](https://www.infoamerica.org/documentos_word/shannon-wiener.htm#_ftn14)

[![Everything Is AWESOME](http://i.imgur.com/Ot5DWAW.png)](https://youtu.be/StTqXEQ2l-Y?t=35s " Everything Is AWESOME")


## 1/04/2018

- [Rabbit MQ in Action]()
  + screenshot
- [Mahout in Action]()
  + screenshot
- [Dynamic Bayes (Murphy)](http://www.cs.ubc.ca/~murphyk/papers/dbntalk.pdf)

## 1/05/2018
[ Thomas Virtanen](http://www.cs.tut.fi/~tuomasv/publications.html)

- [Stephen Boyd](https://web.stanford.edu/~boyd)
  + [Convex Optimization Notes](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf)

## 1/09/2018

[Machine Learing](http://techbus.safaribooksonline.com/book/programming/machine-learning/9780128015223/chapter-18-neural-networks-and-deep-learning/s0075_html_4?percentage=&reader=html#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODAxMjgwMTUyMjMlMkZzMDAxMF9odG1sXzE0JnF1ZXJ5PQ==)

## 1/10/2018
http://cs.brown.edu/~jes/book/pdfs/ModelsOfComputation_Chapter8.pdf
http://foresthillshs.enschool.org/ourpages/auto/2011/11/2/74617344/Number%20Theory%20and%20Music.pdf

[ porpouise looking for ]( http://www.math.psu.edu/cao/ )
[ relaxor ferromagnetic ](https://www.ncbi.nlm.nih.gov/pubmed/18276540)


## 1/11/2018
[Mind MOdeling](https://mindmodeling.org)

## 1/12/2018
[Hinton Git](https://github.com/khanhnamle1994/neural-nets)


## 1/15/2018
[Adam](https://arxiv.org/abs/1412.6980)
[ Torch ](http://torch.ch)

## 1/19/2018
[Cpp notes](Cppnotes.md)

## 1/22/2018
 
[Godbolt on Spectre and Meltdown](https://www.youtube.com/watch?v=IPhvL3A-e6E)

- Meltdown
  + Effects mostly Intel, some ARM
  + access any physical RAM
  + somewhat straight forward software fix
- Spectre
  + access RAM via privileged process
  + can work from JS
  + all modern CPU types affected
  + highly involved fixes
- KAISER patches change OS so that kernel memory is not mapped into every process.
- *retpoline* for indirect jumps
- LFENCE after bounds checks
- \mucode update

[Daniel Gruss, Microarchitectural Incontenence](https://www.youtube.com/watch?v=cAWmNp3Ukqk)
[cache_template_attacks](https://github.com/IAIK/cache_template_attacks)


## 1/25/2018

- [Python Design Patterns](http://techbus.safaribooksonline.com/book/programming/python/9781783989324/firstchapter#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODE3ODM5ODkzMjQlMkZjaDAzczAyX2h0bWwmcXVlcnk9)
- Factory, Builder, Prototype, Adaptor (Grok, Traits), 

## 1/26/2018

[parallel algorithms](https://www.cs.cmu.edu/~guyb/papers/BM04.pdf)
[Parallel Algorithms come of age](https://www.cs.cmu.edu/~guyb/papers/Qatar17.pdf)

  - Three models that characterize a network in terms of its latency and bandwidth are the Postal model [14], the Bulk-Synchronous Parallel (BSP) model [85], and the LogP model [29].
  - Because there are so many different ways to organize parallel computers, and hence to model them, it is difficult to select one multiprocessor model that is appropriate for all machines. The alternative to focusing on the machine is to focus on the algorithm.
  - An algorithm’s work W is the total number of operations that it performs; its depth D is the longest chain of dependencies among its operations. We call the ratio P = W/D the parallelism of the algorithm.
  - Graph problems are often difficult to parallelize since many standard sequential graph techniques, such as depth-first or priority-first search, do not parallelize well.
  - If l = m = n, (direct dense matrix multiplication) does O(n^s3) work and has depth O(logn)
  - When using Gauss-Jordan elimination, two of the three nested loops can be parallelized, leading to an algorithm that runs with O(n3) work and O(n) depth.
  - Quicksort P = O(nlogn)/O(logn^2) = O(n/logn)
  
## 1/29/2018

[Eric Humphrey](https://steinhardt.nyu.edu/marl/people/humphrey)

- [ Four Key](https://pdfs.semanticscholar.org/5df1/8ba8d4be6daf3ec2c3d617ac2fae9231b35a.pdf)
- Constant Q -> HCDF, peak picking -> Chroma segmentation and averaging -> Chord Recognizer

[4GENNOW tax structures](https://4gennow.com/resources/)

- Scedule C gets audited more often
- S corp
  + Income tax paid by individual share holders
- C corp Corporate tax rate is reduced to a maximum rate of 21%
  + Tax Cuts and jobs act of 2017
  + 20% deduction of "qualified business income" 

[SOU Address]()


## 2/1/2018
[Integer programming]()
[PuLP](https://pypi.python.org/pypi/PuLP
[git PuLP](https://github.com/coin-or/pulp)
[COIN CBC](https://projects.coin-or.org/Cbc)
[git CBC](https://github.com/coin-or/Cbc)
[nauty traces](http://pallini.di.uniroma1.it)

[HMM again](http://cs.brown.edu/research/ai/dynamics/tutorial/Documents/HiddenMarkovModels.html)

## 2/9/2018

[MARL](https://steinhardt.nyu.edu/marl/publications#)
[ACE](https://pdfs.semanticscholar.org/5df1/8ba8d4be6daf3ec2c3d617ac2fae9231b35a.pdf)
[music21](http://web.mit.edu/music21/)
[Warp](http://www.cs.cmu.edu/afs/cs/project/iwarp/archive/iWarp-papers/spaa92-mesh-sorting.ps)
[WARP systolic](https://en.wikipedia.org/wiki/WARP_(systolic_array))
[AD9371](http://www.analog.com/en/products/rf-microwave/integrated-transceivers-transmitters-receivers/wideband-transceivers-ic/AD9371.html#product-overview) 
[l-diversity](https://en.wikipedia.org/wiki/L-diversity)
[Jetson TX-2](https://developer.nvidia.com/embedded/buy/jetson-tx2)


## 2/13/2018

[ Marathon ](https://mesosphere.github.io/marathon/)
[ MESOS ](http://mesos.apache.org)
[ DC/OS ](https://dcos.io)

[isophonics](http://isophonics.net/content/reference-annotations)
[ OMRAS2 ](http://www.matthiasmauch.net/_bilder/late-breaking-C4DM.pdf)
[ FPGA ML](http://www.eproductalert.com/digitaledition/embeddedsystems/2017/12/embedded_systems_engineering_december_2017.pdf)
[ DO-254 ](https://en.wikipedia.org/wiki/DO-254)
[ Linear Programming ](http://techbus.safaribooksonline.com/9781466552647/9781466578609_c02_htm)
[ HMM ](http://hmmlearn.readthedocs.io/en/stable/tutorial.html#training-hmm-parameters-and-inferring-the-hidden-states)
[ Island Algorithm](https://en.wikipedia.org/wiki/Island_algorithm)
[ Forward Backward Algorithm](https://en.wikipedia.org/wiki/Forward–backward_algorithm#RussellNorvig10)
[ Bitonic Sorter ](https://en.wikipedia.org/wiki/Bitonic_sorter)
[ Pulp ](https://github.com/coin-or/pulp/tree/master/src/pulp)
[ hmmlearn ](http://hmmlearn.readthedocs.io/en/stable/tutorial.html#training-hmm-parameters-and-inferring-the-hidden-states)
[ parallel algorithms ](https://www.cs.cmu.edu/~guyb/papers/BM04.pdf)
[ backdrop systolic ](https://medium.com/@yaroslavvb/backprop-and-systolic-arrays-24e925d2050)
[ face recognition ](https://pypi.python.org/pypi/face_recognition)
[ compressive systems ](https://statweb.stanford.edu/%7Ecandes/stats330/index.shtml)
[ dlib ](https://pypi.python.org/pypi/dlib)
[ embedded ](http://www.eproductalert.com/digitaledition/embeddedsystems/2017/12/embedded_systems_engineering_december_2017.pdf)

## 2/15/2018

[chip](http://news.mit.edu/2018/chip-neural-networks-battery-powered-devices-0214)
[ dario gil](Dario Gil)
[ in-memory computation](https://pdfs.semanticscholar.org/4bb5/2ae9d53277e9eec35fc658a14960a5251ede.pdf)


