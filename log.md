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

[ Alan Parsons Video ]( https://www.artandscienceofsound.com    )
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

## 2/21/2018

[ peptide resonance ](http://www.jbc.org/content/193/1/397.long)
[ capsid ](https://en.wikipedia.org/wiki/Capsid)
[ structural biology ](https://en.wikipedia.org/wiki/Structural_biology)
[ acoustic parameters ](http://www.sengpielaudio.com/RelationshipsOfAcousticQuantities.pdf)

## 2/23/2018

[ acoustic pumping ](https://arxiv.org/pdf/1707.04276.pdf)
[ Tensor Comprehensions ](https://arxiv.org/pdf/1802.04730.pdf)
[ ATen ](https://github.com/zdevito/ATen)
[ Halide ](https://github.com/halide/Halide)

## 2/26/2018

[ Memristor ](https://spectrum.ieee.org/nanoclast/semiconductors/materials/molybdenum-disulfide-helps-tune-memristors)
[ Hersam Group ](http://www.hersam-group.northwestern.edu)
- Analog Analogy - any analogy can analagously simulate (compute)
[ LOIHI ](https://newsroom.intel.com/editorials/intels-new-self-learning-chip-promises-accelerate-artificial-intelligence/)


## 3/1/2018

[ Cognitive Systems Engineering ](https://books.google.com/books?id=ASO7BQAAQBAJ&pg=PA47&lpg=PA47&dq=Annelise+Mark+Petjersen&source=bl&ots=X535Guf8gM&sig=k460FlMRV2HOhTWVeryolODEXsc&hl=en&sa=X&ved=0ahUKEwi8q6eOrMvZAhVDtlkKHQ56CuwQ6AEIQDAE#v=onepage&q=Annelise%20Mark%20Petjersen&f=false)

[ VIHO ](http://www.it.uu.se/research/project/viho#deltagare)
[ Mark Ackerman ](https://www.si.umich.edu/people/mark-ackerman)
[ Hala Annabi ](https://ischool.uw.edu/people/faculty/profile/hpannabi)
[ It's FASSE ](https://link.springer.com/article/10.1023/A:1009895915332)
[ Cognitive Systems Research ](https://www.sciencedirect.com/journal/cognitive-systems-research)


## 3/5/2018

[ cognitive systems](https://www.mitre.org/sites/default/files/pdf/05_1361.pdf)
[ CSE ](https://pdfs.semanticscholar.org/ca96/1aa540819e0783514ab58bbdf035c877df74.pdf)
[ Angelica Peer ](https://scholar.google.com/citations?user=5uDjPl0AAAAJ&hl=th)
[ Robert Jenke ](https://scholar.google.com/citations?user=-sk9ynkAAAAJ&hl=de)
[ Deep Slide share ](https://www.slideshare.net/TessFerrandez/notes-from-coursera-deep-learning-courses-by-andrew-ng)


## 3/6/2018

## 3/18/2018

[ google cloud ](https://software.seek.intel.com/Edge_Devices_Webinar_Reg)


## 3/21/2018

[SANS](https://www.sans.org/dfppclp?utm_campaign=NA-Forensics-Digital-Search&utm_medium=cpc&utm_source=Google&utm_content=hands-on-ad&utm_term=P)
[Audio Forensics](https://www.wired.com/2007/10/audio-forensics-experts-reveal-some-secrets/)

- DCLive Forensics
- aural steganography

[Owen Forensics](http://owenforensicservices.com/about.html)
https://www.youtube.com/watch?v=NRRzs0bB0a0&feature=youtu.be
[Brian King](https://www.linkedin.com/in/mrbrianking/)
[ Paris Smaragdis](http://paris.cs.illinois.edu)
[ SiSEC 2013 ](http://sisec.wiki.irisa.fr/tiki-index165d.html?page=Professionally+produced+music+recordings)
[ SiSEC 2018](https://sisec.inria.fr)
[ SiSEC MUS ](https://www.sisec17.audiolabs-erlangen.de/#/)



## 3/25/2018
https://link.springer.com/chapter/10.1007/978-3-319-53547-0_31
http://www.ti.com/product/TMS320C5534
[ Interior Point Methods](https://people.kth.se/~andersf/doc/sirev41494.pdf)
CEH http://techbus.safaribooksonline.com/9780134677552
CompTIA Security+ 
[Sari Greene](https://www.linkedin.com/in/sarigreene/)

- Module 1
  + Lesson 1: Analyze Indicators of Compromise and Determine Malware
    * 1.1 Types of Malware
      - Viruses
        + Types: bootsector, file infector, companion, macro virus
        + Mechanism: stealth, memory resident, armored, polymorphic, metamorphic
      - Worms
        + Self replicating, use network transport,
        + Types: Crypto variant/Ransomware, Command and Control, Advanced Persistant Threat, Bot/ zombie
      - Trojans
        + Malicious code disguised as a legitamate application
        + Types:  RAT, Backdoor, Downloader, Keylogger
      - Rootkit
        + Firmware, Kernel, Persistant, Application, Library
      - Spyware
        + Keylogger, Monitors, Adware, Tracking Cookies, Click Fraud

    * 1.2 Indicators of Compromise
      - specific artifact, virus signature, IP address, malicious URL, command and control connection, file changes
      - Malware Modus Operandi
        + Exploit->Masquerade->Polymorphism->Callback->Accomplish
    * 1.3 Security in Action
      - AV, anti-malware, Post-infection scan, Log inspection, Malware Intelligence, Malware Verification, Reverse Engineering
      - 


## 03/28/2018

- [ Phase reconstruction ](https://pdfs.semanticscholar.org/ade8/d1511a61d78948bb0d43e207593389935421.pdf?%5C_ga=2.229355228.1725061846.1494658711-1308334183.1494658711)
- [Intel persistant memory](https://software.intel.com/en-us/videos/the-nvm-programming-model-persistent-memory-programming-series)


## 04/06/2018
[Tensorflow dev summit](https://www.tensorflow.org/dev-summit/)
- pre-trained models, pods etc.
- [ETL for tensorflow](www.tensorflow.org/programmers_guide/datasets)
- [Eager Mode](www.tensorflow.org/programmers_guide/eager)

## 04/20/2018
- Fuel Economy & Gas Emission Standards for Vehicles
  + 95 RON 3% increase at what cost?
- Transportation Infrastructure
  + Kyle Schneweis - Nebraska Transportation Department Director
  + Dan Gilmartin - D- score on roads, this year particularly bad.
  + Sen. Gary Peters - Michigan infrastructure pipes, internet, etc. problems.
  + 
- Federal Marijuanna Policy (SAM - Smart Approaches to Marijuanna)
  + Kevin Sabat - Federal Marijuanna Policy
  + Jake Nelson - AAA Traffic Advocacy and Research Director
  + Arthur Burnett - DC Superior Court Judge
  + Dr. Roneet Lev - Scripps Mercy Hospital (San Diego)
  + Christine Miller - Psychosis and marijuanna
  + Patrick Kennedy - Former Representative from Rhode Island
- 2019 Defense Budget Request
  + Nuclear Triad 
  + Heather Wilson - USAF Secretary
    * Joint Cyber Warfighting
    * Space Enterprise Consortium
  + Rep. Niki Tsongas - Dem Massachusetts
  + Mark Esper - Army Secretary -> [U.S. Army Futures Command](https://www.army.mil/article/197886/us_army_futures_command_to_reform_modernization_says_secretary_of_the_army)
  + Richard Spencer - Naval Secretary
  + Bradley Byrne - R. Alabama mad about LCS production
  + Seth Moulton - D. Massachusetts -> AI, Cyber, anti-satellite, 
  + Martha McSalley - R. Arizona -> A10 sustainment, Apache ANG decommission/transfer to Active Duty.
  + Trent Kelley -> R. Missisipi ANG more experienced and more qualified.  JSTARS 2025-2035 replacement capability gap.
  + Mike Turner - R. Ohio -> 
  + Don Bacon - R. Nebraska -> Compass Call 
- Housing and Urban Development  
  + 16% Cuts  Sherod Brown vs. Ben Carson. down and dirty.  
- Telecommunications Policy
  + David Redl - NTIA -> DoD operations <- Spectrum Relocation Act
    * [David Redl](https://www.ntia.doc.gov/page/david-j-redl)
  + Tom Power CTIA Wireless Assocation
  + https://defensesystems.com/articles/2018/03/01/ntia-dod-spectrum-5g.aspx
  

## 4/25/2018
- [ 80 Data Science Books ](https://www.octoparse.com/blog/80-best-data-science-books-that-are-worthy-reading/)
- [ linux kernel development](http://techbus.safaribooksonline.com/book/operating-systems-and-server-administration/linux/9781785883057/mastering-linux-kernel-development/48161f75_547b_4e73_98e6_384ba552669d_xhtml?query=((linux+kernel+development))#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODE3ODU4ODMwNTclMkY3MWI2NTUyZV9iNzAwXzQ2NzVfYjEzNV81M2I3Zjg5OTc0NzdfeGh0bWwmcXVlcnk9KChsaW51eCUyMGtlcm5lbCUyMGRldmVsb3BtZW50KSk=)


## 5/1/2018
- [Professional C++](http://techbus.safaribooksonline.com/9781119421306/head_2_67_html?percentage=0&reader=html#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODExMTk0MjEzMDYlMkZoZWFkXzJfNzBfaHRtbCZxdWVyeT0oKGRvaW5nJTIwZGF0YSUyMHNjaWVuY2UpKQ==)
- Abstraction and Reuse
- [typname or class](https://stackoverflow.com/a/213135)
- Designing an Exposed Interface
  + Consider the audience
- Designing Reusable Code
  + Use Abstraction, Reduce Dependencies, Use Templates
- Design for Extensibility, Closed for Modification
-  Dynamic memory: string class, vector container, unique_ptr, shared_ptr, 


## 5/6/2018
http://techbus.safaribooksonline.com/book/programming/cplusplus/9781484218761/firstchapter#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODE0ODQyMTg3NjElMkZhNDE3NjQ5XzFfZW5fYm9va2Zyb250bWF0dGVyX29ubGluZXBkZl9odG1sJnF1ZXJ5PQ==


## 5/31/2018
- Introduction to Persistent Memory Configuration and Analysis Tool Webinark
- [pmdk](https://github.com/pmem/pmdk)
-  [pmemm](http(software.intel.com/pmemm)
- [overview](https://software.intel.com/en-us/articles/introduction-to-programming-with-persistent-memory-from-intel)
- [nvdimm](https://en.wikipedia.org/wiki/NVDIMM)

## 6/04/2018
- [Dask webinar ]( http://bit.ly/2s5Zt4v )
- [Matt Rocklin](http://matthewrocklin.com/blog/work/2017/02/07/dask-sklearn-simple)
- scale to larger than memory data
- [dask setup](https://dask.pydata.org/en/latest/setup.html)
- [dask yarn](https://github.com/dask/dask-yarn)

- Apple Keynote
  + USDZ - AR Augmented Reality 
  + Adobe Creative Cloud -> AR
  + Apple Measure

- [ TDS - Human Like ML](https://towardsdatascience.com/human-like-machine-hearing-with-ai-1-3-a5713af6e2f8)
  + Cochlear Nucleus -> Superior Olive -> Lateral Lemniscus -> Inferior Colliculus -> Medial Geniculate
  + [ J.J. Eggermont ](J. J. Eggermont, “Between sound and perception: reviewing the search for a neural code.,” Hear. Res., vol. 157, no. 1–2, pp. 1–42, Jul. 2001.)
  +   “chopper” (stellate) neurons
  

## 06/12/2018

  [10 Core Guidelines You Need to Start Using Now](https://www.youtube.com/watch?v=XkDEzfpdcSg)
  - [Kate Gregory](http://www.gregcons.com/kateblog/)
  - Bikeshedding is bad
  - C.45: in-class member functions
  - F.51: default arguments
  - Don't hurt yourself
  -  C.47: Define and initialize member variables in the order of the member declaration
  -  I.23: Keep the number of functional arguments low 
  -  Stop using that
  -  ES.50: Don't cast away const
    +  mutable in cachedValue
  - I.11 Never Transfer Ownership by a raw pointer
    +  in 2017 Return by value may be the way.  You don't mind copy which may be elided.
    + [GSL owner ](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#SS-ownership)
  - Use this new thing properly
  - F.21: To return multiple "out" values, prefer returning a tuple or struct 
    - std::optional - if the two things are [an object] and [a bool] about whether or not that object is usable.  value_or(),  std::make_ tuple
      + type, tie, structured_bindings
        * std::tie(answer, number), auto[answer,answer]
        * [structured bindings](http://en.cppreference.com/w/cpp/language/structured_binding)
        *  struct should be first choice if structure has purpose/name
  - Enum.3: Prefer class enums over "plain" enums
  - Guideline Support Library
  - I.12: Declare a pointer that must not be null as not_null
    + gsl::not_null<Service*>
  - ES.46: Avoid lossy (narrowing, truncating) arithmetic conversions
    ```c++ 
      #pragma warning( push )
      #pragma warning( disable: ThatWarning )
      //code with ThatWarning here
      #pragma warning( pop )
    ```
    
[ Practical C++17 ](https://www.youtube.com/watch?v=nnY4e4faNp0)
[Jason Turner](https://github.com/lefticus)
- Structured Bindings: auto [a,b,c] = <expression>; 
- if-switch-init expressions
- ```c++
     if (auto[key,value] = *my_map.begin();key=="mykey"){} 
- emplace_back, emplace_front
- std::string_view
- Nested Namespaces
- Class Template Type Deduction  Pair p{1,2.3}
- if constexpr (compile-time conditionals )
- fold expressions for variadic parameter packs
- noexcept is now part of the type system   (negative..?)
- Storing a map of string_view is extremely risky...
```c++ 
std::map<std::string_view,int>map;
void add(const std::string &t_str,int val)
{ map[t_str]=val; } 
```
- is_transparent{}; ?.... lol!
-  [SFINAE](https://en.wikipedia.org/wiki/Substitution_failure_is_not_an_error)
-  

##### 
- [Structured Bindings]()
- [ It's Complicated, Kate Gregory](https://youtu.be/tTexD26jIN4)
  + AAA almost always auto
  + UB undefined behaviour
  + RVO return value optimization
  + NDR Ill formed, No Diagnostic Required
  + ADL Argument Dependent Lookup
  + CAT Const All the Things, constexpr all the things
  + IIILE  Imediately invoked initializeing lambda expression
  + ODR One Definition Rule
  + SFINAE Substitution Failure is not an Error
  + RAII Resource Aquisition Is Initialization
  + Simplifying Principles
    * Move and hide complexity if that's all you can do
    * Aim to actually eliminate it
    * Making code simpler and adding abstractions reduce bugs as well as just makeing a developer's life happier
    * Readability matters
    * Names matter
  + Const correctness and "mutable" example

```c++

class Stuff
n{
private:
	int number;
	double number2;
	int LongComplicatedCalculation const;
	mutable int cachedValue;
	mutable bool cachedValue;

public:
	Stuff(int n1, double n2) : number1(n1),
number2(n2),cachedvalue(0),cachedvalid(false)P{
	bool Service1(int x);
	bool Service2(int y);
	int getValue() const;
};
```

- Overly simple guidelines
  * Don't use exceptions
  * Don't use templates
  * Whatever you new you must delete
  * Don't use raw pointers
  * Don't use the standard library
- Alternatives
  * Use the above when they help your application be simpler and more correct
  * Getting noexcept right is hard
  * F.21 discussion

### 06/14/2018

[Darpa Open Catalog](https://opencatalog.darpa.mil/XDATA.html)

- [Smallk](http://smallk.github.io)
	- low rank approximation algorithms, like MU, HALS, and BPP, some communication is necessary
	- pro-posed algorithm ensures that after the input data is initially read into memory, it is never communicated
	- Block Principle Pivoting is used
	- [74 ironic](https://arxiv.org/abs/1605.06848) 
	- ["Nonnegative and Matrix and Tensor Factorizations"](https://pdfs.semanticscholar.org/94cc/6daad548a03c6edb0351d686c2d4aa364634.pdf)
- [Skylark](http://xdata-skylark.github.io/libskylark/)

### 06/15/2018

[Blaze Overview and Tutorial](https://www.youtube.com/watch?v=jNfwjZgCj6k)
["C++ atomics, from basic to advanced. What do they really do](https://www.youtube.com/watch?v=ZQFzMfHIxng)
- std::atomic, compare_exchange_strong, fedch_add, 
- atomic speed? -> spinlock > atmomic > CAS > mutex 
- std::atomic<T>::is_lock_free()
- C++17 constexpr is_always_lock_free()
- cache line sharing
- compare_exchange_weak()  times out and returns false?

### 06/20/2018

[XDATA](https://hpcuserforum.com/presentations/virginia-april2015/DARPAxdata-overview.pdf)

### 06/26/2018

[C++ the Newest Old Language ( https://www.youtube.com/watch?v=HAFrggEDr5U)
- *C++11* auto, range-for, lambdas,move, smart pointers, constexpr, atomics, UDLs 
- *c++14* return type deduction, better lambdas
- *c++17* if constexpr, optional, variant, string_view
- *c++20* Concepts, ranges, coroutines... maybe?
```c++ 
// Example of inline lambda using std::accumulate
double rms(const vector<double> &v){
    return sqrt(accumulate(v.begin(), v.end(),0.0,
        [](double partialSum,double elem){return partialSum + elem*elem;}))
}
```
```c++
// Example of using value types
class Pos{
    float x{};
    float y{};
}
public: 
    constexpr Pos() = default;
    constexpr Pos(float x, float y)
        : x(x),y(y){}
    constexpr Pos operator +(Pos other) const{
        return Pos(x + other.x, y + other.y);
    }
```
```c++
// Example illustrating object lifetimes, move of rvalue
class Document {
    vector<unique_ptr<elem>>objects;
public:
    void add(unique_ptr<Elem> &&object) {
        objects_.emplace_back(move(object));
    }
};
```
-fsanitize=address

### [STL algorithms in action](https://www.youtube.com/watch?v=eidEEmGLQcU)
- sorting, nth-element, binary sort, merge, set operations on sorte structures, heap operations, minimum and maximum, lexicographical comparisons, permutation, generators

- [ Tony Bell: Emergence and Submergence in the Nervous Systems](https://archive.org/details/ucbvs265_neural_comp_2012_09_12_Tony_Bell)
- [Postsynaptic Organization of Synapses - Morgan Sheng ](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3225953/)
- [Metabolite changing ion path?](https://en.wikipedia.org/wiki/Macromolecular_crowding)
- [Quantum Biology](http://www.pnas.org/content/108/52/20908)
- [ Multiscale Auditory Entrainment](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4673342/pdf/fnhum-09-00655.pdf)
- [ Entrainment ](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4673342/pdf/fnhum-09-00655.pdf)
- [ Cross-Level learning](http://cseweb.ucsd.edu/~gary/pubs/bell07.pdf)


### 07/11/2018

-[A Hierarchical Latent Vector Model
for Learning Long-Term Structure in Music](https://arxiv.org/pdf/1803.05428.pdf)

- [Nonlinear Acoustics](http://perso.univ-lemans.fr/~vtournat/wa_files/NLALectureVT.pdf)

### 07/14/2018

- [ Hardware Accelerators for Machine Learning ](https://cs217.github.io)
  + [Lectures](https://cs217.github.io/readings)
    * [Donoho](https://cs217.github.io/assets/lectures/StanfordStats385-20170927-Lecture01-Donoho.pdf) 
  + [Slides](https://cs217.github.io/lecture_slides)
- [ Four Horsemen of Silicon](https://cseweb.ucsd.edu/~mbtaylor/papers/DAC_DaSi_Horsemen_2012_Slides_Final.pdf)
- [ TABLA ](https://www.cc.gatech.edu/~hadi/doc/paper/2015-tr-tabla.pdf)
  + [ Denard Scaling ](https://en.wikipedia.org/wiki/Dennard_scaling)
  + Algorithms that can be implemented with Stochastic Gradient
  + 
- [ Why Systolic ](http://www.eecs.harvard.edu/~htk/publication/1982-kung-why-systolic-architecture.pdf)
- [ Anatomy of High Performance GEMM](https://www.cs.utexas.edu/users/pingali/CS378/2008sp/papers/gotoPaper.pdf)
- [ Dark Memory](https://arxiv.org/abs/1602.04183)
- [ Google TPU ](https://arxiv.org/abs/1602.04183)
- [ CoDesign Tradeoff](https://ieeexplore.ieee.org/document/6212466/)
- [ Tesla V100 ](http://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- [ Efficient Processing of Deep Neural Networks: a Tutorial and Survey ](https://arxiv.org/pdf/1703.09039.pdf)
  + Network Output -> Inference
- [ Systems Approach to Blocking ](https://arxiv.org/abs/1606.04209)
- [ Eyeriss](https://people.csail.mit.edu/emer/papers/2016.06.isca.eyeriss_architecture.pdf)
- [ Spatial ](http://arsenalfc.stanford.edu/papers/spatial18.pdf)
- [ Graphcore](https://supercomputersfordl2017.github.io/Presentations/SimonKnowlesGraphCore.pdf)
- [EIE](https://arxiv.org/pdf/1602.01528.pdf)
- [Flexpoint](https://arxiv.org/pdf/1711.02213.pdf)
- [ Boris Ginsberg](https://arxiv.org/abs/1708.03888)
  + [presentation](http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf)
- [ Low Precision Training](https://arxiv.org/abs/1803.03383)
- [ Deep Gradient compression](https://arxiv.org/abs/1712.01887)
- [ Hogwild](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)
- [ Large Scale Destributed Deep Network](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf)
- [ Catapult](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/Cloud-Scale-Acceleration-Architecture.pdf)
- [ Brainwave ](https://www.microsoft.com/en-us/research/uploads/prod/2018/03/mi0218_Chung-2018Mar25.pdf)
  +  Project Brainwave leverages the massive Intel FPGA infra- structure that Microsoft has been deploying over the last few years
  +  
- [ Plastacine](http://dawn.cs.stanford.edu/pubs/plasticine-isca2017.pdf)
- [ DawnBench](https://cs.stanford.edu/~matei/papers/2017/nips_sysml_dawnbench.pdf)


### 07/15/2018

- [Connectomics](https://ai.googleblog.com/2018/07/improving-connectomics-by-order-of.html)
- [ Connectome dB ](https://db.humanconnectome.org/app/template/Login.vm;jsessionid=BC559E14193C7CF00D0BB19B6CB0DB0E)

### 07/17/2018

- [open weather map](https://openweathermap.org/technology)

### 07/18/2018

- [libraries.io](https://libraries.io)

### 07/19/2018
[ CppNow18](CppNow18.md)

[ jason turner embdeddedFM](https://www.embedded.fm/episodes/247)
[ odin Holmes ](https://github.com/odinthenerd)
  [ odin Holmes blogspot](http://odinthenerd.blogspot.com)

[ CppCon Github](https://github.com/CppCon)

#### 07/23/2018

- [ What's Next for Pandas ](https://www.youtube.com/watch?v=_-gJtO0XR48&index=4&list=PLGVZCDnMOq0oywykwgVAcGvsGzagyMbwSP)
  - [ Jeff Reback ](https://github.com/jreback)
  - [ slideshare ](https://www.slideshare.net/secret/sUXFArGxQ1RFX7)
  - [ Pandas2 ](https://pandas-dev.github.io/pandas2/) 
  - [ Ibis](http://docs.ibis-project.org) is an expression language (compiler).  Feels like Spark, Dask, deferred Pandas
  - [ Arrow ](https://arrow.apache.org) is a backend.  Fast, Flexible, Standard
    + columar database
    + Now introducing atomic units of computation
    
    
    ![ ][ images/unfriendly.png ]

- [ Scalable Machine Learning with Dask (SciPy 2018)](https://youtu.be/ccfsbuqsjgI)
  - [ TomAugspurger ](https://github.com/TomAugspurger)
  - [ @TomAugpurger ](https://twitter.com/tomaugspurger?lang=en)
  - [ Dask ](http://dask.pydata.org/en/latest/)
    + [ Dask-ML](http://dask-ml.readthedocs.io/en/latest/) builds on Dask to make machine learning more scalable
    


### 07/24/2018

- [ Szilard Pafka ](https://github.com/szilard)
    + [talk](https://www.youtube.com/watch?v=DqS6EKjqBbY)
    + [slides](https://www.youtube.com/watch?v=DqS6EKjqBbY)


### 07/25/2018

- [ LLNL Training Manuals ](https://hpc.llnl.gov/training/tutorials)


### 07/26/2018

#### FPGA Programming
[ 10 Ways to Program your FPGA ](https://www.eetimes.com/document.asp?doc_id=1329857&page_number=1)
- C/C++
- MyHDL (Python)
- CHISEL (Scala)
- JHDL (Java)
- BSV  (Haskell)
- MATLAB
- Labview FPGA
- SystemVerilog 
- VHDL/Verilog
- Spinal HDL

#### Data Science Platforms

- [Michelangio (Uber)] (https://eng.uber.com/michelangelo/)
  + [ Franziska Bell ](https://www.youtube.com/watch?v=TEYtXfhbsZQ)
    * 500 million time series at Uber
    * < 1 min time to detection
    * Misses expensive (business critical outage, noisy alerts are bad for team health
    ![ ][ images/EngineeringAtUber.png ]
    * Forecasing, personalization, dispatch
    * 5B forecasts per minute


- DeepBird
- [ FBLearn (Facebook)] (https://code.fb.com/core-data/introducing-fblearner-flow-facebook-s-ai-backbone/)


#### C++

- [Jon Kalb ](https://cppnow2018.sched.com/speaker/sched23)
  + [Exception Safe Code](http://exceptionsafecode.com)
    * Exception Handling := Seperation of Error Detection from Error Handling
      + Think Structurely, Maintain Invariance
      + Basic < Strong < No-Throw

    * [Part 2](https://www.youtube.com/watch?v=b9xMIKb1jMk)
      + Throw by value, catch by reference
      + async thread throw
        *  ```cpp std::Async(Func);int v(f.get()); // if Func() threw, it comes out here. ```
        * Nesting exceptions (25:42)
        * Dynamic Exceptions are NOT checked at compile time.
        * Two uses of "noexcept" keyword  (34:51)
          - noexcept specification (of a function)
            + noexcept(true), noexcept(false)
            + destructors are noexcept by default  noexcept(true)
          - noexcept operator
             
    * [Part 3](https://www.youtube.com/watch?v=MiKxfdkMJW8)


### 7/30/2018

- [ 10 minutes of yoga ](https://www.workandmoney.com/s/10-minute-yoga-routine-3a7e2b5bfee54695?utm_medium=cpc&utm_source=tab&utm_campaign=10minuteyoga-d994ddd21df26a22&utm_term=bonnier-popscinew)

- [Machine Learning Notes](Machine%20Learning.md)


### 7/31/2018

- tried to build Tensorflow<-Bazel from source for local Magenta demo... I know not why... fail fail :(
  - [ build tensorflow](https://gist.github.com/kmhofmann/e368a2ebba05f807fa1a90b3bf9a1e03)
  * brew, conda versions instead
- [ C++ Today The Beast is Back ](https://www.oreilly.com/programming/free/files/c++-today.pdf?mkt_tok=eyJpIjoiTURSak1USXdNR0ZsTURaaSIsInQiOiJOMEwrUGNPZVJJdDRnUlhlM3ZkMU1Oa2xXbDdrdEVOZHNMdkZENVlKdUZaTnlRME9FTTFpd1ZcL1pBaGg5dzlPQ3c5eFVYaTU2STdtcmN0enZic2lyTEJkTDhJMjJhc1wvbzZTb0VHVnVQRGNQak85WUJEVjRMTEtMVWp4dVQzVDRVIn0%3D)
  - [ Jon Kalb ](https://www.linkedin.com/in/jonkalb/)
  - [ @JonKalb ](https://twitter.com/_jonkalb?lang=en)

```c++
// Using lambdas for inner functions to simplify algorithms
template <typename InputIterator1, typename InputIterator2, typename OutputIterator>
auto move_merge(InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, InputIterator2 last2,
OutputIterator&& out) -> OutputIterator { 
  using std::move; using std::forward;
  auto drain = [&out](auto& first, auto& last){
    return move(first, last, forward<OutputIterator>(out));
  };
  auto push = [&out](auto& value) { *out = move(value); ++out; };
  auto advance = [&](auto& first_a, auto& last_a, auto& value_a,
                     auto& first_b, auto& last_b, auto& value_b) {
    push(value_a);
   if (++first_a != last_a) {
    value_a = move(*first_a); 
    return true;
  } else { // the sequence has ended. Drain the other one.
  push(value_b);
  out = drain(++first_b, last_b); return false;
  } 
};
if
else if (first2 == last2) { return drain(first1, last1); } auto value1(move(*first1));
auto value2(move(*first2));
for (bool not_done = true; not_done;) {
  if (value2 < value1) {
  not_done = advance(first2, last2, value2,
  }else{ }
}
  return out; 
}
```

```c++
//Demonstration of move semantics  from https://en.cppreference.com/w/cpp/utility/move
#include <iostream>
#include <utility>
#include <vector>
#include <string>
 
int main()
{
    std::string str = "Hello";
    std::vector<std::string> v;
 
    // uses the push_back(const T&) overload, which means 
    // we'll incur the cost of copying str
    v.push_back(str);
    std::cout << "After copy, str is \"" << str << "\"\n";
 
    // uses the rvalue reference push_back(T&&) overload, 
    // which means no strings will be copied; instead, the contents
    // of str will be moved into the vector.  This is less
    // expensive, but also means str might now be empty.
    v.push_back(std::move(str));
    std::cout << "After move, str is \"" << str << "\"\n";
 
    std::cout << "The contents of the vector are \"" << v[0]
                                         << "\", \"" << v[1] << "\"\n";
}
```

[move-forward](http://bajamircea.github.io/coding/cpp/2016/04/07/move-forward.html)

```c++
//Lambdas as scope with a return value
deque<int> queue;
bool done = false;
mutex queue_mutex;
condition_variable queue_changed;
thread producer([&] {
    for (int i = 0; i < 1000; ++i)
    {
        {
            unique_lock<mutex> lock{queue_mutex};
            queue.push_back(i);
        }
        // one must release the lock before notifying
        queue_changed.notify_all();
    } // end for
    {
        unique_lock<mutex> lock{queue_mutex};
        done = true;
    }
    queue_changed.notify_all();
});
thread consumer([&] {
    while (true)
    {
        auto maybe_data = [&]() -> boost::optional<int> { // (1)
            unique_lock<mutex> lock{queue_mutex};
            queue_changed.wait(lock,
                               [&] { return done || !queue.empty(); });
            if (!queue.empty())
            {
                auto data = move(queue[0]);
                queue.pop_front();
                return boost::make_optional(move(data));
            }
            return {};
        }(); // release lock
        // do stuff with data!
        if (maybe_data)
        {
            std::cout << *maybe_data << '\n';
        }
        else
        {
            break;
        }
    }
});
producer.join();
consumer.join();
```


#### Dask
- [dsk_ml kmean](https://github.com/dask/dask-ml/blob/master/dask_ml/cluster/k_means.py) uses [cblas_sdot](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-sdot)
- [ K-means ](https://en.wikipedia.org/wiki/K-means_clustering)
  - The running time of [Lloyd's algorithm ](http://www-evasion.imag.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/lloyd-1982.pdf)(and most variants) is O(nkdi)
  - [Scalable K-mean](https://arxiv.org/abs/1203.6402) 
  - [Partition Function](https://en.wikipedia.org/wiki/Partition_function_(mathematics))->?[ Fredholm Theory](https://en.wikipedia.org/wiki/Fredholm_theory)
  - [Green's Function](https://en.wikipedia.org/wiki/Green%27s_function)

  ### VxWorks
  -[ Basic RTOS Functions in VxWorks ](http://www.dauniv.ac.in/downloads/EmbsysRevEd_PPTs/Chap_9Lesson09EmsysNewVxWorks.pdf)
  - [ VxWorks / Tornado FAQ ](https://borkhuis.home.xs4all.nl/vxworks/vxworks.html)
-


  ### 8/2/2018
  - [ Jason Turner on Embeded.fm ](https://www.embedded.fm/episodes/247)
  - [Cassandra](http://cassandra.apache.org) 
    - [ Mastering Cassandra ](http://techbus.safaribooksonline.com/book/databases/9781782162681/firstchapter#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODE3ODIxNjI2ODElMkZjaDAybHZsMXNlYzE0X2h0bWwmcXVlcnk9)
      - CAP theorem that says to choose any two out of consistency, availability, and partition-tolerance
       - Cassandra has tunable consistency
       - NoSQL is a blanket term for the databases that solve the scalability issues that are common among relational databases. 

       
  - [skylla](https://github.com/scylladb/scylla) 
  -  [Thrift](https://thrift.apache.org)
  - [ llvm ](http://llvm.org/git/llvm)
  - [ arduino due ](https://store.arduino.cc/usa/arduino-due)
  - 


  ### 8/3/2018

  - [Predictive Model Markup Language (PMML)](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) 
  - [ Deploying Keras Models with Flask](https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2)

  #### Automated Machine Learning

  - [ AutoML ](https://www.forbes.com/sites/janakirammsv/2018/04/15/why-automl-is-set-to-become-the-future-of-artificial-intelligence/#25cec883780a)
  - [ Data Robot]()
  - [ H20 ]()
  - [ Auto Keras]()
  - [ Learning Transferable Architectures for Scalable Image Recognitinos](https://arxiv.org/abs/1707.07012)


  ### 8/6/2018
  
[ Seastar ](http://seastar.io), [OSv](http://osv.io), SoftInt , 
- std::functional, std::bind -> used to handle callbacks and such. [safari online video](http://techbus.safaribooksonline.com/video/programming/cplusplus/9781491934623)
- [std::lock_guard](https://en.cppreference.com/w/cpp/thread/lock_guard)
- [ threads (copper spice)](https://youtu.be/LNYTYVUIFXw)

#### USB emergency divergency
- [phison controllers Richard Harman](https://www.slideshare.net/xabean/controlling-usb-flash-drive-controllers-expose-of-hidden-features)
- [pyusb](http://pyusb.github.io/pyusb/)
- [libusb](http://libusb.sourceforge.net/api-1.0/libusb_api.html)
- [UnRAID LimeTech](https://lime-technology.com/application-server/)

#### Open Architecture Comparison
- [ FACE ](http)
  - Segments
    - Operating System Segment 
    - Input/Output Services Segment
    - Platform-Specific Services Segment
    - Transport Services Segment
    - Portable Components Segment
  - Interfaces
    - Operating System Segment Interface
    - I/O Services Interface
    - Transport Services Interface
- [ Open Mission Systems](https://www.vdl.afrl.af.mil/programs/uci/oms.php)
  - 

#### Java
- JEP 286: Local Variable Type Inference
- JEP 316: Heap Allocation on Alternative Memory Devices

#### OpenGL
- [ example ](https://www.opengl.org/archives/resources/code/samples/glut_examples/examples/halomagic.c)


#### Avionics
- Busses
  - ARINC 429 := Single Source Multiple Sink,  100 kbps,  32 bit words, 2 Mbps
  - ARINC 629 := (1995) Multiple Source Multiple Sink (Full Duplex)
  - MIL-STD 1553 := (1975) Multiple Source Multiple Sink (Full Duplex)
  - MIL-STD 1773

  #### Parallel Programming
  - Flynn Classification
    - SISD : stands for Single Instruction Single Data, basically it is the classical Von Neumann machine;
    - SIMD : Single instruction Multiple Data, for example CUDA is perfect for those kind of problems. Note that this classification evolved to Single Program Multiple Data i.e: signal/image/video processing, or even decryption...etc.
    - MISD : Multiple Instruction Single Data, it is a theoretical model, I do not think it really exists.
    - MIMD : Multiple Instruction Multiple Data, I think you adapt OpenMP and MPI (separately or hybrid) and keep in mind that the first one is for shared memory and second one is for distributed memory.
  - OpenMP
    - Shared Memory Model
    - Relatively easy to learn
    - [ Youtube from Tim Mattson et al ](https://www.youtube.com/watch?v=nE-xN4Bf8XI)
  - MPI
    - Message Passing 
     - Uses network communication
    - each MPI process allocates the same amount of memory
    - not fault tolerant
  - OpenCL
    - [ example ](https://developer.apple.com/library/archive/samplecode/OpenCL_Hello_World_Example/Listings/hello_c.html)
  - GPU
    - [Thrust / Cuda](https://docs.nvidia.com/cuda/thrust/index.html)
    - OpenCL
    - Cekirdekler API
    - [arrayfire]( https://arrayfire.com)
    - [ amd parallel ](http://developer.amd.com/wordpress/media/2013/07/AMD_Accelerated_Parallel_Processing_OpenCL_Programming_Guide-rev-2.7.pdf)
    - [ open CL and 13 dwarfs](http://developer.amd.com/wordpress/media/2013/06/2155_final.pdf)

  - FPGA
    - C/C++
    - MyHDL (Python)
    -  CHISEL (Scala)
    - JHDL (Java)
    - BSV  (Haskell)
    - MATLAB
    - Labview FPGA
    - SystemVerilog 
    - VHDL/Verilog
    - Spinal HDL
    - [ 10 Ways to Program your FPGA ](https://www.eetimes.com/document.asp?doc_id=1329857&page_number=1)

- Distributed ( Big Data )
  - Map Reduce
    - Hadoop
    - [BOINC](https://boinc.berkeley.edu)


- Algorithms
  -[ Dynamic Programming ](https://www.geeksforgeeks.org/dynamic-programming/)

- Data Structures
  - [HashMap](https://www.geeksforgeeks.org/java-util-hashmap-in-java/)
  

### 08/10/2018

[snowflake](https://arxiv.org/abs/1708.02579)
- (CNN) layer characteristics vary widely from one model to the next
  - Conventional Layers
  - Inception Modules
  - Residual Modules


### 8/13/2018

- [KairosDB on Scyllas](https://www.scylladb.com/webinar/on-demand-webinar-how-to-build-a-highly-available-time-series-solution-with-kairosdb/?aliId=6609490)


### 8/14/2018
- [ Deep EEG ](https://arxiv.org/pdf/1703.05051.pdf)
- [ Regulus Cyber ](https://www.regulus.com)

- [ A survey of Power and Energy Efficient Techniques for High Performance Linear Algebra ](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.709.107&rep=rep1&type=pdf)
  - CPU-bound, network- bound, memory-bound, and disk-bound), power and energy consumption on CPU dominate system power and energy costs (35%􏰂48%), and the second most power-/energy-consuming hardware component is memory (16%􏰂27%)
  - [ Energy Efficient Matix Multiplicaiton on FPGA](http://halcyon.usc.edu/~pk/prasannawebsite/papers/jangFPL02.pdf)
    - 􏰨􏰁Energy, Area, Time
  - [ Facebook Music AI ](https://medium.com/the-artificial-intelligence-journal/understanding-how-facebooks-new-ai-translates-between-music-genres-in-7-minutes-61d6cb1e5b4a)
  
  [Facebook AI](https://research.fb.com/category/facebook-ai-research/)
  [ FAIR UMTN ](https://research.fb.com/facebook-researchers-use-ai-to-turn-whistles-into-orchestral-music-and-power-other-musical-translations/)
    - [NVWaveNet]( https://github.com/NVIDIA/nv-wavenet)
     􏰉􏰂􏰃􏰣􏰕 􏰐􏰤 􏰀􏰔􏰵 􏰳􏰀􏰁􏰂􏰃􏰄􏰅􏰆􏰔

### 8/17/2018

- [ AI cheat sheet (Medium)](https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463)

### 8/20/2018

[ Travis Oliphant presentation ](https://speakerdeck.com/teoliphant/ml-in-python?slide=17)
   - [ Chainer and Extremely Large Minibatch ](https://arxiv.org/abs/1711.04325)
   - [ IDEEP, DNN-MKL ](https://github.com/intel/mkl-dnn)


[ 10 Statistical Techniques ](https://medium.com/cracking-the-data-science-interview/the-10-statistical-techniques-data-scientists-need-to-master-1ef6dbd531f7)

### 8/21/2018
[ Deep Learning Book](https://www.deeplearningbook.org/contents/prob.html)
[ Model Depot ](https://www.modeldepot.io/?source=user_profile---------------------------)
[ Hastie book ](https://web.stanford.edu/~hastie/ElemStatLearn/)
[ Darknet ](https://pjreddie.com/darknet/)
[ Joseph Redmon ](https://pjreddie.com)



### 8/23/2018
[ Intel Acceleration stack ](https://www.intel.com/content/www/us/en/programmable/documentation/iyu1522005567196.html)
[ Comparison ](https://www.nextplatform.com/2017/03/21/can-fpgas-beat-gpus-accelerating-next-generation-deep-learning/)



### 8/24/2018

[KNIME](https://www.knime.com)


#### Graph Database
[ aws neptune ](https://aws.amazon.com/neptune/)
[ Neo4j ](https://neo4j.com)



### 8/25/2018

[ Plenoptic Camera ](http://graphics.stanford.edu/papers/lfcamera/lfcamera-150dpi.pdf)
 [ Raytrix ](https://raytrix.de)
[ Artificial Neural Systems]( https://anuradhasrinivas.files.wordpress.com/2013/08/29721562-zurada-introduction-to-artificial-neural-systems-wpc-1992.pdf )
[ AWS-fpga README.md](https://github.com/aws/aws-fpga/blob/master/README.md)
  - [ ]()
    - Amazon Machine Image (AMI):= golden image, prototype
    - Amazon FPGA Image (AFI):= Amazon FPGA Image
    

### 8/28/2018
- [Xilinx SDAccel](https://www.xilinx.com/video/software/sdaccel-development-environment-demo.html)
  - [ Smith Waterman](https://github.com/Xilinx/SDAccel_Examples/tree/master/acceleration/smithwaterman)
    - [Wikipedia](https://en.wikipedia.org/wiki/Smith–Waterman_algorithm)
    - [ Genetic Sequence Alignment on a Supercomputing platform](https://repository.tudelft.nl/islandora/object/uuid%3Abbbbd3c8-7b27-4a1b-bfd6-67695eec7449)
      - [ Linear Systolic Array ]()
      - [ Convey HC-1 ]()
      - [ Intel implementaiton ](https://www.intel.com/content/dam/altera-www/global/en_US/pdfs/literature/wp/wp-01035.pdf)
      - [ Another S-W thesis ?](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=2ahUKEwja0J2Ml5HdAhVM_IMKHRBfA1cQFjABegQIBxAC&url=https%3A%2F%2Frepository.tudelft.nl%2Fislandora%2Fobject%2Fuuid%3Ac35fa6e0-e632-4f17-bf84-0f1cc8f98c0c%2Fdatastream%2FOBJ%2Fdownload&usg=AOvVaw174A7zblm2ZFMKTIA8Pgdl)
      - [ Lake Khasan ](https://www.linkedin.com/in/laiq-hasan-46270210/)
- [Advanced Scipy lecture](Full Name: 
Contact Nos.:  
Current Location: 
Open to relocate: 
Open for Travel:  
Availability:       
Work Permit:- 
Current Company:  
Current Salary : 
Annual salary expectation:  
Email:
Alternate Email-ID:- 
Skype ID :
Best time to call: 
Best Time for the Interview:- 
Total IT Experience [Yrs.]:- 
Key Skill Set:-
Looking forward to your response..)


### 8/29/2018

#### Presentation tools
- [ 4 Markdown powered slide generators ](https://opensource.com/article/18/5/markdown-slide-generators)
- [Marp](https://yhatt.github.io/marp/)
- [Adobe Spark](https://spark.adobe.com/video/SURSphvyCEhXV)
  - Great dynamic content, start with beautiful pictures.
- [ Jupyter2Slides](https://github.com/datitran/jupyter2slides)
- [ HackerSlides]()
- [ Reveal.js](https://github.com/hakimel/reveal.js/wiki/Example-Presentations)
  - Github, browswer web host required?
  - [ Slides ](https://slides.com)


#### Some good looking presentations
- [ Lemi Orhan ](https://speakerdeck.com/lemiorhan/let-the-elephants-leave-the-room-remove-waste-in-software-development?slide=7)

### 8/31/2018

[dask-tutorial](https://ww.youtube.com/watch?v=mbfsog3e5DA)
  - [ dask-tutorial github](https://github.com/dask/dask-tutorial)
  - Methods on delayed objects just work
  - Containers of delayed objects can be passed to compute
  


### 9/7/2018
#### Dask
- Arrays
  - Dask arrays supports most of the Numpy interface like the following:
    - Arithmetic and scalar mathematics, +, *, exp, log, ...  
    - Reductions along axes, sum(), mean(), std(), sum(axis=0), ...
    - Tensor contractions / dot products / matrix multiply, tensordot
    - Axis reordering / transpose, transpose
    - Slicing, x[:100, 500:100:-2]
    - Fancy indexing along single axes with lists or numpy arrays, x[:, [10, 1, 5]]
    - Array protocols like __array__, and __array_ufunc__
    - Some linear algebra svd, qr, solve, solve_triangular, lstsq
    - df Partitions should be around 100MB each: [repartition](http://dask.pydata.org/en/latest/dataframe-performance.html?highlight=partition) can adjust
    

- [ Modern Pandas ](https://tomaugspurger.github.io/modern-1-intro)
  - Any time you see back to back square brackets you are asking for trouble [][].

- [ Scikitlearn dask ](https://youtu.be/ccfsbuqsjgI)
  - Use parallel backend for random forest
      - from sklearn.internals.joblib import parallel backend
      - with parallel_backend("dask"):
  - User incremental for larger than memory
    - from daskml import incremental
  
  - [ Pangeo ](https://www.youtube.com/watch?v=mDrjGxaXQT4)
    - [ Xarray , by Stephan Hoyer ](http://xarray.pydata.org/en/stable/why-xarray.html)
    - [Pangeo homepage](http://pangeo.io)
  

  - [ Apache Arrow ](https://arrow.apache.org)
    - [ PyArrow ](https://pypi.org/project/pyarrow/)
  - [ Pytest ](https://docs.pytest.org/en/latest/)
  - [ PBS ](http://www.arc.ox.ac.uk/content/pbs-job-scheduler)
  - [ Slurm ](https://slurm.schedmd.com/overview.html)
  - [ XGBoost ](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
  - [ Apache Parquet ](http://parquet.apache.org)
  - [ Zarr ](https://zarr.readthedocs.io/en/stable/index.html)
 - [ CF Conventions ](http://cfconventions.org/Data/cf-documents/overview/viewgraphs.pdf)
    - [Nasa](https://earthdata.nasa.gov/user-resources/standards-and-references/climate-and-forecast-cf-metadata-conventions)


### 09/19/2018

  #### [Dask Futures](http://dask.pydata.org/en/latest/futures.html)
  - "```python remote_df = client.scatter(df) ```"

- [ VSCode Unit Testing ](https://code.visualstudio.com/docs/python/unit-testing)
- ```git merge --strategy-option theirs```


#### Google Cloud Services Coursera Course
- Discount based on percentage of month used
- Preemptable machine allow discounts for interrupt tolerant jobs.
- Cloud Storage is a Blob
  - Bucket like domain name gs://acme-sales/data
  - use gsutil {cp,mv,rsync,etc.}
  - REST API
  - Use Cloud Storage as a holding area
  - Zone locallity to reduce latency, distribute for redundancy & global access.s


### 09/21/2018

- [ OpenVino ](https://software.intel.com/sites/default/files/OpenVINO-Product-Brief-062718.pdf)

#### Google Cloud Services Coursera Course



#### Dask

##### ``df.map_partitions with pd.methods()``

#### What doesn't work
Dask.dataframe only covers a small but well-used portion of the Pandas API.
This limitation is for two reasons:

1.  The Pandas API is *huge*
2.  Some operations are genuinely hard to do in parallel (e.g. sort)

Additionally, some important operations like ``set_index`` work, but are slower
than in Pandas because they include substantial shuffling of data, and may write out to disk.

#### What definately works

* Trivially parallelizable operations (fast):
    *  Elementwise operations:  ``df.x + df.y``
    *  Row-wise selections:  ``df[df.x > 0]``
    *  Loc:  ``df.loc[4.0:10.5]``
    *  Common aggregations:  ``df.x.max()``
    *  Is in:  ``df[df.x.isin([1, 2, 3])]``
    *  Datetime/string accessors:  ``df.timestamp.month``
* Cleverly parallelizable operations (also fast):
    *  groupby-aggregate (with common aggregations): ``df.groupby(df.x).y.max()``
    *  value_counts:  ``df.x.value_counts``
    *  Drop duplicates:  ``df.x.drop_duplicates()``
    *  Join on index:  ``dd.merge(df1, df2, left_index=True, right_index=True)``
* Operations requiring a shuffle (slow-ish, unless on index)
    *  Set index:  ``df.set_index(df.x)``
    *  groupby-apply (with anything):  ``df.groupby(df.x).apply(myfunc)``
    *  Join not on the index:  ``pd.merge(df1, df2, on='name')``
* Ingest operations
    *  Files: ``dd.read_csv, dd.read_parquet, dd.read_json, dd.read_orc``, etc.
    *  Pandas: ``dd.from_pandas``
    *  Anything supporting numpy slicing: ``dd.from_array``
    *  From any set of functions creating sub dataframes via ``dd.from_delayed``.
    *  Dask.bag: ``mybag.to_dataframe(columns=[...])``
 
#####  ? Why is groupby().apply() slow but not groupby.max()

#### Techs
binder, django, openteam, docker-compose, xdn, chainer, [sparse](sparse.pydata.org)

[ tornado vs asyncio ](https://github.com/universe-proton/universe-topology/issues/14)

##### Google Cloud Services Coursera Course

- Google interconnects are Petabit/s...


### 9/22/2018

#### dask distributed
- ```@contextmanager
def ignoring(*exceptions):
    try:
        yield
    except exceptions as e:
        pass```
-  [bisect](https://docs.python.org/2/library/bisect.html),[contextlib](https://docs.python.org/2/library/contextlib.html),

```grep -r "^def [a-z]" *  | awk -F ":" '{print $2}'```, ```psutil.virtual_memory()```, 


### 9/24/2018

#### Google Cloud Services 	

- DataProc is Google managed Hadoop, Pig, Hive, Spark
- Storing on GCS instead of in DataProc Cluster saves $ due to decoupling of compute and storage
- 
#### Dask

- Each collection has a default scheduler
- 
- [ Python and parallelism ](http://jessenoller.com/2009/02/01/python-threads-and-the-global-interpreter-lock/))
	- A Thread is simply an agent spawned by the application to perform work independent of the parent process.
	- "Green Threads", "Native Threads"
	- Threads fundamentally differ from processes in that they are light weight and share memory.  
	- Thread based programming models don't necessarily scale well to multiple macines
	- [ Python API Reference Manual ](https://docs.python.org/3/c-api/init.html) 
```python 
from threading import Lock
from __future__ import with_statement
def synchronized():
    the_lock = Lock()
    def fwrap(function):
        def newFunction(*args, **kw):
            with the_lock:
                return function(*args, **kw)
        return newFunction
    return fwrap

...
    @synchronized()
    def transfer(self, name, afrom, ato, amount):
        if self.accounts[afrom] < amount: return
...
```
- [ pandas categorical encoder ](https://distributed.readthedocs.io/en/latest/setup.html)
- [ Pandas transformers](jorisvandenbossche.github.io/talks/2018_Scipy_sklearn_pandas)
- [ numpy dispatch ](http://www.numpy.org/neps/nep-0018-array-function-protocol.html)
- parallel covariance
	- [Green et. all Technion](https://arxiv.org/pdf/1303.2285.pdf)
- [Helm](https://helm.sh)
- [joblib](https://joblib.readthedocs.io/en/latest/)
- [ tall skinny SVD ](https://arxiv.org/abs/1301.1071)
- [ Dummy coding is the process of coding a categorical variable into dichotomous variables (one-hot)](https://en.wikiversity.org/wiki/Dummy_variable_(statistics))
	- the number of dummy-coded variables needed is one less than the number of categories
- [ Async/await in Python 3.4 ](https://snarky.ca/how-the-heck-does-async-await-work-in-python-3-5/) 
	- Asynchronous programming is basically programming where execution order is not known ahead of time (hence asynchronous instead of synchronous).
	-  Concurrent programming is writing code to execute independently of other parts, even if it all executes in a single thread (concurrency is **not** parallelism). 
```python 

import asyncio

# Borrowed from http://curio.readthedocs.org/en/latest/tutorial.html.
@asyncio.coroutine
def countdown(number, n):
    while n > 0:
        print('T-minus', n, '({})'.format(number))
        yield from asyncio.sleep(1)
        n -= 1

loop = asyncio.get_event_loop()
tasks = [
    asyncio.ensure_future(countdown("A", 2)),
    asyncio.ensure_future(countdown("B", 3))]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
```
- from dask-tutorial
```python
def random_array():
    if os.path.exists(os.path.join(data_dir, 'random.hdf5')):
        return

    print("Create random data for array exercise")
    import h5py

    with h5py.File(os.path.join(data_dir, 'random.hdf5')) as f:
        dset = f.create_dataset('/x', shape=(1000000000,), dtype='f4')
        for i in range(0, 1000000000, 1000000):
            dset[i: i + 1000000] = np.random.exponential(size=1000000)a
```

### 9/27/2018

- [ XGBoost ](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
- [ Dask XGboost ](http://matthewrocklin.com/blog/work/2017/03/28/dask-xgboost) 


- [ Grip! ](https://github.com/joeyespo/grip)

- [ Linux memory management ](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/performance_tuning_guide/sect-red_hat_enterprise_linux-performance_tuning_guide-configuration_tools-configuring_system_memory_capacity)
- [ Directory listing and stuff](https://www.saltycrane.com/blog/2010/04/options-listing-files-directory-python/)
- [ Azure swap ]()
- [ Rodney Brooks on Artificial Intelligence ](http://www.econtalk.org/rodney-brooks-on-artificial-intelligence/)

### 

#### Matrix Partition Empathy
- [ Matrix Partition ](https://www.cs.utexas.edu/users/plapack/papers/ipps98/ipps98.html)
- [ HAL Id ](https://hal.inria.fr/hal-01670672/document)
[Advances in Parallel Partitioning, Load Balancing and Matrix Ordering for Scientific Computin](https://cscapes.cs.purdue.edu/pub/Boman-SciDAC09.pdf)
##### Einstein notation, loop order, real estate 
##### Scatter, optimize <-> search , Gather, Merge, push,  	

### Open Source Directions

- Sympy
	- Alternatives: Maple, Mathematica,  
	- [ AppSec ](https://en.wikipedia.org/wiki/Application_security)
- [PyTheory](https://github.com/kennethreitz/pytheory?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)


#### Google Cloud Platform
- Intro to Scaling Data Analysis
	- [!DataStore is like a persistant HashMap](file://./screenshots/Screen Shot 2018-09-28 at 9.25.13 AM.png)
	- [!Crud Operations are easily implemented in Datastore](file://./screenshots/Screen Shot 2018-09-28 at 9.47.00 PM.png)
	- [!Choose Storage Option based on Usage Pattern](file://./screenshots/Screen Shot 2018-09-28 at 9.58.43 PM.png)
		- Cloud Storage: File System
		- Cloud SQL: Relational
		- Datastore: Hierarchical
		- Bigtable: High Throughput
			- search only based on key
			- HBASE API
		- [BigQuery](bigquery.cloud.google.com)
			- SQL queries on Petabytes
			- Load data
				- Files on disc or Cloud Storage
				- Stream Data: POST
				- Federated Data Source: CSV, JSON, AVRO, Google Sheets (**e.g. join sheets and Bigquery**)
			- DataLab open-source notebook
				- datalab create my-datalab-vm --machine-type n1-highmem-8 --zone us-central1-a
				- [ gcloud install](https://cloud.google.com/sdk/docs/quickstart-macos) 
				- datalab supports BigQuery
				 
	- Lab
```Python		
import shutil
%bq tables describe --name bigquery-public-data.new_york.tlc_yellow_trips_2015
%bq query -n taxiquery

WITH trips AS (
  SELECT EXTRACT (DAYOFYEAR from pickup_datetime) AS daynumber 
  FROM `bigquery-public-data.new_york.tlc_yellow_trips_*`
  where _TABLE_SUFFIX = @YEAR
)
SELECT daynumber, COUNT(1) AS numtrips FROM trips
GROUP BY daynumber ORDER BY daynumber
query_parameters = [
  {
    'name': 'YEAR',
    'parameterType': {'type': 'STRING'},
    'parameterValue': {'value': 2015}
  }
]
trips = taxiquery.execute(query_params=query_parameters).result().to_dataframe()
trips[:5]
avg = np.mean(trips['numtrips'])
print('Just using average={0} has RMSE of {1}'.format(avg, np.sqrt(np.mean((trips['numtrips'] - avg)**2))))
%bq query
SELECT * FROM `bigquery-public-data.noaa_gsod.stations`
WHERE state = 'NY' AND wban != '99999' AND name LIKE '%LA GUARDIA%'
%bq query -n wxquery
SELECT EXTRACT (DAYOFYEAR FROM CAST(CONCAT(@YEAR,'-',mo,'-',da) AS TIMESTAMP)) AS daynumber,
       MIN(EXTRACT (DAYOFWEEK FROM CAST(CONCAT(@YEAR,'-',mo,'-',da) AS TIMESTAMP))) dayofweek,
       MIN(min) mintemp, MAX(max) maxtemp, MAX(IF(prcp=99.99,0,prcp)) rain
FROM `bigquery-public-data.noaa_gsod.gsod*`
WHERE stn='725030' AND _TABLE_SUFFIX = @YEAR
GROUP BY 1 ORDER BY daynumber DESC
query_parameters = [
  {
    'name': 'YEAR',
    'parameterType': {'type': 'STRING'},
    'parameterValue': {'value': 2015}
  }
]
weather = wxquery.execute(query_params=query_parameters).result().to_dataframe()
weather[:5]
data = pd.merge(weather, trips, on='daynumber')
data[:5]
j = data.plot(kind='scatter', x='maxtemp', y='numtrips')
j = data.plot(kind='scatter', x='dayofweek', y='numtrips')
j = data[data['dayofweek'] == 7].plot(kind='scatter', x='maxtemp', y='numtrips')
data2 = data # 2015 data
for year in [2014, 2016]:
    query_parameters = [
      {
        'name': 'YEAR',
        'parameterType': {'type': 'STRING'},
        'parameterValue': {'value': year}
      }
    ]
    weather = wxquery.execute(query_params=query_parameters).result().to_dataframe()
    trips = taxiquery.execute(query_params=query_parameters).result().to_dataframe()
    data_for_year = pd.merge(weather, trips, on='daynumber')
    data2 = pd.concat([data2, data_for_year])
data2.describe()
j = data2[data2['dayofweek'] == 7].plot(kind='scatter', x='maxtemp', y='numtrips')
import tensorflow as tf
shuffled = data2.sample(frac=1, random_state=13)
# It would be a good idea, if we had more data, to treat the days as categorical variables
# with the small amount of data, we have though, the model tends to overfit
#predictors = shuffled.iloc[:,2:5]
#for day in range(1,8):
#  matching = shuffled['dayofweek'] == day
#  key = 'day_' + str(day)
#  predictors[key] = pd.Series(matching, index=predictors.index, dtype=float)
predictors = shuffled.iloc[:,1:5]
predictors[:5]
shuffled[:5]
targets = shuffled.iloc[:,5]
targets[:5]
 Scedule C gets audited more often
- S corp
  + Income tax paid by individual share holders
- C corp Corporate tax rate is reduced to a maximum rate of 21%
  + Tax Cuts and jobs act of 2017
  + 20% deduction of "qualified business income" 

#### Dask
- [dsk_ml kmean](https://github.com/dask/dask-ml/blob/master/dask_ml/cluster/k_means.py) uses [cblas_sdot](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-sdot)
- [ K-means ](https://en.wikipedia.org/wiki/K-means_clustering)
  - The running time of [Lloyd's algorithm ](http://www-evasion.imag.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/lloyd-1982.pdf)(and most variants) is O(nkdi)
  - [Scalable K-mean](https://arxiv.org/abs/1203.6402) 
  - [Partition Function](https://en.wikipedia.org/wiki/Partition_function_(mathematics))->?[ Fredholm Theory](https://en.wikipedia.org/wiki/Fredholm_theory)
  - [Green's Function](https://en.wikipedia.org/wiki/Green%27s_function)

### VxWorks
  -[ Basic RTOS Functions in VxWorks ](http://www.dauniv.ac.in/downloads/EmbsysRevEd_PPTs/Chap_9Lesson09EmsysNewVxWorks.pdf)
  - [ VxWorks / Tornado FAQ ](https://borkhuis.home.xs4all.nl/vxworks/vxworks.html)
-


  ### 8/2/2018
  - [ Jason Turner on Embeded.fm ](https://www.embedded.fm/episodes/247)
  - [Cassandra](http://cassandra.apache.org) 
    - [ Mastering Cassandra ](http://techbus.safaribooksonline.com/book/databases/9781782162681/firstchapter#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODE3ODIxNjI2ODElMkZjaDAybHZsMXNlYzE0X2h0bWwmcXVlcnk9)
      - CAP theorem that says to choose any two out of consistency, availability, and partition-tolerance
       - Cassandra has tunable consistency
       - NoSQL is a blanket term for the databases that solve the scalability issues that are common among relational databases. 

       
  - [skylla](https://github.com/scylladb/scylla) 
  -  [Thrift](https://thrift.apache.org)
  - [ llvm ](http://llvm.org/git/llvm)
  - [ arduino due ](https://store.arduino.cc/usa/arduino-due)
  - 


  ### 8/3/2018

  - [Predictive Model Markup Language (PMML)](https://en.wikipedia.org/wiki/Predictive_Model_Markup_Language) 
  - [ Deploying Keras Models with Flask](https://towardsdatascience.com/deploying-keras-deep-learning-models-with-flask-5da4181436a2)

  #### Automated Machine Learning

  - [ AutoML ](https://www.forbes.com/sites/janakirammsv/2018/04/15/why-automl-is-set-to-become-the-future-of-artificial-intelligence/#25cec883780a)
  - [ Data Robot]()
  - [ H20 ]()
  - [ Auto Keras]()
  - [ Learning Transferable Architectures for Scalable Image Recognitinos](https://arxiv.org/abs/1707.07012)


  ### 8/6/2018
  
[ Seastar ](http://seastar.io), [OSv](http://osv.io), SoftInt , 
- std::functional, std::bind -> used to handle callbacks and such. [safari online video](http://techbus.safaribooksonline.com/video/programming/cplusplus/9781491934623)
- [std::lock_guard](https://en.cppreference.com/w/cpp/thread/lock_guard)
- [ threads (copper spice)](https://youtu.be/LNYTYVUIFXw)

#### USB emergency divergency
- [phison controllers Richard Harman](https://www.slideshare.net/xabean/controlling-usb-flash-drive-controllers-expose-of-hidden-features)
- [pyusb](http://pyusb.github.io/pyusb/)
- [libusb](http://libusb.sourceforge.net/api-1.0/libusb_api.html)
- [UnRAID LimeTech](https://lime-technology.com/application-server/)

#### Open Architecture Comparison
- [ FACE ](http)
  - Segments
    - Operating System Segment 
    - Input/Output Services Segment
    - Platform-Specific Services Segment
    - Transport Services Segment
    - Portable Components Segment
  - Interfaces
    - Operating System Segment Interface
    - I/O Services Interface
    - Transport Services Interface
- [ Open Mission Systems](https://www.vdl.afrl.af.mil/programs/uci/oms.php)
  - 

#### Java
- JEP 286: Local Variable Type Inference
- JEP 316: Heap Allocation on Alternative Memory Devices

#### OpenGL
- [ example ](https://www.opengl.org/archives/resources/code/samples/glut_examples/examples/halomagic.c)


#### Avionics
- Busses
  - ARINC 429 := Single Source Multiple Sink,  100 kbps,  32 bit words, 2 Mbps
  - ARINC 629 := (1995) Multiple Source Multiple Sink (Full Duplex)
  - MIL-STD 1553 := (1975) Multiple Source Multiple Sink (Full Duplex)
  - MIL-STD 1773

  #### Parallel Programming
  - Flynn Classification
    - SISD : stands for Single Instruction Single Data, basically it is the classical Von Neumann machine;
    - SIMD : Single instruction Multiple Data, for example CUDA is perfect for those kind of problems. Note that this classification evolved to Single Program Multiple Data i.e: signal/image/video processing, or even decryption...etc.
    - MISD : Multiple Instruction Single Data, it is a theoretical model, I do not think it really exists.
    - MIMD : Multiple Instruction Multiple Data, I think you adapt OpenMP and MPI (separately or hybrid) and keep in mind that the first one is for shared memory and second one is for distributed memory.
  - OpenMP
    - Shared Memory Model
    - Relatively easy to learn
    - [ Youtube from Tim Mattson et al ](https://www.youtube.com/watch?v=nE-xN4Bf8XI)
  - MPI
    - Message Passing 
     - Uses network communication
    - each MPI process allocates the same amount of memory
    - not fault tolerant
  - OpenCL
    - [ example ](https://developer.apple.com/library/archive/samplecode/OpenCL_Hello_World_Example/Listings/hello_c.html)
  - GPU
    - [Thrust / Cuda](https://docs.nvidia.com/cuda/thrust/index.html)
    - OpenCL
    - Cekirdekler API
    - [arrayfire]( https://arrayfire.com)
    - [ amd parallel ](http://developer.amd.com/wordpress/media/2013/07/AMD_Accelerated_Parallel_Processing_OpenCL_Programming_Guide-rev-2.7.pdf)
    - [ open CL and 13 dwarfs](http://developer.amd.com/wordpress/media/2013/06/2155_final.pdf)

  - FPGA
    - C/C++
    - MyHDL (Python)
    -  CHISEL (Scala)
    - JHDL (Java)
    - BSV  (Haskell)
    - MATLAB
    - Labview FPGA
    - SystemVerilog 
    - VHDL/Verilog
    - Spinal HDL
    - [ 10 Ways to Program your FPGA ](https://www.eetimes.com/document.asp?doc_id=1329857&page_number=1)

- Distributed ( Big Data )
  - Map Reduce
    - Hadoop
    - [BOINC](https://boinc.berkeley.edu)


- Algorithms
  -[ Dynamic Programming ](https://www.geeksforgeeks.org/dynamic-programming/)

- Data Structures
  - [HashMap](https://www.geeksforgeeks.org/java-util-hashmap-in-java/)
  

### 08/10/2018

[snowflake](https://arxiv.org/abs/1708.02579)
- (CNN) layer characteristics vary widely from one model to the next
  - Conventional Layers
  - Inception Modules
  - Residual Modules


### 8/13/2018

- [KairosDB on Scyllas](https://www.scylladb.com/webinar/on-demand-webinar-how-to-build-a-highly-available-time-series-solution-with-kairosdb/?aliId=6609490)


### 8/14/2018
- [ Deep EEG ](https://arxiv.org/pdf/1703.05051.pdf)
- [ Regulus Cyber ](https://www.regulus.com)

- [ A survey of Power and Energy Efficient Techniques for High Performance Linear Algebra ](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.709.107&rep=rep1&type=pdf)
  - CPU-bound, network- bound, memory-bound, and disk-bound), power and energy consumption on CPU dominate system power and energy costs (35%􏰂48%), and the second most power-/energy-consuming hardware component is memory (16%􏰂27%)
  - [ Energy Efficient Matix Multiplicaiton on FPGA](http://halcyon.usc.edu/~pk/prasannawebsite/papers/jangFPL02.pdf)
    - 􏰨􏰁Energy, Area, Time
  - [ Facebook Music AI ](https://medium.com/the-artificial-intelligence-journal/understanding-how-facebooks-new-ai-translates-between-music-genres-in-7-minutes-61d6cb1e5b4a)
  
  [Facebook AI](https://research.fb.com/category/facebook-ai-research/)
  [ FAIR UMTN ](https://research.fb.com/facebook-researchers-use-ai-to-turn-whistles-into-orchestral-music-and-power-other-musical-translations/)
    - [NVWaveNet]( https://github.com/NVIDIA/nv-wavenet)
     􏰉􏰂􏰃􏰣􏰕 􏰐􏰤 􏰀􏰔􏰵 􏰳􏰀􏰁􏰂􏰃􏰄􏰅􏰆􏰔

### 8/17/2018

- [ AI cheat sheet (Medium)](https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463)

### 8/20/2018

[ Travis Oliphant presentation ](https://speakerdeck.com/teoliphant/ml-in-python?slide=17)
   - [ Chainer and Extremely Large Minibatch ](https://arxiv.org/abs/1711.04325)
   - [ IDEEP, DNN-MKL ](https://github.com/intel/mkl-dnn)


[ 10 Statistical Techniques ](https://medium.com/cracking-the-data-science-interview/the-10-statistical-techniques-data-scientists-need-to-master-1ef6dbd531f7)

### 8/21/2018
[ Deep Learning Book](https://www.deeplearningbook.org/contents/prob.html)
[ Model Depot ](https://www.modeldepot.io/?source=user_profile---------------------------)
[ Hastie book ](https://web.stanford.edu/~hastie/ElemStatLearn/)
[ Darknet ](https://pjreddie.com/darknet/)
[ Joseph Redmon ](https://pjreddie.com)



### 8/23/2018
[ Intel Acceleration stack ](https://www.intel.com/content/www/us/en/programmable/documentation/iyu1522005567196.html)
[ Comparison ](https://www.nextplatform.com/2017/03/21/can-fpgas-beat-gpus-accelerating-next-generation-deep-learning/)



### 8/24/2018

[KNIME](https://www.knime.com)


#### Graph Database
[ aws neptune ](https://aws.amazon.com/neptune/)
[ Neo4j ](https://neo4j.com)



### 8/25/2018

[ Plenoptic Camera ](http://graphics.stanford.edu/papers/lfcamera/lfcamera-150dpi.pdf)
 [ Raytrix ](https://raytrix.de)
[ Artificial Neural Systems]( https://anuradhasrinivas.files.wordpress.com/2013/08/29721562-zurada-introduction-to-artificial-neural-systems-wpc-1992.pdf )
[ AWS-fpga README.md](https://github.com/aws/aws-fpga/blob/master/README.md)
  - [ ]()
    - Amazon Machine Image (AMI):= golden image, prototype
    - Amazon FPGA Image (AFI):= Amazon FPGA Image
    

### 8/28/2018
- [Xilinx SDAccel](https://www.xilinx.com/video/software/sdaccel-development-environment-demo.html)
  - [ Smith Waterman](https://github.com/Xilinx/SDAccel_Examples/tree/master/acceleration/smithwaterman)
    - [Wikipedia](https://en.wikipedia.org/wiki/Smith–Waterman_algorithm)
    - [ Genetic Sequence Alignment on a Supercomputing platform](https://repository.tudelft.nl/islandora/object/uuid%3Abbbbd3c8-7b27-4a1b-bfd6-67695eec7449)
      - [ Linear Systolic Array ]()
      - [ Convey HC-1 ]()
      - [ Intel implementaiton ](https://www.intel.com/content/dam/altera-www/global/en_US/pdfs/literature/wp/wp-01035.pdf)
      - [ Another S-W thesis ?](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=2ahUKEwja0J2Ml5HdAhVM_IMKHRBfA1cQFjABegQIBxAC&url=https%3A%2F%2Frepository.tudelft.nl%2Fislandora%2Fobject%2Fuuid%3Ac35fa6e0-e632-4f17-bf84-0f1cc8f98c0c%2Fdatastream%2FOBJ%2Fdownload&usg=AOvVaw174A7zblm2ZFMKTIA8Pgdl)
      - [ Lake Khasan ](https://www.linkedin.com/in/laiq-hasan-46270210/)

### 8/29/2018

#### Presentation tools
- [ 4 Markdown powered slide generators ](https://opensource.com/article/18/5/markdown-slide-generators)
- [Marp](https://yhatt.github.io/marp/)
- [Adobe Spark](https://spark.adobe.com/video/SURSphvyCEhXV)
  - Great dynamic content, start with beautiful pictures.
- [ Jupyter2Slides](https://github.com/datitran/jupyter2slides)
- [ HackerSlides]()
- [ Reveal.js](https://github.com/hakimel/reveal.js/wiki/Example-Presentations)
  - Github, browswer web host required?
  - [ Slides ](https://slides.com)


#### Some good looking presentations
- [ Lemi Orhan ](https://speakerdeck.com/lemiorhan/let-the-elephants-leave-the-room-remove-waste-in-software-development?slide=7)

### 8/31/2018

[dask-tutorial](https://ww.youtube.com/watch?v=mbfsog3e5DA)
  - [ dask-tutorial github](https://github.com/dask/dask-tutorial)
  - Methods on delayed objects just work
  - Containers of delayed objects can be passed to compute
  


### 9/7/2018
#### Dask
- Arrays
  - Dask arrays supports most of the Numpy interface like the following:
    - Arithmetic and scalar mathematics, +, *, exp, log, ...  
    - Reductions along axes, sum(), mean(), std(), sum(axis=0), ...
    - Tensor contractions / dot products / matrix multiply, tensordot
    - Axis reordering / transpose, transpose
    - Slicing, x[:100, 500:100:-2]
    - Fancy indexing along single axes with lists or numpy arrays, x[:, [10, 1, 5]]
    - Array protocols like __array__, and __array_ufunc__
    - Some linear algebra svd, qr, solve, solve_triangular, lstsq
    - df Partitions should be around 100MB each: [repartition](http://dask.pydata.org/en/latest/dataframe-performance.html?highlight=partition) can adjust
    

- [ Modern Pandas ](https://tomaugspurger.github.io/modern-1-intro)
  - Any time you see back to back square brackets you are asking for trouble [][].

- [ Scikitlearn dask ](https://youtu.be/ccfsbuqsjgI)
  - Use parallel backend for random forest
      - from sklearn.internals.joblib import parallel backend
      - with parallel_backend("dask"):
  - User incremental for larger than memory
    - from daskml import incremental
  
  - [ Pangeo ](https://www.youtube.com/watch?v=mDrjGxaXQT4)
    - [ Xarray , by Stephan Hoyer ](http://xarray.pydata.org/en/stable/why-xarray.html)
    - [Pangeo homepage](http://pangeo.io)
  

  - [ Apache Arrow ](https://arrow.apache.org)
    - [ PyArrow ](https://pypi.org/project/pyarrow/)
  - [ Pytest ](https://docs.pytest.org/en/latest/)
  - [ PBS ](http://www.arc.ox.ac.uk/content/pbs-job-scheduler)
  - [ Slurm ](https://slurm.schedmd.com/overview.html)
  - [ XGBoost ](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
  - [ Apache Parquet ](http://parquet.apache.org)
  - [ Zarr ](https://zarr.readthedocs.io/en/stable/index.html)
 - [ CF Conventions ](http://cfconventions.org/Data/cf-documents/overview/viewgraphs.pdf)
    - [Nasa](https://earthdata.nasa.gov/user-resources/standards-and-references/climate-and-forecast-cf-metadata-conventions)


### 09/19/2018

  #### [Dask Futures](http://dask.pydata.org/en/latest/futures.html)
  - "```python remote_df = client.scatter(df) ```"

- [ VSCode Unit Testing ](https://code.visualstudio.com/docs/python/unit-testing)
- ```git merge --strategy-option theirs```


#### Google Cloud Services Coursera Course
- Discount based on percentage of month used
- Preemptable machine allow discounts for interrupt tolerant jobs.
- Cloud Storage is a Blob
  - Bucket like domain name gs://acme-sales/data
  - use gsutil {cp,mv,rsync,etc.}
  - REST API
  - Use Cloud Storage as a holding area
  - Zone locallity to reduce latency, distribute for redundancy & global access.s


### 09/21/2018

- [ OpenVino ](https://software.intel.com/sites/default/files/OpenVINO-Product-Brief-062718.pdf)

#### Google Cloud Services Coursera Course



#### Dask

##### ``df.map_partitions with pd.methods()``

#### What doesn't work
Dask.dataframe only covers a small but well-used portion of the Pandas API.
This limitation is for two reasons:

1.  The Pandas API is *huge*
2.  Some operations are genuinely hard to do in parallel (e.g. sort)

Additionally, some important operations like ``set_index`` work, but are slower
than in Pandas because they include substantial shuffling of data, and may write out to disk.

#### What definately works

* Trivially parallelizable operations (fast):
    *  Elementwise operations:  ``df.x + df.y``
    *  Row-wise selections:  ``df[df.x > 0]``
    *  Loc:  ``df.loc[4.0:10.5]``
    *  Common aggregations:  ``df.x.max()``
    *  Is in:  ``df[df.x.isin([1, 2, 3])]``
    *  Datetime/string accessors:  ``df.timestamp.month``
* Cleverly parallelizable operations (also fast):
    *  groupby-aggregate (with common aggregations): ``df.groupby(df.x).y.max()``
    *  value_counts:  ``df.x.value_counts``
    *  Drop duplicates:  ``df.x.drop_duplicates()``
    *  Join on index:  ``dd.merge(df1, df2, left_index=True, right_index=True)``
* Operations requiring a shuffle (slow-ish, unless on index)
    *  Set index:  ``df.set_index(df.x)``
    *  groupby-apply (with anything):  ``df.groupby(df.x).apply(myfunc)``
    *  Join not on the index:  ``pd.merge(df1, df2, on='name')``
* Ingest operations
    *  Files: ``dd.read_csv, dd.read_parquet, dd.read_json, dd.read_orc``, etc.
    *  Pandas: ``dd.from_pandas``
    *  Anything supporting numpy slicing: ``dd.from_array``
    *  From any set of functions creating sub dataframes via ``dd.from_delayed``.
    *  Dask.bag: ``mybag.to_dataframe(columns=[...])``
 
#####  ? Why is groupby().apply() slow but not groupby.max()

#### Techs
binder, django, openteam, docker-compose, xdn, chainer, [sparse](sparse.pydata.org)

[ tornado vs asyncio ](https://github.com/universe-proton/universe-topology/issues/14)

##### Google Cloud Services Coursera Course

- Google interconnects are Petabit/s...


### 9/22/2018

#### dask distributed
- ```@contextmanager
def ignoring(*exceptions):
    try:
        yield
    except exceptions as e:
        pass```
-  [bisect](https://docs.python.org/2/library/bisect.html),[contextlib](https://docs.python.org/2/library/contextlib.html),

```grep -r "^def [a-z]" *  | awk -F ":" '{print $2}'```, ```psutil.virtual_memory()```, 


### 9/24/2018

#### Google Cloud Services 	

- DataProc is Google managed Hadoop, Pig, Hive, Spark
- Storing on GCS instead of in DataProc Cluster saves $ due to decoupling of compute and storage
- 
#### Dask

- Each collection has a default scheduler
- 
- [ Python and parallelism ](http://jessenoller.com/2009/02/01/python-threads-and-the-global-interpreter-lock/))
	- A Thread is simply an agent spawned by the application to perform work independent of the parent process.
	- "Green Threads", "Native Threads"
	- Threads fundamentally differ from processes in that they are light weight and share memory.  
	- Thread based programming models don't necessarily scale well to multiple macines
	- [ Python API Reference Manual ](https://docs.python.org/3/c-api/init.html) 
```python 
from threading import Lock
from __future__ import with_statement
def synchronized():
    the_lock = Lock()
    def fwrap(function):
        def newFunction(*args, **kw):
            with the_lock:
                return function(*args, **kw)
        return newFunction
    return fwrap

...
    @synchronized()
    def transfer(self, name, afrom, ato, amount):
        if self.accounts[afrom] < amount: return
...
```
- [ pandas categorical encoder ](https://distributed.readthedocs.io/en/latest/setup.html)
- [ Pandas transformers](jorisvandenbossche.github.io/talks/2018_Scipy_sklearn_pandas)
- [ numpy dispatch ](http://www.numpy.org/neps/nep-0018-array-function-protocol.html)
- parallel covariance
	- [Green et. all Technion](https://arxiv.org/pdf/1303.2285.pdf)
- [Helm](https://helm.sh)
- [joblib](https://joblib.readthedocs.io/en/latest/)
- [ tall skinny SVD ](https://arxiv.org/abs/1301.1071)
- [ Dummy coding is the process of coding a categorical variable into dichotomous variables (one-hot)](https://en.wikiversity.org/wiki/Dummy_variable_(statistics))
	- the number of dummy-coded variables needed is one less than the number of categories
- [ Async/await in Python 3.4 ](https://snarky.ca/how-the-heck-does-async-await-work-in-python-3-5/) 
	- Asynchronous programming is basically programming where execution order is not known ahead of time (hence asynchronous instead of synchronous).
	-  Concurrent programming is writing code to execute independently of other parts, even if it all executes in a single thread (concurrency is **not** parallelism). 
```python 

import asyncio

# Borrowed from http://curio.readthedocs.org/en/latest/tutorial.html.
@asyncio.coroutine
def countdown(number, n):
    while n > 0:
        print('T-minus', n, '({})'.format(number))
        yield from asyncio.sleep(1)
        n -= 1

loop = asyncio.get_event_loop()
tasks = [
    asyncio.ensure_future(countdown("A", 2)),
    asyncio.ensure_future(countdown("B", 3))]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()
```
- from dask-tutorial
```python
def random_array():
    if os.path.exists(os.path.join(data_dir, 'random.hdf5')):
        return

    print("Create random data for array exercise")
    import h5py

    with h5py.File(os.path.join(data_dir, 'random.hdf5')) as f:
        dset = f.create_dataset('/x', shape=(1000000000,), dtype='f4')
        for i in range(0, 1000000000, 1000000):
            dset[i: i + 1000000] = np.random.exponential(size=1000000)a
```

### 9/27/2018

- [ XGBoost ](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)
- [ Dask XGboost ](http://matthewrocklin.com/blog/work/2017/03/28/dask-xgboost) 


- [ Grip! ](https://github.com/joeyespo/grip)

- [ Linux memory management ](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/performance_tuning_guide/sect-red_hat_enterprise_linux-performance_tuning_guide-configuration_tools-configuring_system_memory_capacity)
- [ Directory listing and stuff](https://www.saltycrane.com/blog/2010/04/options-listing-files-directory-python/)
- [ Azure swap ]()
- [ Rodney Brooks on Artificial Intelligence ](http://www.econtalk.org/rodney-brooks-on-artificial-intelligence/)

### 

#### Matrix Partition Empathy
- [ Matrix Partition ](https://www.cs.utexas.edu/users/plapack/papers/ipps98/ipps98.html)
- [ HAL Id ](https://hal.inria.fr/hal-01670672/document)
[Advances in Parallel Partitioning, Load Balancing and Matrix Ordering for Scientific Computin](https://cscapes.cs.purdue.edu/pub/Boman-SciDAC09.pdf)
##### Einstein notation, loop order, real estate 
##### Scatter, optimize <-> search , Gather, Merge, push,  	

### Open Source Directions

- Sympy
	- Alternatives: Maple, Mathematica,  
	- [ AppSec ](https://en.wikipedia.org/wiki/Application_security)
- [PyTheory](https://github.com/kennethreitz/pytheory?utm_source=mybridge&utm_medium=blog&utm_campaign=read_more)


#### Google Cloud Platform
- Intro to Scaling Data Analysis
	- [!DataStore is like a persistant HashMap](file://./screenshots/Screen Shot 2018-09-28 at 9.25.13 AM.png)
	- [!Crud Operations are easily implemented in Datastore](file://./screenshots/Screen Shot 2018-09-28 at 9.47.00 PM.png)
	- [!Choose Storage Option based on Usage Pattern](file://./screenshots/Screen Shot 2018-09-28 at 9.58.43 PM.png)
		- Cloud Storage: File System
		- Cloud SQL: Relational
		- Datastore: Hierarchical
		- Bigtable: High Throughput
			- search only based on key
			- HBASE API
		- [BigQuery](bigquery.cloud.google.com)
			- SQL queries on Petabytes
			- Load data
				- Files on disc or Cloud Storage
				- Stream Data: POST
				- Federated Data Source: CSV, JSON, AVRO, Google Sheets (**e.g. join sheets and Bigquery**)
			- DataLab open-source notebook
				- datalab create my-datalab-vm --machine-type n1-highmem-8 --zone us-central1-a
				- [ gcloud install](https://cloud.google.com/sdk/docs/quickstart-macos) 
				- datalab supports BigQuery
				 
	- Lab
```Python		
import shutil
%bq tables describe --name bigquery-public-data.new_york.tlc_yellow_trips_2015
%bq query -n taxiquery

WITH trips AS (
  SELECT EXTRACT (DAYOFYEAR from pickup_datetime) AS daynumber 
  FROM `bigquery-public-data.new_york.tlc_yellow_trips_*`
  where _TABLE_SUFFIX = @YEAR
)
SELECT daynumber, COUNT(1) AS numtrips FROM trips
GROUP BY daynumber ORDER BY daynumber
query_parameters = [
  {
    'name': 'YEAR',
    'parameterType': {'type': 'STRING'},
    'parameterValue': {'value': 2015}
  }
]
trips = taxiquery.execute(query_params=query_parameters).result().to_dataframe()
trips[:5]
avg = np.mean(trips['numtrips'])
print('Just using average={0} has RMSE of {1}'.format(avg, np.sqrt(np.mean((trips['numtrips'] - avg)**2))))
%bq query
SELECT * FROM `bigquery-public-data.noaa_gsod.stations`
WHERE state = 'NY' AND wban != '99999' AND name LIKE '%LA GUARDIA%'
%bq query -n wxquery
SELECT EXTRACT (DAYOFYEAR FROM CAST(CONCAT(@YEAR,'-',mo,'-',da) AS TIMESTAMP)) AS daynumber,
       MIN(EXTRACT (DAYOFWEEK FROM CAST(CONCAT(@YEAR,'-',mo,'-',da) AS TIMESTAMP))) dayofweek,
       MIN(min) mintemp, MAX(max) maxtemp, MAX(IF(prcp=99.99,0,prcp)) rain
FROM `bigquery-public-data.noaa_gsod.gsod*`
WHERE stn='725030' AND _TABLE_SUFFIX = @YEAR
GROUP BY 1 ORDER BY daynumber DESC
query_parameters = [
  {
    'name': 'YEAR',
    'parameterType': {'type': 'STRING'},
    'parameterValue': {'value': 2015}
  }
]
weather = wxquery.execute(query_params=query_parameters).result().to_dataframe()
weather[:5]
data = pd.merge(weather, trips, on='daynumber')
data[:5]
j = data.plot(kind='scatter', x='maxtemp', y='numtrips')
j = data.plot(kind='scatter', x='dayofweek', y='numtrips')
j = data[data['dayofweek'] == 7].plot(kind='scatter', x='maxtemp', y='numtrips')
data2 = data # 2015 data
for year in [2014, 2016]:
    query_parameters = [
      {
        'name': 'YEAR',
        'parameterType': {'type': 'STRING'},
        'parameterValue': {'value': year}
      }
    ]
    weather = wxquery.execute(query_params=query_parameters).result().to_dataframe()
    trips = taxiquery.execute(query_params=query_parameters).result().to_dataframe()
    data_for_year = pd.merge(weather, trips, on='daynumber')
    data2 = pd.concat([data2, data_for_year])
data2.describe()
j = data2[data2['dayofweek'] == 7].plot(kind='scatter', x='maxtemp', y='numtrips')
import tensorflow as tf
shuffled = data2.sample(frac=1, random_state=13)
# It would be a good idea, if we had more data, to treat the days as categorical variables
# with the small amount of data, we have though, the model tends to overfit
#predictors = shuffled.iloc[:,2:5]
#for day in range(1,8):
#  matching = shuffled['dayofweek'] == day
#  key = 'day_' + str(day)
#  predictors[key] = pd.Series(matching, index=predictors.index, dtype=float)
predictors = shuffled.iloc[:,1:5]
predictors[:5]
shuffled[:5]
targets = shuffled.iloc[:,5]
targets[:5]
trainsize = int(len(shuffled['numtrips']) * 0.8)
avg = np.mean(shuffled['numtrips'][:trainsize])
rmse = np.sqrt(np.mean((targets[trainsize:] - avg)**2))
print('Just using average={0} has RMSE of {1}'.format(avg, rmse))
SCALE_NUM_TRIPS = 600000.0
trainsize = int(len(shuffled['numtrips']) * 0.8)
testsize = len(shuffled['numtrips']) - trainsize
npredictors = len(predictors.columns)
noutputs = 1
tf.logging.set_verbosity(tf.logging.WARN) # change to INFO to get output every 100 steps ...
shutil.rmtree('./trained_model_linear', ignore_errors=True) # so that we don't load weights from previous runs
estimator = tf.contrib.learn.LinearRegressor(model_dir='./trained_model_linear',
                                             feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(predictors.values))

print("starting to train ... this will take a while ... use verbosity=INFO to get more verbose output")
def input_fn(features, targets):
  return tf.constant(features.values), tf.constant(targets.values.reshape(len(targets), noutputs)/SCALE_NUM_TRIPS)
estimator.fit(input_fn=lambda: input_fn(predictors[:trainsize], targets[:trainsize]), steps=10000)

pred = np.multiply(list(estimator.predict(predictors[trainsize:].values)), SCALE_NUM_TRIPS )
rmse = np.sqrt(np.mean(np.power((targets[trainsize:].values - pred), 2)))
print('LinearRegression has RMSE of {0}'.format(rmse))
SCALE_NUM_TRIPS = 600000.0
trainsize = int(len(shuffled['numtrips']) * 0.8)
testsize = len(shuffled['numtrips']) - trainsize
npredictors = len(predictors.columns)
noutputs = 1
tf.logging.set_verbosity(tf.logging.WARN) # change to INFO to get output every 100 steps ...
shutil.rmtree('./trained_model', ignore_errors=True) # so that we don't load weights from previous runs
estimator = tf.contrib.learn.DNNRegressor(model_dir='./trained_model',
                                          hidden_units=[5, 5],                             
                                          feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(predictors.values))

print("starting to train ... this will take a while ... use verbosity=INFO to get more verbose output")
def input_fn(features, targets):
  return tf.constant(features.values), tf.constant(targets.values.reshape(len(targets), noutputs)/SCALE_NUM_TRIPS)
estimator.fit(input_fn=lambda: input_fn(predictors[:trainsize], targets[:trainsize]), steps=10000)

pred = np.multiply(list(estimator.predict(predictors[trainsize:].values)), SCALE_NUM_TRIPS )
rmse = np.sqrt(np.mean((targets[trainsize:].values - pred)**2))
print('Neural Network Regression has RMSE of {0}'.format(rmse))
input = pd.DataFrame.from_dict(data = 
                               {'dayofweek' : [4, 5, 6],
                                'mintemp' : [60, 40, 50],
                                'maxtemp' : [70, 90, 60],
                                'rain' : [0, 0.5, 0]})
# read trained model from ./trained_model
estimator = tf.contrib.learn.LinearRegressor(model_dir='./trained_model_linear',
                                          feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(input.values))

pred = np.multiply(list(estimator.predict(input.values)), SCALE_NUM_TRIPS )
print(pred)
```

- CloudML Engine simplifies the use of distributed Tensorflow in no-ops
- pip install google-api-python-client
- 

##### Array <-> Tree  ... Simplex?


- References:  [datastore]](https://cloud.google.com/datastore/), [bigtable](https://cloud.google.com/bigtable/), [bigquery](https://cloud.google.com/bigquery/), [cloud datalab](https://cloud.google.com/datalab/), [ tensorflow](https://www.tensorflow.org/), [cloud ml](https://cloud.google.com/ml/), [vision API](https://cloud.google.com/vision/), [google translate](https://cloud.google.com/translate/), [speech api](https://cloud.google.com/speech-to-text), [video intelligence](https://cloud.google.com/video-intelligence), [ ml-enging](https://cloud.google.com/ml-engine)

- Cloud pub/sub provides serverless global message queue for asynchronous processing
- Cloud data flow is the execurion framework for Apache beam pipelines
	- Dataflow does ingest, transform, and load; similar to Spark
- [https://cloud.google.com/pubsub/](https://cloud.google.com/pubsub/)
- [https://cloud.google.com/dataflow/](https://cloud.google.com/dataflow/)
- [https://cloud.google.com/solutions/reliable-task-scheduling-compute-engine](https://cloud.google.com/solutions/reliable-task-scheduling-compute-engine)
- [https://cloud.google.com/solutions/real-time/kubernetes-pubsub-bigquery](https://cloud.google.com/solutions/real-time/kubernetes-pubsub-bigquery)
- [https://cloud.google.com/solutions/processing-logs-at-scale-using-dataflow](https://cloud.google.com/solutions/processing-logs-at-scale-using-dataflow)
- cloud.google.com/training
- [ https://cloud.google.com/blog/big-data/](https://cloud.google.com/blog/big-data/)
- [ https://cloudplatform.googleblog.com/ ]( https://cloudplatform.googleblog.com/ )
- [ https://medium.com/google-cloud]( https://medium.com/google-cloud )


### 10/1/2018

#### Coursera: Leveraging Unstructured Data with Cloud Dataproc on Google Cloud Platform
- VVV - Voracity, Velocity, and Volume are three reasons that data is collected but not analyzed
- Declarative vs Imperitive programming Spark etc.
- "It's a lot of work adminstering servers.... (sigh) to be read, give us your $$
- DataProc Cluster: 
- Using Cloud storage rather than resident HDFS allows one to shut down the cluster, without losing data....
- gcloud dataproc clusters create test-cluster --worker-machine-type custom-6-3072 --master-machine-type custom-6-23040
-  ROT 50:50 pre-emptable, persistant VM 

#### Linux
- [ Linux ate RAM ](https://www.linuxatemyram.com)
- [ 5 commands to check memory usage](https://www.binarytides.com/linux-command-check-memory-usage/)
- [ Simplify your life with SSH config file ](https://nerderati.com/2011/03/17/simplify-your-life-with-an-ssh-config-file/)
- [ Dynamic Foward ](https://starkandwayne.com/blog/setting-up-an-ssh-tunnel-with-ssh-config/)
- [ disk stuff ][https://www.binarytides.com/linux-command-check-disk-partitions/]
- Hive: declarative - specifies exactly, Pig:imperative - better fit for pipelined
- ``` gcloud compute --projec=t --firewall-rules create ```
- sharding:=


### 10/4/2018
- [installing python packages from jupyter](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/)

```
import sys
!conda install --yes --prefix {sys.prefix} h5py
!conda install --yes --prefix {sys.prefix} scikit-image
# Install a pip package in the current Jupyter kernel
!{sys.executable} -m pip install scikit-image:w

```
 
#### dask
- submit -> gather
- pd.cut
- random.choice

```bash
sudo su -

yum -y install nfs-utils

mkdir /etl-storage/ /etl-storage/code/ /etl-storage/data/ /etl-storage/logs/
chown -R centos:centos /etl-storage

echo """
10.40.1.17:/srv/etl-storage/code           /etl-storage/code        nfs   auto            0  0
10.40.1.17:/srv/etl-storage/data           /etl-storage/data        nfs   auto            0  0
10.40.1.17:/srv/etl-storage/logs           /etl-storage/logs        nfs   auto            0  0
""" >> /etc/fstab

mount /etl-storage/code/
mount /etl-storage/data/
mount /etl-storage/logs/

exit
```

### 10/5/2018
[VSCode py.test debugggin](https://code.visualstudio.com/docs/python/debugging)

```  
	{
           “name”: “Python: pytest debugger)“,
           “type”: “python”,
           “request”: “launch”,
           “program”: “/anaconda3/bin/py.test”,
           “console”: “integratedTerminal”
        }
 
```
- CNN learning from Cellular Automata.
- pandas + async + caching -> trio, curio
-  [azure pipelines](https://app.vsaex.visualstudio.com/signup/pipelines?acquisitionId=7fc93e14-aa1b-4d7e-9219-dc48b25dc902&campaign=acom~azure~devops~pipelines~main~hero&projectVisibility=Everyone%2CEveryone&WebUserId=d2ff8659-da93-4b3e-8563-bc1afc0c2895), [xonsh](https://xon.sh), CI with azure pipelines with conda environments
	- Azure could be a one stop shop for [conda-forge](https://conda-forge.org/docs/news_announce.html).  Maybe better than Travis, 
- [Sourcegraph](https://sourcegraph.com/start), [github experiments](https://experiments.github.com)

#### Google Cloud Platform
 - Leveraging Unstructured Data with Cloud Dataproc on Google Cloud Platform
	- google cloud network has petabyte bisectional bandwidth
	- pyspark examples. flatmap
- Dataflow builds on Apache Beam
- HBase and BigTable use the same API. NoSQL
- Big Query Analytics is seperate from BigQuery Storage, and can be used to query cloud storage (CSV) and others.
- Big Query connectors allow spark to read directly from BQ
- Hosting scripts in a bucket makes sure they are accesible ``` gsutil cp my_init.sh gs://mybucket/init-actions/my_init.sh```
- gs://dataproc-initializatin-actions
-gs://dataproc-initialization-actions/datalab/datalab.sh
- https://cloud.google.com/dataproc/docs/concepts/configuring-clusters/network
```
gcloud dataproc clusters create cluster-custom \
--bucket $BUCKET \
--subnet default \
--zone $MYZONE \
--master-machine-type n1-standard-2 \
--master-boot-disk-size 100 \
--num-workers 2 \
--worker-machine-type n1-standard-1 \
--worker-boot-disk-size 50 \
--num-preemptible-workers 2 \
--image-version 1.2 \
--scopes 'https://www.googleapis.com/auth/cloud-platform' \
--tags customaccess \
--project $PROJECT_ID \
--initialization-actions 'gs://'$BUCKET'/init-script.sh','gs://dataproc-initialization-actions/datalab/datalab.sh'


gcloud compute \
--project=$PROJECT_ID \
firewall-rules create allow-custom \
--direction=INGRESS \
--priority=1000 \
--network=default \
--action=ALLOW \
--rules=tcp:9870,tcp:8088,tcp:8080 \
--source-ranges=$BROWSER_IP/32 \
--target-tags=customaccess
   
```
```
# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
  This program reads a text file and passes to a Natural Language Processing
  service, sentiment analysis, and processes the results in Spark.
  
'''
import logging
import argparse
import json
import os
from googleapiclient.discovery import build
from pyspark import SparkContext
sc = SparkContext("local", "Simple App")
'''
You must set these values for the job to run.
'''
APIKEY="AIzaSyCSj2vumT5dXDY_-9vdtpW0iQq1ueoHPq8"   # CHANGE
print(APIKEY)
PROJECT_ID="qwiklabs-gcp-56a1060e0c7e4e19"  # CHANGE
print(PROJECT_ID) 
BUCKET="qwiklabs-gcp-56a1060e0c7e4e19"   # CHANGE
## Wrappers around the NLP REST interface
def SentimentAnalysis(text):
    from googleapiclient.discovery import build
    lservice = build('language', 'v1beta1', developerKey=APIKEY)
    response = lservice.documents().analyzeSentiment(
        body={
i          'document': {
                'type': 'PLAIN_TEXT',
                'content': text
            }
        }).execute()
    
    return response
## main
# We could use sc.textFiles(...)
#
#   However, that will read each line of text as a separate object.
#   And using the REST API to NLP for each line will rapidly exhaust the rate-limit quota 
#   producing HTTP 429 errors
#
#   Instead, it is more efficient to pass an entire document to NLP in a single call.
#
#   So we are using sc.wholeTextFiles(...)
#
#      This provides a file as a tuple.
#      The first element is the file pathname, and second element is the content of the file.
#
sample = sc.wholeTextFiles("gs://{0}/sampledata/time-machine.txt".format(BUCKET))
# Calling the Natural Language Processing REST interface
#
# results = SentimentAnalysis(sampleline)
rdd1 = sample.map(lambda x: SentimentAnalysis(x[1]))
# The RDD contains a dictionary, using the key 'sentences' picks up each individual sentence
# The value that is returned is a list. And inside the list is another dictionary
# The key 'sentiment' produces a value of another list.
# And the keys magnitude and score produce values of floating numbers. 
#
rdd2 =  rdd1.flatMap(lambda x: x['sentences'] )\
            .flatMap(lambda x: [(x['sentiment']['magnitude'], x['sentiment']['score'], [x['text']['content']]
)] )
# First item in the list tuple is magnitude
# Filter on only the statements with the most intense sentiments
#
rdd3 =  rdd2.filter(lambda x: x[0]>.75)
results = sorted(rdd3.take(50))
print('\n\n')
for item in results:
  print('Magnitude= ',item[0],' | Score= ',item[1], ' | Text= ',item[2],'\n')
```
- 
```
from operator import add
lines = sc.textFile("/sampledata/sherlock-holmes.txt")

words =  lines.flatMap(lambda x: x.split(' '))
pairs = words.map(lambda x: (len(x),1))
wordsize = pairs.reduceByKey(add)
output = wordsize.sortByKey().collect()
``` 
-[ setting up Pelican bloc](https://rsip22.github.io/blog/create-a-blog-with-pelican-and-github-pages.html), [ pelican-resume ](https://github.com/cmenguy/pelican-resume),[pelican-themes](https://github.com/getpelican/pelican-themes), [pelicanthemes.com](http://www.pelicanthemes.com), [ pelican docs](http://docs.getpelican.com/en/stable/), [ pelican -cebong](https://github.com/getpelican/pelican-themes/tree/master/cebong), [pelican readthedocs](https://media.readthedocs.org/pdf/pelican/latest/pelican.pdf), 

### 10/09/2018
##### Serverless Data Analysis with Google BigQuery and Cloud Dataflow
- BigQuery facilitates sharing with no-ops
- Projects contain users, datasets
	- access control at the dataset level, not at the table level
		- Reader/Writer/Owner applied to all tables/views in dataset 
- Project -> Dataset -> Table, Job
- BigQuery is Columnar storage 
- WHERE clauses as early as possible to reduce data throughput (minimize cost)
- DO biggest JOINs first
- Know WHERE table partitions are such as _PARTITIONTIME
- Review BQ explanation plans
- Look for skew, do filtering earlier, 
	- Monitory BQ with Stackdriver
- BQ Pricing
	- Storage: Amount of data, Ingest rate of streaming, Automatic discount on older data
	- Processing: On-demand or Flat-rate, 1TB/month free
	- Free : Loading, Expoerting, Queries on metadata, Cached queries, Queries with errors
- References: [BigQuery Documentation](https://cloud.google.com/bigquery/docs/), [Tutorials](https://cloud.google.com/bigquery/docs/tutorials), [Pricing](https://cloud.google.com/bigquery/pricing), [client-libraries](https://cloud.google.com/bigquery/client-libraries)
- Dataflow is used for data processing pipelines
- MapReduce, Pardo like Map in MapReduce
- Python: Map vs. Flatmap
	- Use Map for 1:1 relationshiop between input and output:  ``` 'Wordlengths' >> beam.Map( lambda word: (word, len(word))) ```  
	- FlatMap for non 1:1 relationships, usually with generator
		- ``` def my_grep(line,term):
			if term in line:
				yield line

			'Grep' >> beam.FlatMap(lambda line: my_grep(line,searchTerm))
		```
- GroupBy operation is akin to shuffle
- GroupByKey is less efficient for skewed data at scale
- [Google Cloud Training Examples]](https://github.com/GoogleCloudPlatform/training-data-analyst)
#### XGBoost
[ airline-data dask-xgboost](https://gist.github.com/mrocklin/19c89d78e34437e061876a9872f4d2df)
[ ROC review ](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
[ XGBOOST ](http://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
[ ADABoost](https://en.wikipedia.org/wiki/AdaBoost)
[ XGBoost4J ](http://dmlc.ml/2016/03/14/xgboost4j-portable-distributed-xgboost-in-spark-flink-and-dataflow.html)
[ Benchmark-ML szilard](https://github.com/szilard/benchm-ml)

#### [ Predictive Analytics ](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29)

[Lambda Calculus for Transfinite Arrays ](https://arxiv.org/pdf/1710.03832.pdf)

```bash
curl https://storage.googleapis.com/cloud-training/CPB200/BQ/lab4/schema_flight_performance.json -o schema_flight_
performance.json

bq load --source_format=NEWLINE_DELIMITED_JSON $DEVSHELL_PROJECT_ID:cpb101_flight_data.flights_2014 gs://cloud-tra
ining/CPB200/BQ/lab4/domestic_2014_flights_*.json ./schema_flight_performance.json

bq ls $DEVSHELL_PROJECT_ID:cpb101_flight_data


```
### 10/16/2018
-[ zero to kubernetes]( https://zero-to-jupyterhub.readthedocs.io/en/latest/)
-[Google Zones](https://cloud.google.com/compute/docs/regions-zones/#available)


### 10/18/2018
#### Severless machine learning with Tensorflow on Google Cloud Platform
- MSE : for regression, Cross-Entropy: provides a differentiable error metric for classification 
- For balanced data, Accuracy : (TP + TN) / Total is a good measure, unbalanced requires Precision : TP / (TP + FP), and ... Recall : TP / (TP + FN) true positive rate 
```datalab create dataengvm --zone <ZONE>
%%javascript
$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')
```
-[Kubeflow](https://github.com/kubeflow/kubeflow)

```[python setup.py install --record files.txt
cat files.txt | xargs rm -rf 
```
- [What's your opinion of what to do with __init__.py](https://www.reddit.com/r/Python/comments/1bbbwk/whats_your_opinion_on_what_to_include_in_init_py/)
- [Contextlib.contextmanager](https://docs.python.org/2/library/contextlib.html)
- [Chainer](https://chainer.org)
- [ selfish ](https://github.com/ipython/ipython/wiki/Cookbook:-Connecting-to-a-remote-kernel-via-ssh)

### 11/27/2018
#### gcloud continues

```
gcloud auth configure-docker
```

- Case Studies
    - Flow Logistics: 
        Data: Cassandra, Kafka
        Applications: Tomcat, Nginx, 
        Storage: iSCSI, FC SAN, NAS
        Analytics: Apache Hadoop / Spark
        Misc: Jenkins, bastion, billing, monitoring, security,
    - Data Lake? -> [Avro](https://en.wikipedia.org/wiki/Apache_Avro)?
    - [Cloud Dataproc](https://cloud.google.com/solutions/images/using-apache-hive-on-cloud-dataproc-1.svg) is a fast, easy-to-use, fully managed service on GCP for running Apache Spark and Apache Hadoop workloads in a simple, cost-efficient way. Even though Cloud Dataproc instances can remain stateless, we recommend persisting the Hive data in Cloud Storage and the Hive metastore in MySQL on Cloud SQL. 
- [Table Partitionaing](https://www.cathrinewilhelmsen.net/2015/04/12/table-partitioning-in-sql-server/)


### 12/05/2018
#### Windows Interoperatbility
[ MSDev](https://blogs.msdn.microsoft.com/wsl/2016/10/19/windows-and-ubuntu-interoperability)
[ Share WSL ](https://blogs.msdn.microsoft.com/commandline/2017/12/22/share-environment-vars-between-wsl-and-windows/)
[ WSL DS ](https://docs.microsoft.com/en-us/windows/wsl/install-win10)


### Parsers in Python
[ Sly ](https://github.com/dabeaz/sly)
[ The C Pre-Processor](https://web.eecs.umich.edu/~prabal/teaching/eecs373-f11/readings/Preprocessor.pdf)
### 12/06/2018
#### ??= Computer software architecture entropy reduces from FBP -> Functional.  Conjecture that graph complexity , "loopiness" reduction, hot-spots (space-time, non-uniformity of computation and memory utilization), should optimized.

### 12/09/2018
- GCS practice with TRavis CI demo on app engine
- Azure (notebook ), Configured Dask system on Digital Ocean (scheduler), Google Cloud Services ( worker )
- 
- [ osquery](https://osquery.io), [ hubblestack ](https://blogs.adobe.com/security/2017/12/introducing-hubblestack.html)
- [ Travis CI tutorial ](https://cloud.google.com/solutions/continuous-delivery-with-travis-ci)  


### 12/11/2018
- [ docker on Azure ](https://github.com/charlieding/Virtualization-Documentation/tree/live/hyperv-tools/Nested### 12/11/2018)
- [ Docker Hyper-v](https://docs.docker.com/machine/drivers/hyper-v/#options)
- [ Hyper V Setup ](https://docs.docker.com/machine/drivers/hyper-v/#3-reboot)
- [ Linux Java GCE ](https://cloud.google.com/java/docs/setup)
- [ JPype ](http://jpype.sourceforge.net)
- [ PaloAlto Linx ](https://www.paloaltonetworks.com/documentation/41/globalprotect/globalprotect-app-user-guide/globalprotect-app-for-linux/use-the-globalprotect-app-for-linux#id181NC060RNM)
- [ Nested Virtualization](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nested-virtualization)
- [ VMWare Player ](https://my.vmware.com/web/vmware/free#desktop_end_user_computing/vmware_player/7_0)
- [ Callng Java from Python ](https://exceptionshub.com/calling-java-from-python-closed.html)
- [ Jniius vs JPype ](https://docs.google.com/spreadsheets/d/1fHBJJDQ0BavUMX8O7rKAjpGwY6AB_OO_AsudM4ch57k/edit#gid=0)
- [ Pyjnius ](https://pyjnius.readthedocs.io/en/latest/)
 
- [ jtds ](https://sourceforge.net/projects/jtds/files/jtds/1.2.2/)
- [ open edge ](https://www.progress.com/download/thank-you?interface=jdbc&ds=openedge&os=std)
- [ jaydebapi ](https://pypi.org/project/JayDeBeApi/)
- [ pyodbc ](https://pypi.org/project/pyodbc/)
- [ pymssql ](http://www.pymssql.org/en/stable/)
- [ oracle java ](https://www.oracle.com/technetwork/java/javase/downloads/jdk11-downloads-5066655.html)

```
jar tf <jarfile>
```

- [ Ilya ](http://ikuz.eu/2011/12/19/connect-to-mssql-from-matlab-on-mac-os-x/)

- [ DBVisualizer](https://www.dbvis.com/download/10.0)

``` 
launchctl list 
launchctl remove <name_from_list_command>
 ls /Library/LaunchAgents/
 ls /Library/LaunchDaemons/
 System Preferences > Users & Groups.
```
[ launcht items ](http://yorkerfrank.blogspot.com/2011/07/how-to-remove-unneeded-startup-services.html)
[ launc itesm ](https://www.macworld.com/article/2047747/take-control-of-startup-and-login-items.html)
[ reload nginx ](https://www.cyberciti.biz/faq/nginx-linux-restart/) 

[ CI Nvidia ](http://on-demand.gputechconf.com/gtc/2018/presentation/s8563-building-a-gpu-focused-ci-solution.pdf)
[ TSQL ](https://github.com/tsqllint/tsqllint)
[ RAPIDS ](https://rapids.ai/)
[ GPUCI ](https://github.com/rapidsai/libgdf/wiki/gpuCI)


[ git commands ](https://tapaswenipathak.wordpress.com/2016/02/15/git-fetch-merge-git-fetch-rebase-git-pull/)
``` 
git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=3600'

```
### 12/20/2018
[echo "ServerAliveInterval 120" > /etc/ssh/ssh_config](https://askubuntu.com/a/142430)
[MS restore ](https://docs.microsoft.com/en-us/sql/t-sql/statements/restore-statements-transact-sql?view=sql-server-2017)

```git rebase -i HEAD~2```
``` set_diff_df = pd.concat([df2, df1, df1]).drop_duplicates(keep=False) ``` results in  right difference in pandas.

### 12/21/2018
- Ibis, Xonsh, custom writes in google groups, 
- Q, Argparse, logger, chainmap 
[ dog_tunnel plist ](http://www.zenspider.com/ruby/2011/11/ssh-tunneling-via-osx-s-launchctl.html)
``` launchctl load ~/Library/LaunchAgents/com.zenspider.ssh-tunnel.plist```

[ ssh config tunnels ](https://starkandwayne.com/blog/setting-up-an-ssh-tunnel-with-ssh-config/)
