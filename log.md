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
