### Dask

- Delayed
- Futures
- Cache
- compute()
- persist()
- scatter

distributed.cli.dask_worker
~/.config/dask/distributed.yaml

-[ Scalable Scientific Computing Using Dask ](https://youtu.be/OhstDq8l3OM) 

#### dask
- [dsk_ml kmean](https://github.com/dask/dask-ml/blob/master/dask_ml/cluster/k_means.py) uses [cblas_sdot](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-sdot)
- [ k-means ](https://en.wikipedia.org/wiki/k-means_clustering)
  - the running time of [lloyd's algorithm ](http://www-evasion.imag.fr/people/franck.hetroy/teaching/projetsimage/2007/bib/lloyd-1982.pdf)(and most variants) is o(nkdi)
  - [scalable k-mean](https://arxiv.org/abs/1203.6402) 
  - [partition function](https://en.wikipedia.org/wiki/partition_function_(mathematics))->?[ fredholm theory](https://en.wikipedia.org/wiki/fredholm_theory)
  - [green's function](https://en.wikipedia.org/wiki/green%27s_function)

[dask-tutorial](https://ww.youtube.com/watch?v=mbfsog3e5DA)
  - [ dask-tutorial github](https://github.com/dask/dask-tutorial)
  - Methods on delayed objects just work
  - Containers of delayed objects can be passed to compute

- [Dask webinar ]( http://bit.ly/2s5Zt4v )
- [Matt Rocklin](http://matthewrocklin.com/blog/work/2017/02/07/dask-sklearn-simple)
- scale to larger than memory data
- [dask setup](https://dask.pydata.org/en/latest/setup.html)
- [dask yarn](https://github.com/dask/dask-yarn)

- [ Scalable Machine Learning with Dask (SciPy 2018)](https://youtu.be/ccfsbuqsjgI)
  - [ TomAugspurger ](https://github.com/TomAugspurger)
  - [ @TomAugpurger ](https://twitter.com/tomaugspurger?lang=en)
  - [ Dask ](http://dask.pydata.org/en/latest/)
    + [ Dask-ML](http://dask-ml.readthedocs.io/en/latest/) builds on Dask to make machine learning more scalable
    
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


  #### [Dask Futures](http://dask.pydata.org/en/latest/futures.html)
  - "```python remote_df = client.scatter(df) ```"

- [ VSCode Unit Testing ](https://code.visualstudio.com/docs/python/unit-testing)
- ```git merge --strategy-option theirs```

[ airline-data dask-xgboost](https://gist.github.com/mrocklin/19c89d78e34437e061876a9872f4d2df)
[ ROC review ](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
[ XGBOOST ](http://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
[ ADABoost](https://en.wikipedia.org/wiki/AdaBoost)
[ XGBoost4J ](http://dmlc.ml/2016/03/14/xgboost4j-portable-distributed-xgboost-in-spark-flink-and-dataflow.html)
[ Benchmark-ML szilard](https://github.com/szilard/benchm-ml)


