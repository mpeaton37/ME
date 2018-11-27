#### Google Cloud Services Coursera Course
- Discount based on percentage of month used
- Preemptable machine allow discounts for interrupt tolerant jobs.
- Cloud Storage is a Blob
  - Bucket like domain name gs://acme-sales/data
  - use gsutil {cp,mv,rsync,etc.}
  - REST API
  - Use Cloud Storage as a holding area
  - Zone locallity to reduce latency, distribute for redundancy & global access.s

#### Google Cloud Platform
- Intro to Scaling Data Analysis
	- ![DataStore is like a persistant HashMap](../screenshots/Screen Shot 2018-09-28 at 9.25.13 AM.png)
	- ![Crud Operations are easily implemented in Datastore](file://./screenshots/Screen Shot 2018-09-28 at 9.47.00 PM.png)
	- ![Choose Storage Option based on Usage Pattern](file://./screenshots/Screen Shot 2018-09-28 at 9.58.43 PM.png)
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


[ google cloud ](https://software.seek.intel.com/Edge_Devices_Webinar_Reg)


-Preemptable machine allow discounts for interrupt tolerant jobs.
-Cloud Storage is a Blob
-Bucket like domain name gs://acme-sales/data
-use gsutil {cp,mv,rsync,etc.}
-REST API
-Use Cloud Storage as a holding area
-Zone locallity to reduce latency, distribute for redundancy & global access.

``` 
### gcsf data
from gcsfs import GCFSFileSystem
gcs = GCSFileSystem()
gcs.glob('anaconda-public-data/nyc-taxi/csv/2015/yellow_*.csv')

### Kubernetes cluster
from dask_kubernetes import KubeCluster
cluster = KubeCluster(n_workers=20)
client.get_versions()
###

```
- Cloud Dataproc is a fast, easy-to-use, fully managed service on GCP for running Apache Spark and Apache Hadoop workloads in a simple, cost-efficient way. Even though Cloud Dataproc instances can remain stateless, we recommend persisting the Hive data in Cloud Storage and the Hive metastore in MySQL on Cloud SQL. 

![data transfer methods](https://cloud.google.com/solutions/images/tran5.png)
![defining close](https://cloud.google.com/solutions/images/transfer-speed.png)

``` gsutil -m cp -r [SOURCE_DIRECTORY] gs://[BUCKET_NAME] ```
![ multithread transfer](https://cloud.google.com/solutions/images/big-data-multithreaded-transfer.svg)
[ gsutil tool documentation](https://cloud.google.com/storage/docs/gsutil/commands/perfdiag#options)

```gcloud auth configure-docker```
- Case Studies
    - [Flow Logistics](https://cloud.google.com/certification/guides/data-engineer/casestudy-flowlogistic): 
        Data: Cassandra, Kafka
        Applications: Tomcat, Nginx, 
        Storage: iSCSI, FC SAN, NAS
        Analytics: Apache Hadoop / Spark
        Misc: Jenkins, bastion, billing, monitoring, security,
    - Data Lake? -> [Avro](https://en.wikipedia.org/wiki/Apache_Avro)?
    - [Cloud Dataproc](https://cloud.google.com/solutions/images/using-apache-hive-on-cloud-dataproc-1.svg) is a fast, easy-to-use, fully managed service on GCP for running Apache Spark and Apache Hadoop workloads in a simple, cost-efficient way. Even though Cloud Dataproc instances can remain stateless, we recommend persisting the Hive data in Cloud Storage and the Hive metastore in MySQL on Cloud SQL. 
    - [MJTelco](https://cloud.google.com/certification/guides/data-engineer/casestudy-mjtelco)
        - 
