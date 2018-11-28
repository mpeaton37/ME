[Chris Ostrouchov's notes](https://gist.github.com/costrouc/d9db5f6f81779418842bb0df580a11ca)

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
	- ![DataStore is like a persistant HashMap](https://storage.cloud.google.com/passet/images/gcenotes/Datastore.png?_ga=2.51433632.-80940699.1539256678&_gac=1.208334118.1541249466.EAIaIQobChMI77Lp0t-13gIVC77ACh125w3fEAAYASAAEgK2efD_BwE)
	- ![Crud Operations are easily implemented in Datastore](https://storage.cloud.google.com/passet/images/gcenotes/Crud.png?_ga=2.86431889.-80940699.1539256678&_gac=1.195751448.1541249466.EAIaIQobChMI77Lp0t-13gIVC77ACh125w3fEAAYASAAEgK2efD_BwE)
        - Create, Read, Update, Delete 
	- ![Choose Storage Option based on Usage Pattern](https://storage.cloud.google.com/passet/images/gcenotes/Storage.png?_ga=2.94828693.-80940699.1539256678&_gac=1.253890236.1541249466.EAIaIQobChMI77Lp0t-13gIVC77ACh125w3fEAAYASAAEgK2efD_BwE)
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

- CloudML Engine simplifies the use of distributed Tensorflow in no-ops
- pip install google-api-python-client

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

- Datastore
- Pub/Sub
- Gloud Storage
- Data Proc
- Cloud SQL
- Big Table
- Big Query
- ML Engine
- Data Flow
- Container Storage
- Kubernetes Engine
- Compute Engine
- Key Storage
- 

### DataBase
    - [Spanner](https://cloud.google.com/spanner/docs/)
    - [BigTable](https://cloud.google.com/bigtable/docs/)
        - [Managing Tables](https://cloud.google.com/bigtable/docs/managing-tables)
        - [cbt tool](https://cloud.google.com/bigtable/docs/cbt-overview) or [Hbase shell](https://cloud.google.com/bigtable/docs/installing-hbase-shell)
    
### Storage
- [Storage Classes](https://cloud.google.com/storage/docs/storage-classes)
    - Multi-Regional: hot around the world , Regional: narrow geographical, Nearline: < 1/mo, Coldline: < 1/yr, 
    - Cloud Storage SLA
    - 

### Pricing
- [price list](https://cloud.google.com/pricing/list### Pricing)
- [BigQueryStorage](https://cloud.google.com/bigquery/pricing)
- [ Transfer Appliance](https://cloud.google.com/data-transfer/pricing)
- 

### Data Transfer
- [ Transfering big Data Sets](https://cloud.google.com/solutions/transferring-big-data-sets-to-gcp)
- Transfer Service: 
    - If your data source is an Amazon S3 bucket, an HTTP/HTTPS location, or a Cloud Storage bucket, you can use Storage Transfer Service to transfer your data.:w
- gsutil
- [Google Partners](https://cloud.google.com/storage/partners/)
- Offline Data TRansfer
- [Transfer Appliance](https://cloud.google.com/transfer-appliance/)
     

         
### Practice exam

### Misc
- [Data Partitioning](https://www.cathrinewilhelmsen.net/2015/04/12/table-partitioning-in-sql-server/)

