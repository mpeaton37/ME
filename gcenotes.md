### Google Cloud Platform Fundamentals
- NIST definition of cloud computing: way of using IT that has:
	- On demand
	- Broad network access
	- Resource pooling
	- Rapid elasticity
	- Measured service - GCP Pricing offers per second billing for
	- Google Compute Engine
	- Kubernetes Engine := Container Infrastructure as a service
	- Cloud Data Proc := Hadoop as a Service
	- App Engine := Platform as a Servicee
	- Sustained use discounts are available for over 25% use/month
- Several open API's are available from GCP
	- Cloud Bigtable : Database that implements Hbase interface	
	- Cloud Dataproc := Hadoop 
	- Tansorflow := Software library for machine learning
	- Kubernetes := Container management allows users to mix and match
	- StackDriver := Cloud monitoring service.
	- google provides compatability layers to prevent lock-in
	- 
- Summary
	- Compute, Storage, Big Data, Networking, Machine Learning and  Operations and tools
	- Compute = [ Compute Engine, Kubernetes Engine, App Engine, Cloud Functions ]
	- Storage = [ Bigtable, Cloud Storage, Cloud SQL, Cloud Spanner, Cloud Datastore ] 
	- Big Data = [ BigQuery, Pub/Sub, Dataflow, Dataproc, Datalab]
	- Machine Learning = [ Natural Language API, Vision API, Machine Learning, Speech API, Translate API ]
- Multi-layered security approach
	- Hardware Infrastructure - Hardware design and provenance (Titan chip); secure boot stack; premises security
	- Service deploymnent - Encryption of inter-service communication
	- User identity - Central identity service with support for U2F
	- Storage services - Encryption at rest
	- Internet conmunications - Google Front End; designed in Denial of Service protection
	- Operational security - Intrusion detections systems; techniques to reduce insider risk; employee U2F use; software development practices	
	- Resource hierarchy levels define trust boundaries.  Policies are inherited downward.
	-   
- IAM roles
	- (Who), can do (What), on which resource (Where) 
	- Who = ( account, cloud identity user, service account, google group, gsuite identity or domain )
	- What = ( primitive = (owner, editor, viewer, billing administrator),predefined=(),custom= (),) 
	- Where = ( project, folder or org) 	
- Interacting with Google Cloud
	- Cloud Client Libraries := Latest and recommended libraries
	- Google API Client Library := generallity and completeness
	- Cloud Console Mobile App
- Cloud Launcher (Former Cloud Launcher) 
	-   
- Google Cloud VPC
	- SEgment, Firewall rules, create static routes
	- VPC networks have global scope, subnets are regional and may span any GCP region worldwide
	- Cloud load balancing Frontends and produces cross region failover and load balancing
	- Global Https, SSL proxy, TCP proxy, REgional, Regional Internal load-balancing options
	- VPN, Direct Peering, Carrier Peering, Dedicated Interconnect
	- Cloud Routing over Border Gateway Protocol.
	
- Cloud Storage
	- Object sotrage is not a filesystem
	- Storage Objects are immutable
	- Data at rest and in transit are encrypted
	- Multi-Regional, Regional, Nearline, Coldline
	- Storage Price, vs Transfer Price 
	- 
- BigTable
	- HBase interface
	- Low Latency, large data
	- in from Dataflow, Spark, Storm
	- read and written from Hadoop map reduce, Dataflow, or spark
	- used by google for Google maps
	-  
- Cloud SQL and Cloud Spanner
	- Provides read, failover, and external replica services
	- on demand or scheduled backups
	- verticle (machine type) and horizontal scaling ( read replicas)
	- accesible from other google services
	- Cloud Spanner allows global scaling
- Cloud Datastore
	- highly scalable NoSQL databse
	- SQL like queries
	- free daily quota
- Kubernetes
	- Runs on Compute Engine
	- Managed service releaves developers of operational details
	- Google Cloud container builder
	- Google Cloud container registry
		- Users IAM permissions and stored in google cloud
- App Engine
	- Built in Services for NosqL, in-memory caching, load balancing, health checks, logging etc.a
	- Standard, and Flexible environements
	- Standard has no charge for low utilization
	- Standard is for specific versions of Python, Java, PhP, and Go
	- Flexible App Engine manages Docker containers
	- Cloud Endpoint 
	- Apigee Edge - Analytics of edge and gradually decompose pre-existing monolithic application
- Development in the cloud
	- Cloud Source Repositories
		- Any number of private git repositories
	- Cloud Functions 
		- single purpose functions that respond to events without a server at runtime
	- Infrastructure as Code - YAML markup of deployment environment
		- Store and configure deployment manager template in cloud source repositories
	- Stackdriver	
		- Monitoring, Logging, Error Reporting, Debugger, Trace	
		- Any code location in production can be inspected
- Cloud Big Data Platform
	- Cloud Dataflow -> General Purpose ETL
		-  fraud detection and financial services, IoT analytics in manufacturing, healthcare and logistics and click stream, point of sale and segmentation analysis in retail
		- Cloud Dataflow is a better choice than Dataproc for unpredictable size or rate data
		- Fully automates the management of whatever resources are required
	- Big Query is Google's fully managed, petabyte scale, low cost analytics data warehouse.
		- Load from cloud storage, cloud datastore or stream up to 100,000 rows per second.  	
	 	- Storage and Queries are billed seperately
		- Price drops after 90 days of storage
	- Cloud Pub/Sub is a reliable, scalable foundation for stream analytics 
		- great for decoupling systems
		- at least once delivery at low latency
		- 
	- Cloud Datalab 
		- lets you analyize your data in Jupyter notebooksa
	- *Machine Learning Platforms*
		- applications that improve themselves over time!
		- Tensorflow is an open source software library that's exceptionally well suited for machine learning
		- GCP is an ideal 
		- Each TPU provides up to 180 teraflopw of performance
		
		- Cloud Vision API
		- Cloud Speach API
			- Over 80 languages and variants		
		- Language API 
			- syntax and sentimate analysis
		- Video Intelligence API
			- annotate videos and make content searchable and discoverable
		
- Virtual Networking
	- Implied deny out ingress and egress allow all firewall rules exist for the network
	- Direction, source or destination, protocol/port, action, priority, and instance can all be specified for firewall rules
	- tags can be used between internal vm's for ingress rules.
	- Increased availability can be acheived by adding vm's in different zones to subnets
	- Bastion host provides external interface to an otherwise internal service.
	- VM's offer per-second billing, sustained use discounts  

- Load Balancing
	- Global and Regional
		- Global 
			- HTTP Load balancing: distrubutes based on proximity to user and/or requested URL	
			- SSL proxy load balancing: distributes SSL traffic amont groups of instances based on proximity to user
			- TCP proxy load balancing: "" TCP ""
		- Regional
			- Network load balancing: distributes among pool, any TCP/UDP
			- Internal load balancing: distributes from GCP
- Managed Instance Groups: identical instance based on instance template
	- Health checks determine whether instances are available to do work
	- 
- Health check and load balancing are configurable 
	- https load balancing 80:8080, 443
	- GCLB cookies for user -> instance assignement
	- Instance templates define managed instance groups which can be managed in part by an autoscaler
	- CLIENT_IP_PROT CLIENT_IP specify for persistant usr<->backend mapping
	- provision at 100% for 2/3 service, 150% for full service during outage 	
- Autoscaling
	- one autoscaler per managed instance group
	- policies detemine autoscaling behaviors
	- cpu utilization, http load balancing
- Infrastructure Automation
	- Snapshots can be taken and used to provide boot images withing a project
	- Custom images can be developed and share dbetween projects
	- Custom images can be upload from other platforms or local VM's
	- Deployment manager is an infrastructure automation tool.
	- Orchestration tools: Puppet, Chef, Terraform, Cloud Formation, ( Deployment Manager)
	- Cloud launcher avails common "tested and verified" implementations (paid and free)
- Managed Services
	- Infrastructure -> Automated Infrastructure IaaS -> PaaS -> SaaS
		- Infrastructure: [ Console, Cloud Shell], 
		- IaaS:[ Cloud API, Deployment Manager]
		- PaaS:[Dataproc, Dataflow], Saas: BigQuery   	
			- DataProc: [ Spark, Hadoop, Hive, Pig]
			- Dataflow: [ Apache Beam]
			- BigQuery: Fully managed Petabyte "complete" data analytics solution
- Elastic Cloud Infrastruture: Containers and Services
	- Kubernetes 
- Cloud Pub/Sub
	- Scales Globally
		- Low Latency with dynamic rate limiting
		- Durability, Reliability, application ACK
		- Encryption in motion and at rest
		- Maintenance free
- Google Cloud Endpoints
	- create, deploy, protect, monitor, and analyze and serve your APIs.
	- Apigee provides data analytics and validation.
- Google Cloud Functions
	- Lightweight, compute solution for developers to create single purpose, stand-alone, functions that respond to cloud events without the need to manage a server or runtime environment.
	- Single purpose, short lived, self contained.
- Specialty API's
	- Machine learning, Cloud job discovery, natural language, Speech, Translation, and Vision API
- App Engine
	- Standard and Flexible Environment
	- Flexible environement based on Docker containers
- Containers
	- Google Kubernetes Engine, Google App Engine, or Containers on Compute Engine
	- Containers help decouple Applications from dependencies
	- Kubernetes is a fully managed service on Google
	- A Pod holds more than one container
	- Cluster Federation for deploying across multiple regions or providers
	- Stackdrier login and management
	- labels used for kubernetes to orchestrate containers (query)
	- Deloyment groups specify rules for autoscaling.
	- Cluster Federation offers a single logical compute across regions, clproviders, et.
	- 

[Chris Ostrouchov's notes](https://gist.github.com/costrouc/d9db5f6f81779418842bb0df580a11ca)

#### Google Cloud Services Coursera Course
- Discount based on percentage of month used
- Preemptable machine allow discounts for interrupt tolerant jobs.
- Cloud Storage is a Blob
  - Bucket like domain name gs://acme-sales/data
  - use gsutil {cp,mv,rsync,etc.}
  - REST API - Use Cloud Storage as a holding area
  - Zone locallity to reduce latency, distribute for redundancy & global access.s

#### Google Cloud Platform
- 
- Intro to Scaling Data Analysis
	- ![DataStore is like a persistant HashMap](http://35.225.147.84/images/gcenotes/Datastore.png?_ga=2.51433632.-80940699.1539256678&_gac=1.208334118.1541249466.EAIaIQobChMI77Lp0t-13gIVC77ACh125w3fEAAYASAAEgK2efD_BwE)
	- ![Crud Operations are easily implemented in Datastore](http://35.225.147.84/images/gcenotes/Crud.png)
        - Create, Read, Update, Delete 
	- ![Choose Storage Option based on Usage Pattern](http://35.225.147.84/images/gcenotes/Storage.png?_ga=2.94828693.-80940699.1539256678&_gac=1.253890236.1541249466.EAIaIQobChMI77Lp0t-13gIVC77ACh125w3fEAAYASAAEgK2efD_BwE)
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
            - [ BigQuery cookbook](https://support.google.com/analytics/answer/4419694?hl=en), [Avoiding SQL Anti-pattern](https://cloud.google.com/bigquery/docs/best-practices-performance-patterns)
            - [ BigQuery Dimensions ](https://medium.com/@doctusoft/data-warehouse-in-bigquery-dimensions-part-1-af7c0d24a117)
- CloudML Engine simplifies the use of distributed Tensorflow in no-ops
- [Build a DataLake](https://cloud.google.com/solutions/build-a-data-lake-on-gcp)
 
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
![dataproc decision](https://cloud.google.com/dataflow/images/flow-vs-proc-flowchart.svg)
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

- [Datastore](https://cloud.google.com/datastore/docs/concepts/overview)
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
- Key Storage (KMS)
    - [ KMS ](https://cloud.google.com/kms/?utm_source=google&utm_medium=cpc&utm_campaign=na-US-all-en-dr-bkws-all-all-trial-b-dr-1003905&utm_content=text-ad-none-any-DEV_c-CRE_293220272125-ADGP_Hybrid+%7C+AW+SEM+%7C+BKWS+%7C+US+%7C+en+%7C+BMM+~+Identity/Security+~+KMS+~+Google+Cloud+Key+Management-KWID_43700036550313578-kwd-283025362978&utm_term=KW_%2Bgoogle%20%2Bcloud%20%2Bkey%20%2Bmanagement-ST_%2Bgoogle+%2Bcloud+%2Bkey+%2Bmanagement&gclid=Cj0KCQiA3IPgBRCAARIsABb-iGIv88JnnGl68n_i7Sp_JkKQjlYQw1WOrYpU2rxz7vQmLF-_Prbl1soaAsbWEALw_wcB)
- IAM
    [ Overview ](https://cloud.google.com/storage/docs/access-control/iam)
- Data Studio
    - [Youtube Analytics connector](https://support.google.com/datastudio/answer/7020432?hl=en)
    - Filter for Specific segment, dimension for full segmentation
        - [ Filters ](https://support.google.com/datastudio/answer/6291066?hl=en)
- Vision API
    - [ annotations ](https://medium.com/@srobtweets/exploring-the-cloud-vision-api-1af9bcf080b8)
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
- [ Persistant Disk ](https://cloud.google.com/persistent-disk/) 
- [ Cloud DataStore ](https://cloud.google.com/datastore/pricing)

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

![Data Lake](https://cloud.google.com/solutions/images/data-lake-workflow.svg)


