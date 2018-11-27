### Create push some docker image to container store and instanciate kubernetes cluster       
```
  gcloud compute instances create gcelab2 --zone us-central1-c
  gcloud compute instances create deep-dog --zone us-central1-c
  gcloud auth list
gcloud config set account `mpeaton@gmail.com`
gcloud config set account mpeaton@gmail.com
gcloud config list project
cd training-data-analyst/ 515  curl http://localhost:8888
curl http://localhost:8080
docker ps
gcloud config list project
gcloud config set project dask-dev
gcloud config
gcloud config list
gcloud container list
gcloud container help
gcloud container --help
gcloud container clusters --help
gcloud container clusters list
gcloud auth login
gcloud container clusters list
gcloud --help
gcloud services list
gcloud config set project deep-dog
gcloud services list
gcloud container clusters list
gcloud container clusters list
gcloud container clusters --help
vim server.js
```
###### server.js
```
var http = require('http');
var handleRequest = function(request, response) {
  response.writeHead(200);
  response.end("Hello Kubernetes World!");
}
var www = http.createServer(handleRequest);
www.listen(8080);
```
###### Start node
```
node server.js
ndoe
node
vi Dockerfile
docker build -t gcr.io/PROJECT_ID/hello-node:v1 .
docker build -t gcr.io/deep-dog/hello-node:v1 
docker build -t gcr.io/deep-dog/hello-node:v1 .
ls
vi Dockerfile 
docker build -t gcr.io/deep-dog/hello-node:v1 .
docker run -d -p 8080:8080 gcr.io/PROJECT_ID/hello-node:v1
docker run -d -p 8080:8080 gcr.io/deep-dog/hello-node:v1
curl http://localhost:8080
docker ps
docker stop gcr.io/deep-dog/hello-node:v1
docker stop sharp_noble
docker stop 918e36b7fa66
gcloud docker -- push gcr.io/deep-dog/hello-node:v1
gcloud container clusters create hello-world                 --num-nodes 2                 --machine-type n1-standard-1                 --zone us-central1-a
kubectl run hello-node     --image=gcr.io/PROJECT_ID/hello-node:v1     --port=8080
kubectl run hello-node     --image=gcr.io/deep-dog/hello-node:v1     --port=8080
kubectl get deployments
kubectl get pods
kubectl config view
kubectl get events
kubectl cluster-info
kubectl get pods
kubectl logs hello-node-66db8d9d95-jdhfm
kubectl expose deployment hello-node --type="LoadBalancer"
kubectl get services
kubectl scale deployment hello-node --replicas=4
kubectl get deployment
kubectl get pods
kubectl logs hello-node-66db8d9d95-grmr9

docker build -t gcr.io/PROJECT_ID/hello-node:v2 .
gcloud docker -- push gcr.io/PROJECT_ID/hello-node:v2
docker build -t gcr.io/deep-dog/hello-node:v2 .
gcloud docker -- push gcr.io/deep-dog/hello-node:v2
kubectl edit deployment hello-node
kubectl edit deployment hello-node
kubectl get deployments
gcloud container clusters get-credentials hello-world     --zone us-central1-a --project deep-dog
kubectl -n kube-system describe $(kubectl -n kube-system \ get secret -n kube-system -o name | grep namespace) | grep token:
kubectl proxy --port 8081
kubectl get deployments
```
#### Make a gcloud bucket for project
``` 
 export PROJECT=$(gcloud info --format='value(config.project)')
 gsutil mb -l $REGION gs://$PROJECT-warehouse 
```
#### Create CloudSQL instance for HIVE metastore

``` 
gcloud sql instances create hive-metastore \
    --database-version="MYSQL_5_7" \
    --activation-policy=ALWAYS \
    --gce-zone $ZONE
```
### Create Dataproc cluster
```
gcloud dataproc clusters create hive-cluster \
    --scopes sql-admin \
    --image-version 1.3 \
    --initialization-actions gs://dataproc-initialization-actions/cloud-sql-proxy/cloud-sql-proxy.sh \
    --properties hive:hive.metastore.warehouse.dir=gs://$PROJECT-warehouse/datasets \
    --metadata "hive-metastore-instance=$PROJECT:$REGION:hive-metastore"
```
[etc](https://cloud.google.com/solutions/using-apache-hive-on-cloud-dataproc)

