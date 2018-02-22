# Image Classification with Keras and Cloud ML Engine

This folder contains a number of teaching labs that use Keras to build a basic binary image classifier, implement various techniques to improve training with small datasets, and ultimately schedules training to Google's Cloud ML Engine (CMLE).


## Setup

###Create VM and Jupyter Notebook

For this lab we will use a Jupyter Notebook running on a Google Compute Engine (GCE) VM backed by a GPU.

Open the Cloud Shell in the Google Cloud console and run the following 

```
gcloud compute instances create [vm-name]  \
 --machine-type=n1-standard-4  \
 --image-project=ubuntu-os-cloud  \
 --image-family=ubuntu-1604-lts  \
 --scopes=cloud-platform \
 --accelerator type=nvidia-tesla-p100 \
 --maintenance-policy TERMINATE \
 --restart-on-failure
 --zone=us-central1-c

```

After a minute, we should be able to connect: 

```
gcloud compute ssh [vm-name]
```

###Download code and install dependencies

We need to setup CUDA so Tensorflow can use our image.  The following commands,
run on the evaluation VM, will install CUDA and Tensorflow on our GPU VM. After 
the installation finishes, we recommend you restart the VM.

```
cat > /tmp/setup.sh <<HERE
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-get update
apt-get install -y cuda-9-0
bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list'
apt-get update
apt-get install -y --no-install-recommends libcudnn7=7.0.5.15-1+cuda9.0
apt install -y python-pip python-tk
pip install tensorflow-gpu==1.4
pip install numpy pandas keras
HERE

sudo bash /tmp/setup.sh
```

```
shutdown -r now
```

```
gcloud compute ssh [vm-name]
```

On the newly restarted VM, make sure NVIDIA care is working:
```
nvidia-smi
```

Download code:
```
git clone https://github.com/dhodun/ml-teaching-examples.git
```

### Start Jupyter Notebook and Connect

On VM:
```
jupyter notebook --port 8088
```

In the Google Cloud console click "Web Preview" button on upper right of Cloud Shell and click 'Preview on port 8080' to open Jupyter Window