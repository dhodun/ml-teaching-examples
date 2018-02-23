# Image Classification with Keras and Cloud ML Engine

This folder contains a number of teaching labs that use Keras to build a basic binary image classifier, implement various techniques to improve training with small datasets, and ultimately schedules training to Google's Cloud ML Engine (CMLE).


## Setup

### Install gcloud SDK on your local workstation

Click here: [gcloud installation instructions](https://cloud.google.com/sdk/downloads)

### Create VM and Jupyter Notebook

For this lab we will use a Jupyter Notebook running on a Google Compute Engine (GCE) VM backed by a GPU.

Open a shell on your local workstation with gcloud installed and run the following: 

```
gcloud compute instances create [vm-name]  \
 --machine-type=n1-standard-4  \
 --image-project=dhodun-lab  \
 --image=gpu-tf-image  \
 --scopes=cloud-platform \
 --accelerator type=nvidia-tesla-p100 \
 --maintenance-policy TERMINATE \
 --restart-on-failure \
 --zone=us-central1-c
```

After a minute, we should be able to connect: 

```
gcloud config set compute/zone us-central1-c
gcloud config set project [project-id]
gcloud compute ssh [vm-name] -- -L 8888:localhost:8888
```

### Test GPU and download code


On the newly created VM, make sure NVIDIA card is working:
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

Copy and paste the jupyter link into a local browser if it does not launch automatically. 

Navigate to and open the First notebook in the 'ml-teaching-examples/cats_dogs' directory and proceed with instructions.

## Answers to code samples

### Building our network

```python
from keras import models
from keras import layers

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
```

### Training our network
```python
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

```

### Transfer Learning
```python
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
```

## Appendix

## Manual Setup

### Create VM and Jupyter Notebook

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
 --restart-on-failure \
 --zone=us-central1-c
```

After a minute, we should be able to connect: 

```
gcloud config set compute/zone us-central1-c
gcloud compute ssh [vm-name]
```

### Download code and install dependencies

We need to setup CUDA so Tensorflow can use our image.  The following commands,
run on the evaluation VM, will install CUDA and Tensorflow on our GPU VM. After 
the installation finishes, we recommend you restart the VM.

```bash
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
pip install tensorflow-gpu==1.5
pip install numpy pandas keras jupyter matplotlib datalab h5py
HERE

sudo bash /tmp/setup.sh
```

```bash
sudo shutdown -r now
```

```bash
gcloud compute ssh [vm-name]
```

On the newly restarted VM, make sure NVIDIA card is configured correctly:
```bash
nvidia-smi
```

Download code:
```bash
git clone https://github.com/dhodun/ml-teaching-examples.git
```

### Start Jupyter Notebook and Connect

On VM:
```bash
jupyter notebook --port 8088
```

In the Google Cloud console click "Web Preview" button on upper right of Cloud Shell and click 'Preview on port 8080' to open Jupyter Window