# mnist-recognition-pytorch
This repository is an example of pytorch written number characters recognition written in pytorch. It uses MNIST dataset for both the training and the evaluation. The training and the evaluation is kept in docker image.

Pleare read https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html prior going further.

## Instalation
Make sure you have installed docker prior running those scripts.
```
$ docker-compose build
```
This command will prepare docker image needed for both the training and the evaluation.

## Training and evaluation
The training is invoked by this script.
```
$ docker run --rm -it --init --ipc=host --volume="$PWD":/app mnist-pytorch:v1 python3 /app/src/train.py
```

It will train the network on the training dataset, store the network data to the "net.pth" file and run the evaluation on the trained network.

Similarly the evaluation is done by executing this command:
```
$ docker run --rm -it --init --ipc=host --volume="$PWD":/app mnist-pytorch:v1 python3 /app/src/eval.py
```

It will evaluate the trained network on testing dataset and print the accuracy.
