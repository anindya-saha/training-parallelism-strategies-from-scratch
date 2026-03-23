Based off of https://github.com/aws/deep-learning-containers/blob/master/pytorch/training/docker/2.7/py3/cu128/Dockerfile.gpu

### How to create a base image and push for use (This assusmes you are logged in to Docker)
docker build -f Dockerfile.cuda-12.8-pytorch-2.7-py3.12 -t "mlp.docker.zooxlabs.com/asaha/cuda-12.8-pytorch-2.7-py3.12:1.0.0-rc1" .


docker build -f Dockerfile.cuda-12.8-pytorch-2.8-py3.12 -t "mlp.docker.zooxlabs.com/asaha/cuda-12.8-pytorch-2.8-py3.12:1.0.0-rc1" .


docker build -f Dockerfile -t "mlp.docker.zooxlabs.com/asaha/cuda-12.9-pytorch-2.8-py3.12:1.0.0-rc1" .

docker build -f Dockerfile.experiments -t "mlp.docker.zooxlabs.com/asaha/dist-train/experiments:1.0.0-rc1" .

docker push "${image_name}"