## Docker images

The experiment image is built on top of the AWS Deep Learning Container:

```
public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2
```

### Build the experiment image

```bash
# Using the build script (content-addressed SHA tags):
./scripts/build_image.sh              # build + push
./scripts/build_image.sh --no-push    # build only

# Or manually:
docker build -f docker/Dockerfile.experiments -t ${REGISTRY}/${USER}/dist-train/experiments:latest .
docker push ${REGISTRY}/${USER}/dist-train/experiments:latest
```

Make sure `REGISTRY` is set in your environment (via `dev.env`) before running.
