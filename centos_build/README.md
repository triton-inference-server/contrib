## How it works?  
In order to support centos, we need to prepare a dedicated minimum image for container build.  
Apart from the minmum image, the main build.py script need to be compatible with centos. Currently there are two main modifications:
1. use yum to replace apt-get commands accordingly.
2. dcgm installation command is changed. This will be in building base image stage.

## Currently tested build
1. centos7 server 21.11
2. centos7 server 21.11 + tensorrt backend + python backend
3. centos7 clients commit 87255faf0e9769b55a1282b5ac32820e66ee9326

## How to build server
1. Prepare minmum image:
```bash
docker build -f Dockerfile.centos7.min -t tritonbuilder_centos7_min
```
2. use build.py to replace build.py under server repo
3. build with command: 
```python
python3 build.py --verbose --build-dir=/path/to/triton_build -j 8 --enable-logging --enable-stats --enable-metrics --enable-gpu-metrics --enable-tracing --enable-nvtx --enable-gpu --min-compute-capability=7.0 --endpoint=grpc --endpoint=http --backend=tensorrt --backend=python --cmake-dir=/path/to/server/build --image=base,tritonbuilder_centos7_min --target-platform=centos7 --no-container-pull
```

## How to build clients
```bash
docker build -f Dockerfile.centos7.sdk -t tritonserver_centos_sdk
```

## To be released soon
- [x] tf1.15 base image based on ngc ubuntu container, which uses nvtf, https://github.com/NVIDIA/tensorflow
- [ ] tf2 base image based on ngc ubuntu container
- [ ] PyTorch base image based on PyTorch official repo
- [x] Triton Clients

## TODO
- [ ] OpenVINO backend
- [ ] Dali backend
- [ ] ORT backend
- [ ] Paddlepaddle backend
- [ ] FT backend
- [ ] HugeCTR backend