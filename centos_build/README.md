<!--
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

This is a docker build process for Centos7.  It's experimental and not officially supported yet.

## How it works?  
In order to support centos, we need to prepare a dedicated minimum image for container build.  
Apart from the minmum image, the main build.py script need to be compatible with centos. Currently there are two main modifications:
1. use yum to replace apt-get commands accordingly.
2. dcgm installation command is changed. This will be in building base image stage.

## Currently tested build
1. centos7 server 21.11
2. centos7 server 21.11 + tensorrt backend + python backend + tf1 backend + tf2 backend + pytorch backend
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

## How to build tf1/tf2/pytorch backends
```bash
docker build -f backend_images/tf1/Dockerfile.centos7.tf1 -t tritonbuild_centos_tf1 .
```
For tf2/pytorch, replace tf1 with tf2/pytorch in this command.  
Once you build the docker image for tensorflow backends, just add "--backend=tensorflow1" and "--image=tensorflow1,tritonbuild_centos_tf1" to build.py

## To be released soon
- [x] tf1.15 base image based on ngc ubuntu container, which uses nvtf, https://github.com/NVIDIA/tensorflow
- [x] tf2 base image based on ngc ubuntu container
- [x] PyTorch base image based on PyTorch official repo
- [x] Triton Clients

## TODO
- [ ] OpenVINO backend
- [ ] Dali backend
- [ ] ORT backend
- [ ] Paddlepaddle backend
- [ ] FT backend
- [ ] HugeCTR backend