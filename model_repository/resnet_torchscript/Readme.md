# Loading Pytorch Models To Nvidia Triton Inference Server


Pytorch model after conversion to TorchScript or ONNX format are supported by the Nvidia [Triton Inference Server](https://github.com/triton-inference-server/). 
Here we will see how to convert the Pytorch Model to the Torchscript format and load it to Triton.

TorchScript is a way to create serializable and optimizable models from PyTorch code.  Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.

##  Converting a PyTorch Model defined in Python to TorchScript

## Create and load the model
```python
import torch
from torchvision.models import resnet50
import os
```
To quickly come to conclusion, we will be using Resnet-50 model from the torchvision package.

```python
model  = resnet50(pretrained=True).cuda()
```

### Create test data, and trace the model
Test data to just check the model is working and there's no error.

```python
test_data = torch.randn(1,3,224,224).cuda()
print("Output Shape : ", model(test_data).shape)
```

    Output Shape :  torch.Size([1, 1000])

```python
traced_model = torch.jit.trace(model, (test_data))
```


```python
print("Traced Model :", traced_model.graph) # Printing the graph
```

    Traced Model : graph(%self.1 : __torch__.torchvision.models.resnet.ResNet,
          %input.1 : Float(1:150528, 3:50176, 224:224, 224:1)):
      %3671 : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="fc"](%self.1)
      %3668 : __torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d = prim::GetAttr[name="avgpool"](%self.1)
      %3667 : __torch__.torch.nn.modules.container.___torch_mangle_141.Sequential = prim::GetAttr[name="layer4"](%self.1)
      %3579 : __torch__.torch.nn.modules.container.___torch_mangle_113.Sequential = prim::GetAttr[name="layer3"](%self.1)
      %3413 : __torch__.torch.nn.modules.container.___torch_mangle_61.Sequential = prim::GetAttr[name="layer2"](%self.1)
      %3299 : __torch__.torch.nn.modules.container.___torch_mangle_25.Sequential = prim::GetAttr[name="layer1"](%self.1)
      %3211 : __torch__.torch.nn.modules.pooling.MaxPool2d = prim::GetAttr[name="maxpool"](%self.1)
      %3210 : __torch__.torch.nn.modules.activation.ReLU = prim::GetAttr[name="relu"](%self.1)
      %3209 : __torch__.torch.nn.modules.batchnorm.BatchNorm2d = prim::GetAttr[name="bn1"](%self.1)
      %3203 : __torch__.torch.nn.modules.conv.Conv2d = prim::GetAttr[name="conv1"](%self.1)
      %3854 : Tensor = prim::CallMethod[name="forward"](%3203, %input.1)
      %3855 : Tensor = prim::CallMethod[name="forward"](%3209, %3854)
      %3856 : Tensor = prim::CallMethod[name="forward"](%3210, %3855)
      %3857 : Tensor = prim::CallMethod[name="forward"](%3211, %3856)
      %3858 : Tensor = prim::CallMethod[name="forward"](%3299, %3857)
      %3859 : Tensor = prim::CallMethod[name="forward"](%3413, %3858)
      %3860 : Tensor = prim::CallMethod[name="forward"](%3579, %3859)
      %3861 : Tensor = prim::CallMethod[name="forward"](%3667, %3860)
      %3862 : Tensor = prim::CallMethod[name="forward"](%3668, %3861)
      %3013 : int = prim::Constant[value=1]() # /root/miniconda3/lib/python3.6/site-packages/torchvision/models/resnet.py:214:0
      %3014 : int = prim::Constant[value=-1]() # /root/miniconda3/lib/python3.6/site-packages/torchvision/models/resnet.py:214:0
      %input : Float(1:2048, 2048:1) = aten::flatten(%3862, %3013, %3014) # /root/miniconda3/lib/python3.6/site-packages/torchvision/models/resnet.py:214:0
      %3863 : Tensor = prim::CallMethod[name="forward"](%3671, %input)
      return (%3863)
    


## Saving the Model

An TorchScript model is a single file that by default must be named `model.pt`.A minimal model repository for a single PyTorch model would look like:

```bash
<model-repository-path>/
  <model-name>/
    config.pbtxt
    1/
      model.pt
```


```python
model_dir = "1" # making directory as required by Triton 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
```

First convert your model from GPU to CPU and then save it. This is recommended because the tracer may witness tensor creation on a specific device, so casting an already-loaded model may have unexpected effects. Casting the model before saving it ensures that the tracer has the correct device information.


```python
torch.jit.save(traced_model, os.path.join(model_dir,"model.pt"))
```

## Config File Creation

Each model input and output must specify a name, datatype, and shape.

The name specified for an input or output tensor must match the name expected by the model. PyTorch config I/O naming convention looks like `<name>__<index>` refer to the [documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/model_configuration.html?highlight=output__1#inputs-and-outputs) for details. 

Here is the config.pbtxt file corresponding to our Resnet-50 model

``` json
name: "resnet-torchscript"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [3, 224, 224 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1000 ]
    label_filename: "resnet_labels.txt"
  }
]

```

`"resnet_labels.txt"` file has labels for the ImageNet data.

## Loading Model to Triton

Load the model and start the server : `tritonserver --model-repository=model_repository`

## Doing Performance Test

After you have Triton running you can send inference and other requests to it using the HTTP/REST or GRPC protocols from your Nvidia [Triton Client application](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/client_example.html).

NVIDIA Triton Client SDK contains a [performance analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/perf_analyzer.html) which can be used to run experiments to measure inference throughput and latency in different scenarios.


```python
perf_client -m resnet_torchscript -b 8 --concurrency-range 1:4 # Concurrency test
perf_client -m resnet_torchscript  --shape IMAGE:3,224,224 # using image like request
```

# Deploying More Pytorch Models
In the similar way you can load any model from `torchvision.models` and post conversion to TorchScript or ONNX it can be deployed to Triton. 
