# Example C# Client Using Generated GRPC API

## Prerequisites

Required NuGet Packages:\
`Grpc.Tools`, used to generate C# code for gRPC services.\
`Grpc.Net.Client`, used to communicate with the server.\
`Google.Protobuf`, used to read and write the protocol buffer messages.


## Generating C# GRPC client stub

Clone the [triton-inference-server/common](https://github.com/triton-inference-server/common/)
repository:

```
git clone https://github.com/triton-inference-server/common/ -b <common-repo-branch> common-repo
```

\<common-repo-branch\> should be the version of the Triton server that you
intend to use (e.g. r23.12).

Copy __*.proto__ files to ./Protos

```
$ cd your-project-folder
$ mkdir Protos
$ cp triton-server-folder/common-repo/protobuf/*.proto ./Protos/
```

Make sure csproj has the newly added *.proto and then rebuild the solution:
```
  <ItemGroup>
    <Protobuf Include="Protos\grpc_service.proto" GrpcServices="Client" ProtoRoot="Protos\" />
    <Protobuf Include="Protos\health.proto" GrpcServices="Client" ProtoRoot="Protos\" />
    <Protobuf Include="Protos\model_config.proto" GrpcServices="Client" ProtoRoot="Protos\" />
  </ItemGroup>
```

Once compiled, one should notice the generated *.cs files (GrpcService.cs, GrpcServiceGrpc.cs, ModelConfig.cs, ModelConfigGrpc.cs, etc.) under obj folder, such as 'obj\x64\Debug\net6.0'. The C# example, 'SimpleCSharpclient.cs', provides details on how to use it.

