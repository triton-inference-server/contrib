using Grpc.Net.Client;
using Inference;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using static Inference.ModelInferRequest.Types;

namespace test
{
	internal class SimpleCSharpclient
	{
		static void Main(string[] args)
		{
			var channel = GrpcChannel.ForAddress("http://localhost:8001");
			Inference.GRPCInferenceService.GRPCInferenceServiceClient tritonInferenceClient = new Inference.GRPCInferenceService.GRPCInferenceServiceClient(channel);

			// Check if Triton server is live
			ServerLiveRequest serverLiveRequest = new ServerLiveRequest();
			ServerLiveResponse serverLiveResponse = tritonInferenceClient.ServerLive(serverLiveRequest);
			if (serverLiveResponse is null || serverLiveResponse.Live == false)
			{
				throw new ApplicationException("Triton Server not alive, exiting");
			}

			// Check if Triton server is ready
			ServerReadyRequest serverReadyRequest = new ServerReadyRequest();
			ServerReadyResponse serverReadyResponse = tritonInferenceClient.ServerReady(serverReadyRequest);
			if (serverReadyResponse is null || serverReadyResponse.Ready == false)
			{
				throw new ApplicationException("Triton Server is not ready, exiting");
			}

			// Generate the request
			ModelInferRequest modelInferRequest = new ModelInferRequest();
			modelInferRequest.ModelName = "TestModelName";
			modelInferRequest.ModelVersion = "1";
			int batchCount = 1;
			int inputMaxLength = 16;
			long[] input_ids1 = Enumerable.Range(0, batchCount * inputMaxLength).Select(n => 1L).ToArray();
			long[] input_ids2 = Enumerable.Range(0, batchCount * inputMaxLength).Select(n => 2L).ToArray();

			// Populate the inputs in Triton inference request
			InferInputTensor inferInputTensor = new InferInputTensor();
			inferInputTensor.Name = "Input1";
			inferInputTensor.Datatype = "INT64";
			inferInputTensor.Shape.Add(batchCount);
			inferInputTensor.Shape.Add(inputMaxLength);
			var inputTensorContents = new InferTensorContents();
			inputTensorContents.Int64Contents.Add(input_ids1);
			inferInputTensor.Contents = inputTensorContents;

			InferInputTensor inferInputTensor1 = new InferInputTensor();
			inferInputTensor1.Name = "Input2";
			inferInputTensor1.Datatype = "INT64";
			inferInputTensor1.Shape.Add(batchCount);
			inferInputTensor1.Shape.Add(inputMaxLength);
			var inputTensorContents1 = new InferTensorContents();
			inputTensorContents1.Int64Contents.Add(input_ids2);
			inferInputTensor1.Contents = inputTensorContents1;

			modelInferRequest.Inputs.AddRange(new List<InferInputTensor> { inferInputTensor, inferInputTensor1 });

			// Populate the outputs in the inference request
			InferRequestedOutputTensor inferRequestedOutputTensor = new InferRequestedOutputTensor();
			inferRequestedOutputTensor.Name = "Output";
			modelInferRequest.Outputs.Add(inferRequestedOutputTensor);


			ModelInferResponse modelInferResponse = new ModelInferResponse();
			modelInferResponse = tritonInferenceClient.ModelInfer(modelInferRequest);

			// Get the response outputs
			// Assume Output is fp32
			byte[] byteArray = modelInferResponse.RawOutputContents.ToList().First().ToByteArray();
			if (!BitConverter.IsLittleEndian)
			{
				Array.Reverse(byteArray);
			}
			float[] floatArray = new float[byteArray.Length / sizeof(float)];
			Buffer.BlockCopy(byteArray, 0, floatArray, 0, byteArray.Length);
			List<float> pre_responses = new List<float>(floatArray);
			int outputCount = (int)modelInferResponse.Outputs.First().Shape[0];
			int outputDimension = (int)modelInferResponse.Outputs.First().Shape[1];
			_ = Parallel.For(0, outputCount, j =>
			{
				List<float> vector = pre_responses.GetRange(j * outputDimension, outputDimension);
				// Do postprocess with the vector
			});
		}
	}
}