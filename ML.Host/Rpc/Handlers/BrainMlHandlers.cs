using System.Text.Json;
using ML.Host.Services;
using ML.Shared.Protocol;

namespace ML.Host.Rpc.Handlers;

public sealed class BrainMlHandlers
{
    private readonly BrainMlService _service;

    public BrainMlHandlers(BrainMlService service)
    {
        _service = service;
    }

    public Task<object?> InferAsync(RpcRequest req)
    {
        var request = Deserialize<MlInferRequest>(req);
        var response = _service.Infer(request);
        return Task.FromResult<object?>(response);
    }

    public Task<object?> TrainAsync(RpcRequest req)
    {
        var request = Deserialize<MlTrainRequest>(req);
        var response = _service.Train(request);
        return Task.FromResult<object?>(response);
    }

    public Task<object?> SaveAsync(RpcRequest req)
    {
        var request = Deserialize<MlCheckpointRequest>(req);
        var response = _service.Save(request);
        return Task.FromResult<object?>(response);
    }

    public Task<object?> LoadAsync(RpcRequest req)
    {
        var request = Deserialize<MlCheckpointRequest>(req);
        var response = _service.Load(request);
        return Task.FromResult<object?>(response);
    }

    public Task<object?> PingAsync(RpcRequest req)
    {
        return Task.FromResult<object?>(new { status = "ok" });
    }

    private static T Deserialize<T>(RpcRequest req)
    {
        if (req.@params is not JsonElement el)
            throw new InvalidOperationException("Invalid payload.");

        var value = JsonSerializer.Deserialize<T>(el.GetRawText(), ML.Shared.Protocol.JsonOptions.Default);
        if (value == null)
            throw new InvalidOperationException("Invalid payload.");
        return value;
    }
}
