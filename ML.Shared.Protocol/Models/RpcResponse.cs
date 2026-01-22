namespace ML.Shared.Protocol;

public sealed record RpcResponse(int v, string id, bool ok, object? result, RpcError? error);
