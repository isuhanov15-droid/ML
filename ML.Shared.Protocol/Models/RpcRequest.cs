namespace ML.Shared.Protocol;

public sealed record RpcRequest(int v, string id, string method, object? @params);
