namespace ML.Shared.Protocol;

public sealed record RpcEvent(int v, string type, object? data);
