namespace ML.Shared.Protocol;

public sealed record RpcError(string code, string message, string? details);
