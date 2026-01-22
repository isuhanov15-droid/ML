namespace ML.Shared.Protocol;

public sealed record HostInfoDto(
    string appName,
    string hostVersion,
    int protocolVersion,
    string machineName,
    string osDescription,
    string framework,
    DateTimeOffset utcNow);
