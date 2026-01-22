namespace ML.Shared.Protocol;

public sealed record RunMetricsPointDto(
    string runId,
    int epoch,
    double loss,
    double? accuracy,
    DateTime utc);
