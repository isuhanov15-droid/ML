namespace ML.Shared.Protocol;

public sealed record RunDto(
    string runId,
    string projectId,
    string experimentId,
    string state,
    DateTime? startedUtc,
    DateTime? endedUtc,
    int lastEpoch,
    double? lastLoss,
    string? note);
