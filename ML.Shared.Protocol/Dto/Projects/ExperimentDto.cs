using System.Text.Json;

namespace ML.Shared.Protocol;

public sealed record ExperimentDto(
    string experimentId,
    string projectId,
    string name,
    string? description,
    DateTime createdUtc,
    DateTime updatedUtc,
    JsonElement? trainConfig,
    JsonElement? computeSpec);
