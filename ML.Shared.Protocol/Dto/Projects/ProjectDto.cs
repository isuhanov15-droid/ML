namespace ML.Shared.Protocol;

public sealed record ProjectDto(
    string projectId,
    string name,
    DateTime createdUtc,
    DateTime updatedUtc);
