namespace ML.Shared.Protocol;

public sealed record ProjectFileDto(
    string path,
    long sizeBytes,
    DateTime updatedUtc);
