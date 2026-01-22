namespace ML.Studio.Models;

public sealed record StudioProject(
    string Name,
    string ProjectType,
    string RootPath,
    ProjectFileItem[] Files,
    DateTime CreatedUtc,
    DateTime UpdatedUtc);
