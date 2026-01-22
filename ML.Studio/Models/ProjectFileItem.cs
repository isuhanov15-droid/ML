namespace ML.Studio.Models;

public sealed record ProjectFileItem(
    string RelativePath,
    string Kind,
    string FullPath,
    string DisplayName);
