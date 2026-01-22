using System.Text.Json;

namespace ML.Core.Serialization;

public static class JsonOptions
{
    public static readonly JsonSerializerOptions Default = new()
    {
        WriteIndented = true
    };
}
