using System.Text.Json;

namespace ML.Shared.Protocol;

public static class ProtocolJson
{
    public static JsonSerializerOptions Options => JsonOptions.Default;
}
