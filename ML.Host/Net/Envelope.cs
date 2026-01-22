using System.Text.Json;
using System.Text.Json.Serialization;

namespace ML.Host.Net;

public sealed class Envelope
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "";

    [JsonPropertyName("rid")]
    public string? Rid { get; set; }

    [JsonPropertyName("ts")]
    public DateTimeOffset Ts { get; set; }

    [JsonPropertyName("data")]
    public JsonElement Data { get; set; }
}

public sealed class OutgoingEnvelope
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "";

    [JsonPropertyName("rid")]
    public string? Rid { get; set; }

    [JsonPropertyName("ts")]
    public DateTimeOffset Ts { get; set; }

    [JsonPropertyName("data")]
    public object? Data { get; set; }
}
