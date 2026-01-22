using System.Text.Json.Serialization;

namespace ML.Gui.Models;

public sealed class TrainStartConfig
{
    [JsonPropertyName("network")]
    public TrainNetworkConfig Network { get; set; } = new();

    [JsonPropertyName("train")]
    public TrainTrainConfig Train { get; set; } = new();

    [JsonPropertyName("data")]
    public TrainDataConfig Data { get; set; } = new();

    [JsonPropertyName("resume")]
    public bool Resume { get; set; }
}

public sealed class TrainNetworkConfig
{
    [JsonPropertyName("inputSize")]
    public int InputSize { get; set; }

    [JsonPropertyName("outputSize")]
    public int OutputSize { get; set; }

    [JsonPropertyName("hidden")]
    public int[] Hidden { get; set; } = Array.Empty<int>();

    [JsonPropertyName("activation")]
    public string Activation { get; set; } = "ReLu";

    [JsonPropertyName("seed")]
    public int? Seed { get; set; }
}

public sealed class TrainTrainConfig
{
    [JsonPropertyName("epochs")]
    public int Epochs { get; set; }

    [JsonPropertyName("batchSize")]
    public int BatchSize { get; set; }

    [JsonPropertyName("shuffle")]
    public bool Shuffle { get; set; }

    [JsonPropertyName("dropLast")]
    public bool DropLast { get; set; }

    [JsonPropertyName("gradClipNorm")]
    public double? GradClipNorm { get; set; }

    [JsonPropertyName("accumulationSteps")]
    public int AccumulationSteps { get; set; }

    [JsonPropertyName("learningRate")]
    public double LearningRate { get; set; }

    [JsonPropertyName("uiEveryNEpochs")]
    public int UiEveryNEpochs { get; set; } = 1;
}

public sealed class TrainDataConfig
{
    [JsonPropertyName("preset")]
    public string Preset { get; set; } = "XOR";

    [JsonPropertyName("datasetPath")]
    public string? DatasetPath { get; set; }
}
