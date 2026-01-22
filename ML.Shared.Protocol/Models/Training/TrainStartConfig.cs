namespace ML.Shared.Protocol.Training;

public sealed class TrainStartConfig
{
    public TrainNetworkConfig Network { get; set; } = new();
    public TrainTrainConfig Train { get; set; } = new();
    public TrainDataConfig Data { get; set; } = new();
    public bool Resume { get; set; }
    public string? ModelPath { get; set; }
}

public sealed class TrainNetworkConfig
{
    public int InputSize { get; set; }
    public int OutputSize { get; set; }
    public int[] Hidden { get; set; } = Array.Empty<int>();
    public string Activation { get; set; } = "ReLu";
    public int? Seed { get; set; }
}

public sealed class TrainTrainConfig
{
    public int Epochs { get; set; } = 1;
    public int BatchSize { get; set; } = 32;
    public bool Shuffle { get; set; } = true;
    public bool DropLast { get; set; }
    public double? GradClipNorm { get; set; }
    public int AccumulationSteps { get; set; } = 1;
    public double LearningRate { get; set; } = 0.001;
    public int UiEveryNEpochs { get; set; } = 1;
    public int? EarlyStopPatience { get; set; }
    public double? EarlyStopMinDelta { get; set; }
}

public sealed class TrainDataConfig
{
    public string Preset { get; set; } = "XOR";
    public string? DatasetPath { get; set; }
}
