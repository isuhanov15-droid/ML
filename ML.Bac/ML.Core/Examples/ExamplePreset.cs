using ML.Core;

namespace ML.Core.Examples;

public sealed class ExamplePreset
{
    public required string Name { get; init; }
    public string Description { get; init; } = "";
    public int InputSize { get; init; }
    public int OutputSize { get; init; }
    public string HiddenSizes { get; init; } = "4";
    public ActivationType Activation { get; init; } = ActivationType.ReLu;
    public double LearningRate { get; init; } = 0.01;
    public int Epochs { get; init; } = 100;
    public int BatchSize { get; init; } = 4;
    public int Accumulation { get; init; } = 1;
    public bool Shuffle { get; init; } = true;
    public bool DropLast { get; init; } = false;
    public bool RequiresDatasetPath { get; init; }
    public Func<string?, IEnumerable<(double[] x, int y)>> TrainFactory { get; init; } = _ => Array.Empty<(double[], int)>();
    public Func<string?, IEnumerable<(double[] x, int y)>?>? ValFactory { get; init; }

    public IEnumerable<(double[] x, int y)> BuildTrain(string? datasetPath) => TrainFactory(datasetPath);
    public IEnumerable<(double[] x, int y)>? BuildVal(string? datasetPath) => ValFactory?.Invoke(datasetPath);
}
