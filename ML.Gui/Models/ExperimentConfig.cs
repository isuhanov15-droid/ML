using ML.Core;

namespace ML.Gui.Models;

public sealed class ExperimentConfig
{
    public string? PresetName { get; init; }
    public string? DatasetPath { get; init; }

    public int InputSize { get; init; }
    public int OutputSize { get; init; }
    public string HiddenSizes { get; init; } = "4";
    public ActivationType Activation { get; init; } = ActivationType.ReLu;

    public int Epochs { get; init; }
    public int BatchSize { get; init; }
    public int AccumulationSteps { get; init; }
    public int EpochDisplayEvery { get; init; }
    public bool Shuffle { get; init; }
    public bool DropLast { get; init; }

    public double LearningRate { get; init; }
    public double? GradClipNorm { get; init; }
    public int? Seed { get; init; }

    public string ModelName { get; init; } = "model";
    public string SavePath { get; init; } = "";
    public string LoadPath { get; init; } = "";
}
