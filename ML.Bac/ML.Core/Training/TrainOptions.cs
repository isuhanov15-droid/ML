using System.Collections.Generic;

namespace ML.Core.Training;

public sealed class TrainOptions
{
    public int Epochs { get; init; } = 1;
    public int BatchSize { get; init; } = 64;

    public bool Shuffle { get; init; } = true;
    public bool DropLast { get; init; } = false;

    /// <summary>Ограничение нормы градиента. null = не клипать.</summary>
    public double? GradClipNorm { get; init; } = null;

    /// <summary>Шаг градиентного накопления. 1 = обычный режим.</summary>
    public int GradientAccumulationSteps { get; init; } = 1;

    /// <summary>Seed для shuffle (и любых будущих источников случайности). null = Random.</summary>
    public int? Seed { get; init; } = null;

    /// <summary>Валидация — только Loss. Метрики (accuracy) считаются снаружи (Examples).</summary>
    public IEnumerable<(double[] x, int y)>? Validation { get; init; } = null;

    /// <summary>Callbacks ядра (без доменных метрик).</summary>
    public IEnumerable<ML.Core.Training.Callbacks.ITrainCallback>? Callbacks { get; init; } = null;
}
