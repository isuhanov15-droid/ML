using System.Linq;
using ML.Core.Data;

namespace ML.Core.Examples;

public static class ExampleRegistry
{
    public static IReadOnlyList<ExamplePreset> Presets { get; } = new List<ExamplePreset>
    {
        new()
        {
            Name = "XOR (демо)",
            Description = "2 входа, 2 класса; синтетический датасет XOR",
            InputSize = 2,
            OutputSize = 2,
            HiddenSizes = "4",
            Activation = ActivationType.ReLu,
            LearningRate = 0.01,
            Epochs = 5000,
            BatchSize = 4,
            Accumulation = 1,
            TrainFactory = _ => DemoDatasets.Xor,
            ValFactory = _ => DemoDatasets.Xor
        },
        new()
        {
            Name = "AND (демо)",
            Description = "2 входа, логический AND",
            InputSize = 2,
            OutputSize = 2,
            HiddenSizes = "4",
            Activation = ActivationType.ReLu,
            LearningRate = 0.01,
            Epochs = 2000,
            BatchSize = 4,
            Accumulation = 1,
            TrainFactory = _ => DemoDatasets.And
        },
        new()
        {
            Name = "Пороговая классификация (демо)",
            Description = "1 вход, два класса, разделение по порогу 0.5",
            InputSize = 1,
            OutputSize = 2,
            HiddenSizes = "3",
            Activation = ActivationType.ReLu,
            LearningRate = 0.01,
            Epochs = 1000,
            BatchSize = 8,
            Accumulation = 1,
            TrainFactory = _ => DemoDatasets.Threshold().ToArray()
        },
        new()
        {
            Name = "Пользовательский датасет (CSV/JSON)",
            Description = "Файл с фичами и меткой в последнем столбце или поле Y",
            InputSize = 2,
            OutputSize = 2,
            HiddenSizes = "8",
            Activation = ActivationType.ReLu,
            LearningRate = 0.01,
            Epochs = 500,
            BatchSize = 16,
            Accumulation = 1,
            RequiresDatasetPath = true,
            TrainFactory = path => DatasetLoader.LoadClassification(path ?? throw new InvalidOperationException("Укажите путь к датасету")).ToArray()
        }
    };
}
