using ML.Core;

namespace ML.Gui.Examples;

public static class ExampleRegistry
{
    public static IReadOnlyList<ExamplePreset> Presets { get; } = new List<ExamplePreset>
    {
        new()
        {
            Name = "XOR",
            Description = "Классический XOR на 2 входа, 2 выхода",
            InputSize = 2,
            OutputSize = 2,
            HiddenSizes = "4",
            Activation = ActivationType.ReLu,
            LearningRate = 0.01,
            Epochs = 5000,
            BatchSize = 4,
            Accumulation = 1,
            TrainProvider = () => Services.DemoPipelines.XorData
        },
        new()
        {
            Name = "AND",
            Description = "Логическое AND, 2 входа, 2 класса",
            InputSize = 2,
            OutputSize = 2,
            HiddenSizes = "4",
            Activation = ActivationType.ReLu,
            LearningRate = 0.01,
            Epochs = 2000,
            BatchSize = 4,
            Accumulation = 1,
            TrainProvider = () => new (double[] x, int y)[]
            {
                (new[]{0.0,0.0}, 0),
                (new[]{0.0,1.0}, 0),
                (new[]{1.0,0.0}, 0),
                (new[]{1.0,1.0}, 1),
            }
        },
        new()
        {
            Name = "Threshold",
            Description = "1 вход, бинарная классификация по порогу 0.5",
            InputSize = 1,
            OutputSize = 2,
            HiddenSizes = "3",
            Activation = ActivationType.ReLu,
            LearningRate = 0.01,
            Epochs = 1000,
            BatchSize = 8,
            Accumulation = 1,
            TrainProvider = () =>
            {
                var data = new List<(double[] x, int y)>();
                for (int i = 0; i <= 10; i++)
                {
                    double v = i / 10.0;
                    int cls = v >= 0.5 ? 1 : 0;
                    data.Add((new[]{v}, cls));
                }
                return data;
            }
        }
        ,
        new()
        {
            Name = "Emotion (demo)",
            Description = "Синтетические фичи 52 -> 16 классов, для примера эмоций",
            InputSize = 52,
            OutputSize = 16,
            HiddenSizes = "80,32",
            Activation = ActivationType.ReLu,
            LearningRate = 0.005,
            Epochs = 200,
            BatchSize = 8,
            Accumulation = 1,
            TrainProvider = () => GenerateEmotionSynthetic(123, 128),
            ValProvider = () => GenerateEmotionSynthetic(321, 64)
        }
    };

    private static IEnumerable<(double[] x, int y)> GenerateEmotionSynthetic(int seed, int samples)
    {
        var rnd = new Random(seed);
        var data = new List<(double[] x, int y)>(samples);

        for (int i = 0; i < samples; i++)
        {
            var x = new double[52];
            for (int j = 0; j < x.Length; j++)
                x[j] = rnd.NextDouble();

            int y = rnd.Next(0, 16);
            data.Add((x, y));
        }

        return data;
    }
}
