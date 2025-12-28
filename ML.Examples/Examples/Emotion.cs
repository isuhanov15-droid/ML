using System;
using System.Linq;
using ML.Core;
using ML.Core.Data;
using ML.Core.Training;
using ML.Core.Utils;

namespace ML.Examples;

static class Emotion
{
    // 8 эмоций
    public enum Kind
    {
        Neutral = 0,
        Joy = 1,
        Smile = 2,
        Laugh = 3,
        Fear = 4,
        Pain = 5,
        Suffering = 6,
        Anger = 7
    }

    public static readonly string[] Names =
    {
        "Neutral","Joy","Smile","Laugh","Fear","Pain","Suffering","Anger"
    };

    // индексы признаков
    public const int Threat = 0;
    public const int Loss = 1;
    public const int PhysicalDiscomfort = 2;
    public const int SocialSupport = 3;
    public const int Energy = 4;
    public const int Control = 5;
    public const int Surprise = 6;
    public const int Humor = 7;
    public const int GoalProgress = 8;
    public const int Fatigue = 9;

    public const int InputSize = 10;
    public const int Classes = 8;

    // ручная сборка входа (Threat=0.1, Loss=0.5 ...)
    public static double[] Make(
        double threat, double loss, double physicalDiscomfort,
        double socialSupport, double energy, double control,
        double surprise, double humor, double goalProgress, double fatigue)
    {
        return new[]
        {
            Clamp01(threat),
            Clamp01(loss),
            Clamp01(physicalDiscomfort),
            Clamp01(socialSupport),
            Clamp01(energy),
            Clamp01(control),
            Clamp01(surprise),
            Clamp01(humor),
            Clamp01(goalProgress),
            Clamp01(fatigue)
        };
    }

    // генерация датасета (синтетика)
    public static (double[] X, int Y)[] GenerateSamples(int count, int seed = 42, double noiseStd = 0.03)
    {
        var rnd = new Random(seed);
        var arr = new (double[] X, int Y)[count];

        for (int i = 0; i < count; i++)
            arr[i] = GenerateOne(rnd, noiseStd);

        return arr;
    }

    private static (double[] X, int Y) GenerateOne(Random rnd, double noiseStd)
    {
        var x = new double[InputSize];
        for (int i = 0; i < x.Length; i++)
            x[i] = rnd.NextDouble();

        // корреляции
        x[Energy] = Clamp01(x[Energy] * (1.0 - 0.35 * x[Fatigue]));
        x[Control] = Clamp01(x[Control] * (0.8 + 0.3 * x[SocialSupport]));

        // шум
        for (int i = 0; i < x.Length; i++)
            x[i] = Clamp01(x[i] + NextGaussian(rnd, 0, noiseStd));

        // scores эмоций (учитель)
        double sNeutral =
            PeakMid(x[Threat]) * PeakMid(x[Loss]) * PeakMid(x[PhysicalDiscomfort]) *
            PeakMid(x[Surprise]) * PeakMid(x[Humor]) * PeakMid(x[GoalProgress]);

        double sJoy =
            1.3 * x[GoalProgress] * x[SocialSupport] * x[Energy] * (1.0 - x[Threat]);

        double sSmile =
            1.1 * x[SocialSupport] * (0.7 + 0.3 * x[Energy]) * (1.0 - x[Threat]) * (1.0 - x[Loss]);

        double sLaugh =
            1.4 * x[Humor] * x[Surprise] * (0.8 + 0.2 * x[SocialSupport]) * (1.0 - x[Threat]);

        double sFear =
            1.5 * x[Threat] * (1.0 - x[Control]) * (1.0 - x[SocialSupport]) * (0.6 + 0.4 * x[Surprise]);

        double sPain =
            1.4 * x[PhysicalDiscomfort] * (0.7 + 0.3 * x[Fatigue]) * (1.0 - 0.5 * x[SocialSupport]);

        double sSuffering =
            1.6 * x[Loss] * x[Fatigue] * (1.0 - x[SocialSupport]) * (0.7 + 0.3 * (1.0 - x[Control]));

        double sAnger =
            1.5 * (1.0 - x[Control]) * (1.0 - x[GoalProgress]) * (0.6 + 0.4 * x[Energy]) *
            (0.7 + 0.3 * x[Threat]);

        sNeutral *= 1.05;

        var scores = new[] { sNeutral, sJoy, sSmile, sLaugh, sFear, sPain, sSuffering, sAnger };
        int y = ArgMax(scores);

        // 5% “грязи” для правдоподобия
        if (rnd.NextDouble() < 0.05)
        {
            int k = rnd.Next(0, x.Length);
            x[k] = Clamp01(x[k] + NextGaussian(rnd, 0, 0.12));
        }

        return (x, y);
    }

    // =============================
    // RUN: витрина библиотеки
    // =============================
    public static void Run()
    {
        Console.WriteLine("=== TEST 4: Emotions (Softmax) ===");

        // 1) Dataset
        var samples = GenerateSamples(count: 20000, seed: 10, noiseStd: 0.03);
        var all = new Dataset(samples);

        // 2) Train/Test split
        var (train, test) = all.Split(trainRatio: 0.85, seed: 42);
        Console.WriteLine($"Dataset: total={samples.Length}, train={train.Samples.Count()}, test={test.Samples.Count()}");

        // 3) Network
        var net = BuildEmotions();

        // 4) Trainer + config
        var trainer = new Trainer(net, lr: 0.0005, useMinMax: false);

        var cfg = new TrainerConfig
        {

            Epochs = 10,
            LogEvery = 1,

            CheckpointPath = "emotion_model.json",
            CheckpointEvery = 1,
            AutoLoadCheckpoint = true,
            SaveCheckpoint = true,

            EnableStopFile = true,
            StopFileName = "STOP",
            DeleteStopFileOnExit = true,

            StopCheckInsideEpoch = true,
            StopCheckEverySamples = 128,

            // Dataset v2 features (если они у тебя уже добавлены)
            ShuffleEachEpoch = true,
            ShuffleSeed = 42,
            BatchSize = 128
        };

        // 5) Train
        trainer.Train(train, cfg);

        // 6) Metrics
        var accTrain = Metrics.Accuracy(net, train);
        var accTest = Metrics.Accuracy(net, test);
        Console.WriteLine($"\nAccuracy: train={accTrain:P2}, test={accTest:P2}");

        // 7) Manual input
        Console.WriteLine("\n--- Manual input test ---");
        var manual = Make(
            threat: 0.1,
            loss: 0.5,
            physicalDiscomfort: 0.1,
            socialSupport: 0.6,
            energy: 0.5,
            control: 0.6,
            surprise: 0.2,
            humor: 0.2,
            goalProgress: 0.4,
            fatigue: 0.5
        );

        var probs = net.Forward(manual);
        Console.WriteLine($"Predicted: {Names[ArgMax(probs)]}");
        


        PrintTopK(probs, k: 5);

        // 8) Scenarios
        Console.WriteLine("\n--- Scenarios ---");
        var scenarios = new (string Name, double[] X)[]
        {
            ("WarmSuccess", Make(0.1,0.1,0.1, 0.9,0.8,0.8, 0.2,0.3,0.9, 0.2)),
            ("DarkNoiseAlone", Make(0.9,0.1,0.2, 0.1,0.6,0.2, 0.7,0.1,0.4, 0.3)),
            ("HeavyLossTired", Make(0.3,0.9,0.2, 0.2,0.2,0.3, 0.2,0.0,0.2, 0.9)),
            ("JokeAndSurprise", Make(0.05,0.1,0.0, 0.6,0.8,0.7, 0.9,0.95,0.6, 0.2)),
            ("PhysicalPain", Make(0.2,0.1,0.95, 0.3,0.3,0.5, 0.2,0.0,0.3, 0.7)),
            ("BlockedAndIrritated", Make(0.6,0.2,0.1, 0.3,0.8,0.15, 0.3,0.1,0.1, 0.4)),
        };

        foreach (var (name, x) in scenarios)
        {
            var p = net.Forward(x);
            Console.WriteLine($"\nScenario: {name}");
            Console.WriteLine($"Predicted: {Names[ArgMax(p)]}");
            PrintTopK(p, 3);
        }

        Console.WriteLine();

        Console.WriteLine("\n--- Save/Load v2 check ---");

        var modelPath = "emotion_model_v2.json";

        // 1) Save current net
        net.Save(modelPath);
        Console.WriteLine($"Saved: {modelPath}");

        // 2) Load into fresh network (no manual architecture)
        var net2 = new Network();
        net2.Load(modelPath);
        Console.WriteLine("Loaded into fresh network.");

        // 3) Compare on same input
        var p1 = net.Forward(manual);
        var p2 = net2.Forward(manual);

        Console.WriteLine($"ArgMax original: {Emotion.Names[Emotion.ArgMax(p1)]}");
        Console.WriteLine($"ArgMax loaded:   {Emotion.Names[Emotion.ArgMax(p2)]}");

        double maxDiff = 0;
        for (int i = 0; i < p1.Length; i++)
        {
            var d = Math.Abs(p1[i] - p2[i]);
            if (d > maxDiff) maxDiff = d;
        }
        Console.WriteLine($"Max prob diff: {maxDiff:E6}");

    }

    private static Network BuildEmotions()
    {
        var net = new Network(InputSize);
        net.AddInputLayer();
        net.AddHiddenLayer(32, ActivationType.ReLu);
        net.AddHiddenLayer(16, ActivationType.ReLu);
        net.AddOutputLayer(Classes, ActivationType.Linear);
        net.AddSoftmax();
        return net;
    }

    // =============================
    // Utils
    // =============================
    public static int ArgMax(double[] v)
    {
        int idx = 0;
        double max = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > max) { max = v[i]; idx = i; }
        return idx;
    }

    public static void PrintTopK(double[] probs, int k = 3)
    {
        var top = probs.Select((p, i) => (p, i))
                       .OrderByDescending(t => t.p)
                       .Take(k);

        foreach (var (p, i) in top)
            Console.WriteLine($"{Names[i],10}: {p:F3}");
    }

    private static double PeakMid(double v)
    {
        double t = v - 0.5;
        double r = 1.0 - 4.0 * t * t;
        return r < 0 ? 0 : r;
    }

    private static double Clamp01(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);

    private static double NextGaussian(Random rnd, double mean, double stdDev)
    {
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * z;
    }

    static void DebugProbs(string tag, double[] p)
{
    double sum = 0;
    double min = double.PositiveInfinity;
    double max = double.NegativeInfinity;

    for (int i = 0; i < p.Length; i++)
    {
        sum += p[i];
        if (p[i] < min) min = p[i];
        if (p[i] > max) max = p[i];
    }

    Console.WriteLine($"{tag}: sum={sum:F6}, min={min:F6}, max={max:F6}");
}

}
