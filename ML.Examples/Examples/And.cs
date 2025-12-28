
using ML.Core;
using ML.Core.Data;
using ML.Core.Training;
using ML.Core.Utils;

namespace ML.Examples;

public static class And
{
    public static void Run()
    {
        Console.WriteLine("=== TEST: AND ===");

        var data = new Dataset(new[]
        {
            (new double[]{0,0}, 0),
            (new double[]{0,1}, 0),
            (new double[]{1,0}, 0),
            (new double[]{1,1}, 1),
        });

        var net = Build();
        var trainer = new Trainer(net, lr: 0.01);

        trainer.Train(data, epochs: 2000, logEvery: 100);

        Evaluate(net, data);
        Console.WriteLine();
    }

    private static Network Build()
    {
        var net = new Network(2);
        net.AddInputLayer();
        net.AddHiddenLayer(4, ActivationType.ReLu);
        net.AddOutputLayer(2, ActivationType.Linear);
        net.AddSoftmax();
        return net;
    }

    private static void Evaluate(Network net, Dataset data)
    {
        foreach (var (x, y) in data.Samples)
        {
            var input = Normalizer.MinMax(x);
            var probs = net.Forward(input);

            int pred = ArgMax(probs);

            Console.WriteLine(
                $"Input [{string.Join(",", x)}] → predicted={pred}, target={y}, " +
                $"probs=[{string.Join(", ", probs.Select(p => p.ToString("F3")))}]"
            );
        }
    }

    private static int ArgMax(double[] v)
    {
        int idx = 0;
        double max = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > max) { max = v[i]; idx = i; }
        return idx;
    }
}
