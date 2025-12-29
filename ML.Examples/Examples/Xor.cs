using System;
using System.Linq;
using ML.Core;
using ML.Core.Layers;
using ML.Core.Abstractions;
using ML.Core.Optimizers;
using ML.Core.Training;
using ML.Core.Training.Callbacks;
using ML.Core.Losses;
using ML.Core.Utils;

namespace ML.Examples;

public static class Xor
{
    public static void Run()
    {
        Console.WriteLine("=== TEST: XOR ===");

        var data = new (double[] x, int y)[]
        {
            (new double[]{0,0}, 0),
            (new double[]{0,1}, 1),
            (new double[]{1,0}, 1),
            (new double[]{1,1}, 0),
        };

        var dataset = data.Select(s => (x: Normalizer.MinMax(s.x), y: s.y)).ToArray();

        var model = Build();
        var optimizer = new AdamOptimizer(learningRate: 0.01);
        ILoss loss = new CrossEntropyLoss();

        var trainer = new Trainer(model, optimizer, loss);

        var callbacks = new[]
        {
            new Callback(model, dataset, every: 100)
        };

        trainer.Train(dataset, epochs: 2000, callbacks: callbacks);

        Evaluate(model, dataset);
        Console.WriteLine();
    }

    private static Network Build()
    {
        var net = new Network();
        net.Add(new LinearLayer(2, 8));
        net.Add(new ActivationLayer(8, ActivationType.ReLu));
        net.Add(new LinearLayer(8, 2));
        net.Add(new SoftmaxLayer(2));
        return net;
    }

    private static void Evaluate(IModel model, (double[] x, int y)[] data)
    {
        foreach (var (x, y) in data)
        {
            var probs = model.Forward(x, training: false);
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
