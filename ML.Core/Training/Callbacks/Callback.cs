using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core.Abstractions;
using ML.Core.Training.Collbacks;

namespace ML.Core.Training.Callbacks;


public sealed class Callback : ICallback
{
    private readonly IModel _model;
    private readonly IEnumerable<(double[] x, int y)> _data;
    private readonly int _every;

    public Callback(IModel model, IEnumerable<(double[] x, int y)> data, int every = 100)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _data = data ?? throw new ArgumentNullException(nameof(data));
        _every = System.Math.Max(1, every);
    }

    public void OnEpochEnd(int epoch)
    {
        // epoch в Trainer идёт с 0 :contentReference[oaicite:5]{index=5}
        int e = epoch + 1;
        if (e % _every != 0) return;

        double acc = Accuracy(_model, _data);
        Console.WriteLine($"Epoch {e}: accuracy={acc:P2}");
    }

    private static double Accuracy(IModel model, IEnumerable<(double[] x, int y)> data)
    {
        int total = 0;
        int ok = 0;

        foreach (var (x, y) in data)
        {
            var p = model.Forward(x, training: false);
            int pred = ArgMax(p);
            if (pred == y) ok++;
            total++;
        }

        return total == 0 ? 0 : (double)ok / total;
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
