using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core.Abstractions;
using ML.Core.Layers;
using ML.Core.Serialization;

namespace ML.Core;

public sealed class Network : IModel
{
    private readonly List<ILayer> _layers = new();

    public IReadOnlyList<ILayer> Layers => _layers;

    public Network() { }

    public void Add(ILayer layer)
    {
        _layers.Add(layer ?? throw new ArgumentNullException(nameof(layer)));
    }

    public double[] Forward(double[] input, bool training = true)
    {
        if (_layers.Count == 0) throw new InvalidOperationException("Network has no layers.");

        double[] x = input;
        foreach (var layer in _layers)
            x = layer.Forward(x, training);

        return x;
    }

    public double[] Backward(double[] gradOutput)
    {
        if (_layers.Count == 0) throw new InvalidOperationException("Network has no layers.");

        double[] grad = gradOutput;
        for (int i = _layers.Count - 1; i >= 0; i--)
            grad = _layers[i].Backward(grad);

        return grad;
    }

    public IEnumerable<IParameter> Parameters()
    {
        foreach (var layer in _layers)
            foreach (var p in layer.Parameters())
                yield return p;
    }

    // ✅ Load/Save в ML/Models
    public void Save(string modelName) => ModelStore.Save(modelName, this);
    public static Network Load(string modelName) => ModelStore.Load(modelName);
}
