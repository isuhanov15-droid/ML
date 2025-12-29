using System;
using System.Collections.Generic;
using ML.Core.Abstractions;

namespace ML.Core.Layers;

public sealed class SoftmaxLayer : ILayer
{
    public int Size { get; }

    public double[] LastOutput { get; private set; } = Array.Empty<double>();

    public SoftmaxLayer(int size)
    {
        if (size <= 0) throw new ArgumentException("size must be > 0", nameof(size));
        Size = size;
    }

    public double[] Forward(double[] input, bool training = true)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Length != Size) throw new ArgumentException("Softmax size mismatch", nameof(input));

        // max-trick
        double max = input[0];
        for (int i = 1; i < input.Length; i++)
            if (input[i] > max) max = input[i];

        double sum = 0.0;
        var exp = new double[Size];

        for (int i = 0; i < Size; i++)
        {
            exp[i] = System.Math.Exp(input[i] - max);
            sum += exp[i];
        }

        var p = new double[Size];
        for (int i = 0; i < Size; i++)
            p[i] = exp[i] / sum;

        LastOutput = p;
        return p;
    }

    public double[] Backward(double[] gradOutput)
    {
        // Для Softmax + CrossEntropy градиент приходит как (p - y), можно пропускать.
        return gradOutput;
    }

    public IEnumerable<IParameter> Parameters()
    {
        yield break;
    }
}
