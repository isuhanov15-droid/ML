using System;
using static System.Math;
using ML.Core.Abstractions;

namespace ML.Core.Layers;

public sealed class ActivationLayer : ILayer
{
    public int Size { get; }
    public ActivationType Type { get; }

    private double[] _lastZ = Array.Empty<double>();
    private double[] _lastA = Array.Empty<double>();

    public ActivationLayer(int size, ActivationType type)
    {
        if (size <= 0) throw new ArgumentException("size must be > 0", nameof(size));
        Size = size;
        Type = type;
    }

    public double[] Forward(double[] input, bool training = true)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Length != Size) throw new ArgumentException("Activation size mismatch", nameof(input));

        _lastZ = (double[])input.Clone();
        var a = new double[Size];

        for (int i = 0; i < Size; i++)
            a[i] = Activate(_lastZ[i]);

        _lastA = a;
        return a;
    }

    public double[] Backward(double[] gradOutput)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradOutput.Length != Size) throw new ArgumentException("Activation grad size mismatch", nameof(gradOutput));

        var grad = new double[Size];
        for (int i = 0; i < Size; i++)
            grad[i] = gradOutput[i] * Derivative(i);

        return grad;
    }

    public IEnumerable<IParameter> Parameters()
    {
        yield break; // нет параметров
    }

    private double Activate(double x) => Type switch
    {
        ActivationType.Linear => x,
        ActivationType.Sigmoid => 1.0 / (1.0 + System.Math.Exp(-x)),
        ActivationType.Tanh => System.Math.Tanh(x),
        ActivationType.ReLu => x > 0 ? x : 0,
        ActivationType.LeakyReLu => x > 0 ? x : 0.01 * x,
        ActivationType.ELU => x >= 0 ? x : (System.Math.Exp(x) - 1.0),
        ActivationType.Softplus => System.Math.Log(1.0 + System.Math.Exp(x)),
        _ => x
    };

    private double Derivative(int i) => Type switch
    {
        ActivationType.Linear => 1.0,
        ActivationType.Sigmoid => _lastA[i] * (1.0 - _lastA[i]),
        ActivationType.Tanh => 1.0 - _lastA[i] * _lastA[i],
        ActivationType.ReLu => _lastZ[i] > 0 ? 1.0 : 0.0,
        ActivationType.LeakyReLu => _lastZ[i] > 0 ? 1.0 : 0.01,
        ActivationType.ELU => _lastZ[i] >= 0 ? 1.0 : System.Math.Exp(_lastZ[i]),
        ActivationType.Softplus => 1.0 / (1.0 + System.Math.Exp(-_lastZ[i])),
        _ => 1.0
    };
}
