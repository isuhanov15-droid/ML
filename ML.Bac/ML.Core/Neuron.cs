using System;

namespace ML.Core;

public sealed class Neuron
{
    public NeuronType Type { get; init; }
    public ActivationType ActType { get; init; }

    public double[] Weights { get; }
    public double[] WeightGradients { get; }

    // ✅ Bias теперь массив длины 1 (чтобы оптимизатор мог обновлять по ссылке)
    public double[] Bias { get; } = new double[1];
    public double[] BiasGradient { get; } = new double[1];

    public double Z { get; private set; }
    public double A { get; private set; }

    // ✅ важно: хранить копию входа (у тебя это уже спасло XOR)
    public double[] LastInput { get; private set; } = Array.Empty<double>();

    private static readonly Random Rnd = new Random();

    public Neuron(int inputCount, NeuronType neuronType, ActivationType activationType)
    {
        if (inputCount <= 0) throw new ArgumentException("inputCount must be > 0", nameof(inputCount));

        Type = neuronType;
        ActType = activationType;

        Weights = new double[inputCount];
        WeightGradients = new double[inputCount];

        // Init weights
        // He/Glorot можно улучшать позже, пока оставим простой нормальный масштаб
        double scale = 1.0 / System.Math.Sqrt(inputCount);
        for (int i = 0; i < inputCount; i++)
            Weights[i] = (Rnd.NextDouble() * 2 - 1) * scale;

        Bias[0] = 0.0;
        BiasGradient[0] = 0.0;
    }

    public double Forward(double[] inputs)
    {
        if (inputs == null) throw new ArgumentNullException(nameof(inputs));
        if (inputs.Length != Weights.Length)
            throw new ArgumentException("Input size does not match weights size", nameof(inputs));

        // ✅ критично для корректного backprop
        LastInput = (double[])inputs.Clone();

        double sum = Bias[0];
        for (int i = 0; i < inputs.Length; i++)
            sum += inputs[i] * Weights[i];

        Z = sum;
        A = Activate(Z);
        return A;
    }

    public double Activate(double x)
    {
        return ActType switch
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
    }

    public double ActivationDerivative()
    {
        return ActType switch
        {
            ActivationType.Linear => 1.0,
            ActivationType.Sigmoid => A * (1.0 - A),
            ActivationType.Tanh => 1.0 - A * A,
            ActivationType.ReLu => Z > 0 ? 1.0 : 0.0,
            ActivationType.LeakyReLu => Z > 0 ? 1.0 : 0.01,
            ActivationType.ELU => Z >= 0 ? 1.0 : System.Math.Exp(Z),
            ActivationType.Softplus => 1.0 / (1.0 + System.Math.Exp(-Z)),
            _ => 1.0
        };
    }
}

public enum NeuronType
{
    Input,
    Hidden,
    Output
}


//Колекция функций активации
public enum ActivationType
{
    Linear,
    Sigmoid,
    Tanh,
    ReLu,
    LeakyReLu,
    ELU,
    Gelu,
    Swish,
    Softplus
}
