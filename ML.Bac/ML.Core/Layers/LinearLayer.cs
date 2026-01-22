using System;
using System.Collections.Generic;
using ML.Core.Abstractions;
using ML.Core.Math;

namespace ML.Core.Layers;

/// <summary>
/// Linear: y = W*x + b
/// W: [out, in] row-major
/// b: [out]
/// </summary>
public sealed class LinearLayer : ILayer
{
    private readonly List<IParameter> _parameters = new();

    public int InputSize { get; }
    public int OutputSize { get; }

    public double[] Weights { get; }         // len = out * in
    public double[] WeightGrads { get; }     // same
    public double[] Bias { get; }            // len = out
    public double[] BiasGrads { get; }       // same

    public double[] LastInput { get; private set; } = Array.Empty<double>();

    public LinearLayer(int inputSize, int outputSize, int seed = 123)
    {
        if (inputSize <= 0) throw new ArgumentException("inputSize must be > 0", nameof(inputSize));
        if (outputSize <= 0) throw new ArgumentException("outputSize must be > 0", nameof(outputSize));

        InputSize = inputSize;
        OutputSize = outputSize;

        Weights = new double[outputSize * inputSize];
        WeightGrads = new double[Weights.Length];
        Bias = new double[outputSize];
        BiasGrads = new double[outputSize];

        InitWeights(seed);

        // ✅ кеш параметров (важно для Adam!)
        _parameters.Add(new Parameter(Weights, WeightGrads));
        _parameters.Add(new Parameter(Bias, BiasGrads));
    }

    private void InitWeights(int seed)
    {
        // He/Glorot-подобная инициализация
        var rnd = new Random(seed);
        double scale = System.Math.Sqrt(2.0 / InputSize);

        for (int i = 0; i < Weights.Length; i++)
            Weights[i] = (rnd.NextDouble() * 2.0 - 1.0) * scale;

        Array.Clear(Bias, 0, Bias.Length);
    }

    public double[] Forward(double[] input, bool training = true)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Length != InputSize)
            throw new ArgumentException("Input size mismatch", nameof(input));

        // ✅ хранить копию входа для корректного backprop (как вы уже фиксили в Neuron)
        LastInput = (double[])input.Clone();

        var y = new double[OutputSize];
        Matrix.MatVec(Weights, OutputSize, InputSize, input, y);

        for (int i = 0; i < y.Length; i++)
            y[i] += Bias[i];

        return y;
    }

    public double[] Backward(double[] gradOutput)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradOutput.Length != OutputSize)
            throw new ArgumentException("gradOutput size mismatch", nameof(gradOutput));

        // dW += dy ⊗ x
        Matrix.OuterAdd(WeightGrads, OutputSize, InputSize, gradOutput, LastInput);

        // db += dy
        for (int i = 0; i < OutputSize; i++)
            BiasGrads[i] += gradOutput[i];

        // dx = W^T * dy
        var dx = new double[InputSize];
        Matrix.MatTVec(Weights, OutputSize, InputSize, gradOutput, dx);

        return dx;
    }

    public IEnumerable<IParameter> Parameters() => _parameters;
}
