using System;
using System.Collections.Generic;
using ML.Core.Abstractions;

namespace ML.Core.Layers;

public sealed class NeuronLayer : ILayer
{
    private readonly Neuron[] _neurons;

    // ✅ кеш параметров (как ты уже сделал — оставляем!)
    private readonly List<IParameter> _parameters;

    public NeuronType LayerType { get; }
    public ActivationType ActivationType { get; }

    public int InputSize { get; }
    public int OutputSize => _neurons.Length;

    public Neuron[] Neurons => _neurons;

    public NeuronLayer(int inputSize, int neuronCount, NeuronType layerType, ActivationType activationType)
    {
        if (inputSize <= 0) throw new ArgumentException("inputSize must be > 0", nameof(inputSize));
        if (neuronCount <= 0) throw new ArgumentException("neuronCount must be > 0", nameof(neuronCount));

        InputSize = inputSize;
        LayerType = layerType;
        ActivationType = ResolveActivation(layerType, activationType);

        _neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
            _neurons[i] = new Neuron(inputSize, layerType, ActivationType);

        // ✅ параметры создаём ОДИН раз
        _parameters = new List<IParameter>(neuronCount * 2);
        for (int i = 0; i < _neurons.Length; i++)
        {
            var n = _neurons[i];

            // weights
            _parameters.Add(new Parameter(n.Weights, n.WeightGradients));

            // ✅ bias как массив [1]
            _parameters.Add(new Parameter(n.Bias, n.BiasGradient));
        }
    }

    public double[] Forward(double[] input, bool training = true)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Length != InputSize)
            throw new ArgumentException("Input size mismatch", nameof(input));

        var output = new double[_neurons.Length];
        for (int i = 0; i < _neurons.Length; i++)
            output[i] = _neurons[i].Forward(input);

        return output;
    }

    public double[] Backward(double[] gradOutput)
    {
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (gradOutput.Length != OutputSize)
            throw new ArgumentException("Output gradient size mismatch", nameof(gradOutput));

        var inputGradient = new double[InputSize];

        for (int i = 0; i < _neurons.Length; i++)
        {
            var neuron = _neurons[i];

            double delta = gradOutput[i] * neuron.ActivationDerivative();

            // ✅ bias grad теперь массив
            neuron.BiasGradient[0] = delta;

            for (int j = 0; j < neuron.Weights.Length; j++)
            {
                neuron.WeightGradients[j] = delta * neuron.LastInput[j];
                inputGradient[j] += neuron.Weights[j] * delta;
            }
        }

        return inputGradient;
    }

    public IEnumerable<IParameter> Parameters() => _parameters;

    private static ActivationType ResolveActivation(NeuronType layerType, ActivationType requested)
        => layerType == NeuronType.Input ? ActivationType.Linear : requested;
}
