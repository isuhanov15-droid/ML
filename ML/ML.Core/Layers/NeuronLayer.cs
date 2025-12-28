using System;
using ML.Core.Abstractions;

namespace ML.Core.Layers;

public class NeuronLayer : ITrainableLayer
{
    private readonly Neuron[] _neurons;

    public NeuronType LayerType { get; }
    public ActivationType ActivationType { get; }  // <-- ДОБАВИЛИ

    public int InputSize { get; }
    public int OutputSize => _neurons.Length;

    public Neuron[] Neurons => _neurons;           // <-- ДОБАВИЛИ

    public NeuronLayer(
        int inputSize,
        int neuronCount,
        NeuronType layerType,
        ActivationType activationType)
    {
        if (inputSize <= 0) throw new ArgumentException("inputSize must be > 0");
        if (neuronCount <= 0) throw new ArgumentException("neuronCount must be > 0");

        InputSize = inputSize;
        LayerType = layerType;
        ActivationType = ResolveActivation(layerType, activationType); // <-- фиксируем

        _neurons = new Neuron[neuronCount];

        for (int i = 0; i < neuronCount; i++)
        {
            _neurons[i] = new Neuron(
                inputCount: inputSize,
                neuronType: layerType,
                activationType: ActivationType
            );
        }
    }

    public double[] Forward(double[] input)
    {
        if (input.Length != InputSize)
            throw new ArgumentException("Input size mismatch");

        var output = new double[_neurons.Length];
        for (int i = 0; i < _neurons.Length; i++)
            output[i] = _neurons[i].Forward(input);

        return output;
    }

    public double[] Backward(double[] outputGradient, IOptimizer opt)
    {
        var inputGradient = new double[InputSize];

        for (int i = 0; i < _neurons.Length; i++)
        {
            var neuron = _neurons[i];
            double delta = outputGradient[i] * neuron.ActivationDerivative();

            neuron.BiasGradient = delta;

            for (int j = 0; j < neuron.Weights.Length; j++)
            {
                neuron.WeightGradients[j] = delta * neuron.LastInput[j];
                inputGradient[j] += neuron.Weights[j] * delta;
            }

            opt.Step(neuron);
        }

        return inputGradient;
    }

    private static ActivationType ResolveActivation(NeuronType layerType, ActivationType requested)
        => layerType switch
        {
            NeuronType.Input  => ActivationType.Linear,
            _ => requested
        };
}
