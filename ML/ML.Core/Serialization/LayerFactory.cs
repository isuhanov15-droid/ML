using System;
using ML.Core.Layers;
using ML.Core.Abstractions;

namespace ML.Core.Serialization;

public static class LayerFactory
{
    public static ILayer Create(LayerDto dto, int inputSize)
    {
        return dto.Type switch
        {
            "Dense"   => CreateDense(dto, inputSize),
            "Softmax" => new SoftmaxLayer(inputSize),
            _ => throw new InvalidOperationException($"Unknown layer type: {dto.Type}")
        };
    }

    private static ILayer CreateDense(LayerDto dto, int inputSize)
    {
        if (dto.Neurons == null || dto.Neurons.Count == 0)
            throw new InvalidOperationException("Dense layer DTO has no neurons.");

        var act = ParseActivation(dto.Activation);
        var nt  = ParseNeuronType(dto.NeuronType);

        var layer = new NeuronLayer(inputSize, dto.Neurons.Count, nt, act);

        for (int i = 0; i < dto.Neurons.Count; i++)
        {
            layer.Neurons[i].Weights = dto.Neurons[i].Weights;
            layer.Neurons[i].Bias    = dto.Neurons[i].Bias;
        }

        return layer;
    }

    private static ActivationType ParseActivation(string? act)
        => string.IsNullOrWhiteSpace(act)
            ? ActivationType.Linear
            : Enum.Parse<ActivationType>(act, ignoreCase: true);

    private static NeuronType ParseNeuronType(string? nt)
        => string.IsNullOrWhiteSpace(nt)
            ? NeuronType.Hidden
            : Enum.Parse<NeuronType>(nt, ignoreCase: true);
}
