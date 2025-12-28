
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using ML.Core.Layers;
using ML.Core.Abstractions;
using ML.Core.Serialization;


namespace ML.Core;

public class Network
{
    private readonly List<ILayer> _layers = new();

    private int _inputSize;
    public int LayerCount => _layers.Count;

    public Network() { }

    public Network(int inputSize)
    {
        _inputSize = inputSize;
    }


    public void AddLayer(ILayer layer)
    {
        if (layer == null)
            throw new ArgumentNullException(nameof(layer));

        if (_layers.Count == 0)
        {
            if (layer.InputSize != _inputSize)
                throw new ArgumentException($"First layer input must be {_inputSize}");
        }
        else
        {
            int prevOut = _layers[^1].OutputSize;
            if (layer.InputSize != prevOut)
                throw new ArgumentException($"Layer size mismatch: previous output {prevOut}, current input {layer.InputSize}");
        }

        _layers.Add(layer);
    }

    public void AddInputLayer()
    {
        if (_layers.Count > 0)
            throw new InvalidOperationException("Input layer must be first");

        AddLayer(new NeuronLayer(_inputSize, _inputSize, NeuronType.Input, ActivationType.Linear));
    }

    public void AddHiddenLayer(int neuronCount, ActivationType activation)
    {
        int inputSize = _layers.Count == 0 ? _inputSize : _layers[^1].OutputSize;
        AddLayer(new NeuronLayer(inputSize, neuronCount, NeuronType.Hidden, activation));
    }

    public void AddOutputLayer(int neuronCount, ActivationType activation)
    {
        int inputSize = _layers.Count == 0 ? _inputSize : _layers[^1].OutputSize;
        AddLayer(new NeuronLayer(inputSize, neuronCount, NeuronType.Output, activation));
    }

    public void AddSoftmax()
    {
        if (_layers.Count == 0)
            throw new InvalidOperationException("Add output layer before Softmax");

        int size = _layers[^1].OutputSize;
        AddLayer(new SoftmaxLayer(size));
    }

    public bool EndsWithSoftmax => _layers.Count > 0 && _layers[^1] is SoftmaxLayer;

    public void EnsureSoftmax()
    {
        if (_layers.Count == 0)
            throw new InvalidOperationException("Cannot add Softmax: network has no layers");

        if (!EndsWithSoftmax)
            AddSoftmax();
    }

    public double[] Forward(double[] input)
    {
        if (_layers.Count == 0)
            throw new InvalidOperationException("Network has no layers");

        double[] x = input;
        foreach (var layer in _layers)
            x = layer.Forward(x);

        return x;
    }
    public void Backward(double[] predicted, int targetClass, IOptimizer opt)
    {
        double[] gradient = new double[predicted.Length];
        for (int i = 0; i < predicted.Length; i++)
            gradient[i] = predicted[i];

        gradient[targetClass] -= 1.0;

        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            if (_layers[i] is ITrainableLayer trainable)
                gradient = trainable.Backward(gradient, opt);
        }

        opt.NextStep();
    }

    public void Save(string path)
    {
        var dto = new NetworkDto
        {
            InputSize = _inputSize
        };

        foreach (var layer in _layers)
        {
            if (layer is NeuronLayer dense)
            {
                var ld = new LayerDto
                {
                    Type = "Dense",
                    NeuronType = dense.LayerType.ToString(),
                    Activation = dense.ActivationType.ToString(),
                    Neurons = dense.Neurons.Select(n => new NeuronDto
                    {
                        Weights = n.Weights.ToArray(),
                        Bias = n.Bias
                    }).ToList()
                };

                dto.Layers.Add(ld);
            }
            else if (layer is SoftmaxLayer)
            {
                dto.Layers.Add(new LayerDto
                {
                    Type = "Softmax"
                });
            }
            else
            {
                throw new InvalidOperationException($"Unsupported layer type for serialization: {layer.GetType().Name}");
            }
        }

        var json = JsonSerializer.Serialize(dto, JsonOptions.Default);
        File.WriteAllText(path, json);
    }
    public void Load(string path)
    {
        var json = File.ReadAllText(path);
        var dto = JsonSerializer.Deserialize<NetworkDto>(json)
                  ?? throw new InvalidOperationException("Load: invalid model file.");

        _layers.Clear();
        _inputSize = dto.InputSize;

        int currentInput = _inputSize;

        foreach (var layerDto in dto.Layers)
        {
            var layer = LayerFactory.Create(layerDto, currentInput);
            _layers.Add(layer);
            currentInput = layer.OutputSize;
        }
    }



}
