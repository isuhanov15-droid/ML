
using System.Text.Json;
using ML.Core.Abstractions;
using ML.Core.Layers;
using ML.Core.Serialization;

namespace ML.Core
{
    /// <summary>
    /// Network = модель (IModel).
    /// Обучение: Trainer + ILoss + IOptimizer.
    /// </summary>
    public sealed class Network : IModel
    {
        private readonly List<ILayer> _layers = new();

        private int _inputSize;
        public int LayerCount => _layers.Count;

        public Network() { }

        public Network(int inputSize)
        {
            if (inputSize <= 0) throw new ArgumentException("inputSize must be > 0", nameof(inputSize));
            _inputSize = inputSize;
        }

        // -----------------------------
        // Helpers: размеры слоёв
        // (Потому что ILayer не содержит InputSize/OutputSize)
        // -----------------------------
        private static int GetInputSize(ILayer layer) =>
            layer switch
            {
                NeuronLayer l => l.InputSize,
                SoftmaxLayer l => l.InputSize,
                _ => throw new InvalidOperationException($"Layer {layer.GetType().Name} does not expose InputSize")
            };

        private static int GetOutputSize(ILayer layer) =>
            layer switch
            {
                NeuronLayer l => l.OutputSize,
                SoftmaxLayer l => l.OutputSize,
                _ => throw new InvalidOperationException($"Layer {layer.GetType().Name} does not expose OutputSize")
            };

        public void AddLayer(ILayer layer)
        {
            if (layer == null) throw new ArgumentNullException(nameof(layer));

            int layerIn = GetInputSize(layer);

            if (_layers.Count == 0)
            {
                if (layerIn != _inputSize)
                    throw new ArgumentException($"First layer input must be {_inputSize}");
            }
            else
            {
                int prevOut = GetOutputSize(_layers[^1]);
                if (layerIn != prevOut)
                    throw new ArgumentException($"Layer size mismatch: previous output {prevOut}, current input {layerIn}");
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
            int inputSize = _layers.Count == 0 ? _inputSize : GetOutputSize(_layers[^1]);
            AddLayer(new NeuronLayer(inputSize, neuronCount, NeuronType.Hidden, activation));
        }

        public void AddOutputLayer(int neuronCount, ActivationType activation)
        {
            int inputSize = _layers.Count == 0 ? _inputSize : GetOutputSize(_layers[^1]);
            AddLayer(new NeuronLayer(inputSize, neuronCount, NeuronType.Output, activation));
        }

        public void AddSoftmax()
        {
            if (_layers.Count == 0)
                throw new InvalidOperationException("Add output layer before Softmax");

            int size = GetOutputSize(_layers[^1]);
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

        // -----------------------------
        // IModel
        // -----------------------------
        public double[] Forward(double[] input, bool training = true)
        {
            if (_layers.Count == 0)
                throw new InvalidOperationException("Network has no layers");

            double[] x = input;
            foreach (var layer in _layers)
                x = layer.Forward(x, training);

            return x;
        }

        public double[] Backward(double[] gradOutput)
        {
            if (_layers.Count == 0)
                throw new InvalidOperationException("Network has no layers");

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

        // -----------------------------
        // Serialization
        // -----------------------------
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
                        Activation = dense.ActivationType.ToString(), // оставил как у тебя в dto-логике
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
                    dto.Layers.Add(new LayerDto { Type = "Softmax" });
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

                // LayerFactory должен создавать NeuronLayer/SoftmaxLayer с корректными размерами
                currentInput = GetOutputSize(layer);
            }
        }
    }
}
