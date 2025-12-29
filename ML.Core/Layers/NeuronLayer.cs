using System;
using System.Collections.Generic;
using ML.Core.Abstractions;

namespace ML.Core.Layers
{
    /// <summary>
    /// Полносвязный слой на базе массива Neuron.
    /// ВАЖНО: параметры кешируются, чтобы не плодить миллионы объектов во время обучения.
    /// </summary>
    public sealed class NeuronLayer : ILayer
    {
        private readonly Neuron[] _neurons;
        private readonly List<IParameter> _parameters;   // ✅ кеш

        public NeuronType LayerType { get; }
        public ActivationType ActivationType { get; }

        public int InputSize { get; }
        public int OutputSize => _neurons.Length;

        public Neuron[] Neurons => _neurons;

        public NeuronLayer(
            int inputSize,
            int neuronCount,
            NeuronType layerType,
            ActivationType activationType)
        {
            if (inputSize <= 0) throw new ArgumentException("inputSize must be > 0", nameof(inputSize));
            if (neuronCount <= 0) throw new ArgumentException("neuronCount must be > 0", nameof(neuronCount));

            InputSize = inputSize;
            LayerType = layerType;
            ActivationType = ResolveActivation(layerType, activationType);

            _neurons = new Neuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                _neurons[i] = new Neuron(
                    inputCount: inputSize,
                    neuronType: layerType,
                    activationType: ActivationType
                );
            }

            // ✅ создаём параметры ОДИН РАЗ
            _parameters = new List<IParameter>(neuronCount * 2);
            for (int i = 0; i < _neurons.Length; i++)
            {
                var n = _neurons[i];

                // веса (ссылки на массивы нейрона)
                _parameters.Add(new Parameter(n.Weights, n.WeightGradients));

                // bias (скаляр через прокси)
                _parameters.Add(Parameter.ForScalar(
                    get: () => n.Bias,
                    set: v => n.Bias = v,
                    gradGet: () => n.BiasGradient,
                    gradSet: g => n.BiasGradient = g
                ));
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

                neuron.BiasGradient = delta;

                for (int j = 0; j < neuron.Weights.Length; j++)
                {
                    neuron.WeightGradients[j] = delta * neuron.LastInput[j];
                    inputGradient[j] += neuron.Weights[j] * delta;
                }
            }

            return inputGradient;
        }

        public IEnumerable<IParameter> Parameters()
        {
            // ✅ возвращаем кеш, без аллокаций
            return _parameters;
        }

        private static ActivationType ResolveActivation(NeuronType layerType, ActivationType requested)
            => layerType switch
            {
                NeuronType.Input => ActivationType.Linear,
                _ => requested
            };
    }
}
