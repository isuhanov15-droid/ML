using System;
using System.Collections.Generic;
using ML.Core.Abstractions;

namespace ML.Core.Layers
{
    /// <summary>
    /// Softmax как слой.
    /// Параметров нет.
    ///
    /// ВАЖНО:
    /// Если ты используешь CrossEntropyLoss, который сам делает (softmax + градиент p - y),
    /// то отдельный Backward для Softmax обычно НЕ нужен — мы просто пропускаем gradOutput.
    /// </summary>
    public sealed class SoftmaxLayer : ILayer
    {
        public int InputSize { get; }
        public int OutputSize => InputSize;

        public double[] LastInput { get; private set; } = Array.Empty<double>();
        public double[] LastOutput { get; private set; } = Array.Empty<double>();

        public SoftmaxLayer(int size)
        {
            if (size <= 0) throw new ArgumentException("size must be > 0", nameof(size));
            InputSize = size;
        }

        public double[] Forward(double[] input, bool training = true)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (input.Length != InputSize)
                throw new ArgumentException("Input size mismatch for SoftmaxLayer", nameof(input));

            LastInput = (double[])input.Clone();

            // max-trick for numeric stability
            double max = input[0];
            for (int i = 1; i < input.Length; i++)
                if (input[i] > max) max = input[i];

            var exp = new double[input.Length];
            double sum = 0.0;

            for (int i = 0; i < input.Length; i++)
            {
                exp[i] = Math.Exp(input[i] - max);
                sum += exp[i];
            }

            var output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
                output[i] = exp[i] / sum;

            LastOutput = output;
            return output;
        }

        public double[] Backward(double[] gradOutput)
        {
            if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
            if (gradOutput.Length != OutputSize)
                throw new ArgumentException("Gradient size mismatch for SoftmaxLayer", nameof(gradOutput));

            // Каноничный путь для Softmax + CrossEntropy:
            // градиент уже "правильный" (p - y) приходит из Loss,
            // поэтому SoftmaxLayer просто пропускает его дальше.
            return gradOutput;
        }

        public IEnumerable<IParameter> Parameters()
        {
            yield break; // параметров нет
        }
    }
}
