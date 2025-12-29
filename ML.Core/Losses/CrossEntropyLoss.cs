using System;
using ML.Core.Abstractions;

namespace ML.Core.Losses
{
    /// <summary>
    /// Cross-Entropy Loss для многоклассовой классификации с Softmax.
    ///
    /// ВАЖНО:
    /// Входные данные - вероятности классов (после softmax).
    /// Градиент: dL/dz = p - y (где p - вероятности, y - one-hot целевой вектор).
    /// </summary>;

    public sealed class CrossEntropyLoss : ILoss
    {
        private double[] _lastProbs = Array.Empty<double>();
        private int _lastTarget;

        public double Forward(double[] logits, int target)
        {
            if (logits == null) throw new ArgumentNullException(nameof(logits));
            if (logits.Length == 0) throw new ArgumentException("Empty logits/probs.", nameof(logits));
            if (target < 0 || target >= logits.Length) throw new ArgumentOutOfRangeException(nameof(target));

            // ВАЖНО: здесь logits трактуем как probs (после softmax)
            _lastProbs = (double[])logits.Clone();
            _lastTarget = target;

            const double eps = 1e-12;
            double p = _lastProbs[target];
            if (p < eps) p = eps;

            return -Math.Log(p);
        }

        public double[] Backward()
        {
            // dL/dz = p - y (если вход уже softmax probs)
            var grad = (double[])_lastProbs.Clone();
            grad[_lastTarget] -= 1.0;
            return grad;
        }
    }
}
