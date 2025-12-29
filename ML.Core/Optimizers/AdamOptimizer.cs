using System;
using System.Collections.Generic;
using ML.Core.Abstractions;

namespace ML.Core.Optimizers
{
    /// <summary>
    /// Adam optimizer, работает с IParameter.
    /// НЕ знает про Neuron. Всё состояние хранит на параметр.
    /// </summary>
    public sealed class AdamOptimizer : IOptimizer
    {
        private readonly double _lr;
        private readonly double _beta1;
        private readonly double _beta2;
        private readonly double _eps;

        private int _t = 0;

        private sealed class State
        {
            public double[] M = Array.Empty<double>();
            public double[] V = Array.Empty<double>();
        }

        private readonly Dictionary<IParameter, State> _states = new();

        public AdamOptimizer(
            double learningRate = 0.001,
            double beta1 = 0.9,
            double beta2 = 0.999,
            double epsilon = 1e-8)
        {
            _lr = learningRate;
            _beta1 = beta1;
            _beta2 = beta2;
            _eps = epsilon;
        }

        public void Step(IEnumerable<IParameter> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));

            _t++;

            foreach (var p in parameters)
            {
                if (p.Value.Length != p.Grad.Length)
                    throw new InvalidOperationException("Parameter Value/Grad length mismatch.");

                if (!_states.TryGetValue(p, out var st))
                {
                    st = new State
                    {
                        M = new double[p.Value.Length],
                        V = new double[p.Value.Length]
                    };
                    _states[p] = st;
                }
                else
                {
                    // если вдруг параметр изменил размер (не должен), переинициализируем
                    if (st.M.Length != p.Value.Length)
                    {
                        st.M = new double[p.Value.Length];
                        st.V = new double[p.Value.Length];
                    }
                }

                // bias correction
                double b1t = 1.0 - System.Math.Pow(_beta1, _t);
                double b2t = 1.0 - System.Math.Pow(_beta2, _t);

                for (int i = 0; i < p.Value.Length; i++)
                {
                    double g = p.Grad[i];

                    st.M[i] = _beta1 * st.M[i] + (1.0 - _beta1) * g;
                    st.V[i] = _beta2 * st.V[i] + (1.0 - _beta2) * g * g;

                    double mHat = st.M[i] / b1t;
                    double vHat = st.V[i] / b2t;

                    p.Value[i] -= _lr * mHat / (System.Math.Sqrt(vHat) + _eps);
                }
            }
        }

        public void ZeroGrad(IEnumerable<IParameter> parameters)
        {
            if (parameters == null) throw new ArgumentNullException(nameof(parameters));
            foreach (var p in parameters)
                p.ZeroGrad();
        }
    }
}
