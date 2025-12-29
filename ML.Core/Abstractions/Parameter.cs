using System;
using ML.Core.Abstractions;

namespace ML.Core.Abstractions
{
    /// <summary>
    /// Универсальный параметр.
    /// - Для массивов: хранит ссылки Value/Grad.
    /// - Для скаляра: оборачивает get/set (чтобы обновления реально попадали в источник).
    /// </summary>
    public sealed class Parameter : IParameter
    {
        private readonly Func<double>? _getScalar;
        private readonly Action<double>? _setScalar;
        private readonly Func<double>? _getScalarGrad;
        private readonly Action<double>? _setScalarGrad;

        public double[] Value { get; }
        public double[] Grad { get; }

        public Parameter(double[] value, double[] grad)
        {
            Value = value ?? throw new ArgumentNullException(nameof(value));
            Grad = grad ?? throw new ArgumentNullException(nameof(grad));
            if (Value.Length != Grad.Length)
                throw new ArgumentException("Value and Grad must have same length.");
        }

        private Parameter(
            Func<double> get,
            Action<double> set,
            Func<double> gradGet,
            Action<double> gradSet)
        {
            _getScalar = get;
            _setScalar = set;
            _getScalarGrad = gradGet;
            _setScalarGrad = gradSet;

            // ВАЖНО: Value/Grad — “прокси”-массивы длины 1.
            Value = new double[1];
            Grad = new double[1];

            SyncFromSource();
        }

        public static Parameter ForScalar(
            Func<double> get,
            Action<double> set,
            Func<double> gradGet,
            Action<double> gradSet)
        {
            if (get == null) throw new ArgumentNullException(nameof(get));
            if (set == null) throw new ArgumentNullException(nameof(set));
            if (gradGet == null) throw new ArgumentNullException(nameof(gradGet));
            if (gradSet == null) throw new ArgumentNullException(nameof(gradSet));

            return new Parameter(get, set, gradGet, gradSet);
        }

        public void ZeroGrad()
        {
            if (_setScalarGrad != null)
            {
                _setScalarGrad(0.0);
                Grad[0] = 0.0;
                return;
            }

            Array.Clear(Grad, 0, Grad.Length);
        }

        /// <summary>
        /// Для скалярного параметра: перед оптимизацией подтянуть актуальное значение.
        /// </summary>
        private void SyncFromSource()
        {
            if (_getScalar != null)
                Value[0] = _getScalar();
            if (_getScalarGrad != null)
                Grad[0] = _getScalarGrad();
        }

        /// <summary>
        /// Для скалярного параметра: после шага оптимизатора протолкнуть значение назад в источник.
        /// </summary>
        private void SyncToSource()
        {
            if (_setScalar != null)
                _setScalar(Value[0]);
            if (_setScalarGrad != null)
                _setScalarGrad(Grad[0]);
        }

        /// <summary>
        /// Хак-но-честный: оптимизатор меняет Value[0] напрямую.
        /// Мы должны протолкнуть это в источник.
        /// Поэтому вызови это ПОСЛЕ optimizer.Step(parameters) для скаляров,
        /// либо сделай это в Trainer (см. ниже).
        /// </summary>
        public void Sync()
        {
            if (_getScalar != null || _setScalar != null)
                SyncToSource();
        }
    }
}
