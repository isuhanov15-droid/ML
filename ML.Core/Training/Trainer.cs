using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core.Abstractions;
using ML.Core.Training.Collbacks;

namespace ML.Core.Training;

public sealed class Trainer
{
    private readonly IModel _model;
    private readonly IOptimizer _optimizer;
    private readonly ILoss _loss;

    private static readonly Random Rnd = new();

    public Trainer(IModel model, IOptimizer optimizer, ILoss loss)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        _loss = loss ?? throw new ArgumentNullException(nameof(loss));
    }

    /// <summary>
    /// Старый режим (SGD): шаг на каждом примере.
    /// Оставляем как есть, чтобы ничего не сломать.
    /// </summary>
    public void Train(
        IEnumerable<(double[] x, int y)> dataset,
        int epochs,
        IEnumerable<ICallback>? callbacks = null)
    {
        if (dataset == null) throw new ArgumentNullException(nameof(dataset));
        if (epochs <= 0) throw new ArgumentException("epochs must be > 0", nameof(epochs));

        var data = dataset as (double[] x, int y)[] ?? dataset.ToArray();
        var cbs = callbacks?.ToArray() ?? Array.Empty<ICallback>();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var (x, y) in data)
            {
                var output = _model.Forward(x, training: true);

                _loss.Forward(output, y);
                var grad = _loss.Backward();

                _model.Backward(grad);

                var ps = _model.Parameters();
                _optimizer.Step(ps);
                _optimizer.ZeroGrad(ps);
                
            }

            foreach (var cb in cbs)
                cb.OnEpochEnd(epoch);
        }
    }

    /// <summary>
    /// Новый режим: Mini-Batch Gradient Descent (с Adam тоже отлично).
    /// ВАЖНО: градиенты усредняются по размеру батча.
    /// </summary>
    public void TrainMiniBatch(
        IEnumerable<(double[] x, int y)> dataset,
        int epochs,
        int batchSize = 64,
        bool shuffle = true,
        IEnumerable<ICallback>? callbacks = null)
    {
        if (dataset == null) throw new ArgumentNullException(nameof(dataset));
        if (epochs <= 0) throw new ArgumentException("epochs must be > 0", nameof(epochs));
        if (batchSize <= 0) throw new ArgumentException("batchSize must be > 0", nameof(batchSize));

        var data = dataset as (double[] x, int y)[] ?? dataset.ToArray();
        var cbs = callbacks?.ToArray() ?? Array.Empty<ICallback>();

        // индексы — один раз, потом тасуем in-place
        var indices = Enumerable.Range(0, data.Length).ToArray();

        // параметры модели (у слоёв у тебя кеш, так что это дёшево)
        var ps = _model.Parameters();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            if (shuffle)
                Shuffle(indices);

            for (int start = 0; start < indices.Length; start += batchSize)
            {
                int end = System.Math.Min(start + batchSize, indices.Length);
                int actualBatch = end - start;

                // важно: обнуляем грады ПЕРЕД батчем
                _optimizer.ZeroGrad(ps);

                // 1) копим градиенты по всем примерам батча
                for (int t = start; t < end; t++)
                {
                    var (x, y) = data[indices[t]];

                    var output = _model.Forward(x, training: true);

                    _loss.Forward(output, y);
                    var grad = _loss.Backward();

                    _model.Backward(grad);
                }

                // 2) усредняем градиенты по батчу
                ScaleGradients(ps, 1.0 / actualBatch);

                // 3) шаг оптимизатора один раз на батч
                _optimizer.Step(ps);
                
            }

            foreach (var cb in cbs)
                cb.OnEpochEnd(epoch);
        }
    }

    private static void ScaleGradients(IEnumerable<IParameter> parameters, double scale)
    {
        foreach (var p in parameters)
        {
            var g = p.Grad;
            for (int i = 0; i < g.Length; i++)
                g[i] *= scale;
        }
    }

    private static void Shuffle(int[] a)
    {
        for (int i = a.Length - 1; i > 0; i--)
        {
            int j = Rnd.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }
}
