using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core.Abstractions;
using ML.Core.Training.Callbacks;

namespace ML.Core.Training;

public sealed class Trainer
{
    private readonly IModel _model;
    private readonly IOptimizer _optimizer;
    private readonly ILoss _loss;

    private readonly Random _rnd;

    public Trainer(IModel model, IOptimizer optimizer, ILoss loss)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
        _loss = loss ?? throw new ArgumentNullException(nameof(loss));

        // seed задаём в TrainOptions — здесь пусть будет дефолтный
        _rnd = new Random();
    }

    /// <summary>
    /// Каноническая сигнатура обучения ядра.
    /// Ядро считает только Loss (train + optional val).
    /// Метрики (accuracy) — снаружи (Examples).
    /// </summary>
    public void Train(IEnumerable<(double[] x, int y)> dataset, TrainOptions options, CancellationToken cancellationToken = default)
    {
        if (dataset == null) throw new ArgumentNullException(nameof(dataset));
        if (options == null) throw new ArgumentNullException(nameof(options));

        if (options.Epochs <= 0) throw new ArgumentException("Epochs must be > 0", nameof(options));
        if (options.BatchSize <= 0) throw new ArgumentException("BatchSize must be > 0", nameof(options));
        if (options.GradientAccumulationSteps <= 0) throw new ArgumentException("GradientAccumulationSteps must be > 0", nameof(options));

        // фиксируем данные (один раз)
        var trainData = dataset as (double[] x, int y)[] ?? dataset.ToArray();
        if (trainData.Length == 0) throw new ArgumentException("Dataset is empty", nameof(dataset));

        var valData = options.Validation == null
            ? null
            : (options.Validation as (double[] x, int y)[] ?? options.Validation.ToArray());

        var callbacks = options.Callbacks?.ToArray() ?? Array.Empty<ITrainCallback>();

        // Seed для shuffle
        var rnd = options.Seed.HasValue ? new Random(options.Seed.Value) : _rnd;

        // Параметры модели — берём один раз
        var ps = _model.Parameters().ToArray();

        // Индексы для shuffle
        var indices = Enumerable.Range(0, trainData.Length).ToArray();

        int batchSize = options.BatchSize;

        for (int epoch = 1; epoch <= options.Epochs; epoch++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (options.Shuffle)
                Shuffle(indices, rnd);

            double epochLossSum = 0.0;
            int epochSamples = 0;

            // grad accumulation
            int accumSteps = options.GradientAccumulationSteps;
            int accumCounter = 0;

            // Важно: начинаем эпоху с чистых градиентов
            _optimizer.ZeroGrad(ps);

            for (int start = 0; start < indices.Length; start += batchSize)
            {
                cancellationToken.ThrowIfCancellationRequested();

                int end = System.Math.Min(start + batchSize, indices.Length);
                int actualBatch = end - start;

                if (options.DropLast && actualBatch < batchSize)
                    break;

                // 1) копим градиенты по батчу
                for (int t = start; t < end; t++)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var (x, y) = trainData[indices[t]];

                    var output = _model.Forward(x, training: true);

                    double lossValue = _loss.Forward(output, y);
                    if (double.IsNaN(lossValue) || double.IsInfinity(lossValue))
                        throw new InvalidOperationException($"Loss became NaN/Inf at epoch={epoch}.");

                    epochLossSum += lossValue;
                    epochSamples++;

                    var grad = _loss.Backward();
                    _model.Backward(grad);
                }

                // 2) усредняем градиенты по батчу (чтобы LR не зависел от batchSize)
                ScaleGradients(ps, 1.0 / actualBatch);

                accumCounter++;

                // 3) накопление: step раз в N батчей
                if (accumCounter >= accumSteps)
                {
                    if (options.GradClipNorm.HasValue)
                        ClipGradientsByNorm(ps, options.GradClipNorm.Value);

                    _optimizer.Step(ps);
                    _optimizer.ZeroGrad(ps);

                    accumCounter = 0;
                }
            }

            // если остались накопленные грады — делаем финальный шаг
            if (accumCounter > 0)
            {
                if (options.GradClipNorm.HasValue)
                    ClipGradientsByNorm(ps, options.GradClipNorm.Value);

                _optimizer.Step(ps);
                _optimizer.ZeroGrad(ps);
            }

            // Train loss (средний по эпохе)
            double trainLoss = epochSamples == 0 ? double.NaN : epochLossSum / epochSamples;

            // Val loss (средний) — только loss, без метрик
            double? valLoss = null;
            if (valData != null && valData.Length > 0)
                valLoss = EvaluateLoss(valData, cancellationToken);

            var result = new TrainEpochResult(epoch, trainLoss, valLoss);

            // Callbacks
            foreach (var cb in callbacks)
            {
                cb.OnEpochEnd(result);
                if (result.StopRequested)
                    break;
            }

            if (result.StopRequested)
                break;
        }
    }

    private double EvaluateLoss((double[] x, int y)[] data, CancellationToken cancellationToken)
    {
        double sum = 0.0;
        int n = 0;

        foreach (var (x, y) in data)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var output = _model.Forward(x, training: false);
            double lossValue = _loss.Forward(output, y);

            if (double.IsNaN(lossValue) || double.IsInfinity(lossValue))
                throw new InvalidOperationException("Validation loss became NaN/Inf.");

            sum += lossValue;
            n++;
        }

        return n == 0 ? double.NaN : sum / n;
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

    private static void ClipGradientsByNorm(IEnumerable<IParameter> parameters, double maxNorm)
    {
        if (maxNorm <= 0) return;

        // считаем общую L2 норму
        double sumSq = 0.0;
        foreach (var p in parameters)
        {
            var g = p.Grad;
            for (int i = 0; i < g.Length; i++)
                sumSq += g[i] * g[i];
        }

        double norm = System.Math.Sqrt(sumSq);
        if (norm <= maxNorm || norm == 0.0) return;

        double scale = maxNorm / norm;

        foreach (var p in parameters)
        {
            var g = p.Grad;
            for (int i = 0; i < g.Length; i++)
                g[i] *= scale;
        }
    }

    private static void Shuffle(int[] a, Random rnd)
    {
        for (int i = a.Length - 1; i > 0; i--)
        {
            int j = rnd.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }
}
