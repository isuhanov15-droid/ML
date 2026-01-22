using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using ML.Core;
using ML.Core.Training;
using ML.Core.Training.Callbacks;
using ML.Core.Serialization;

namespace ML.Gui.Services;

/// <summary>
/// Thin wrapper around Trainer to make it cancellable and observable from UI.
/// </summary>
public sealed class TrainingHost
{
    private Func<Network>? _networkFactory;
    private Func<Network, Trainer>? _trainerFactory;
    private Func<IEnumerable<(double[] x, int y)>>? _trainProvider;
    private Func<IEnumerable<(double[] x, int y)>?>? _valProvider;
    private Task? _runTask;
    private CancellationTokenSource? _cts;
    private volatile bool _stopRequested;

    private Network? _preloadedNetwork;
    private Network? _currentNetwork;
    private (double[] x, int y)[]? _evalData;

    public bool IsConfigured => _trainerFactory != null && _trainProvider != null && _networkFactory != null;
    public bool IsRunning => _runTask is { IsCompleted: false };
    public Network? CurrentNetwork => _currentNetwork;
    public Func<IEnumerable<(double[] x, int y)>>? CurrentTrainProvider => _trainProvider;
    public Func<IEnumerable<(double[] x, int y)>?>? CurrentValProvider => _valProvider;
    public bool HasCheckpoint => _currentNetwork != null;

    public void Configure(
        Func<Network> networkFactory,
        Func<Network, Trainer> trainerFactory,
        Func<IEnumerable<(double[] x, int y)>> trainProvider,
        Func<IEnumerable<(double[] x, int y)>?>? valProvider = null)
    {
        _networkFactory = networkFactory ?? throw new ArgumentNullException(nameof(networkFactory));
        _trainerFactory = trainerFactory ?? throw new ArgumentNullException(nameof(trainerFactory));
        _trainProvider = trainProvider ?? throw new ArgumentNullException(nameof(trainProvider));
        _valProvider = valProvider;
    }

    public void SetDataProviders(
        Func<IEnumerable<(double[] x, int y)>> trainProvider,
        Func<IEnumerable<(double[] x, int y)>?>? valProvider = null)
    {
        _trainProvider = trainProvider ?? throw new ArgumentNullException(nameof(trainProvider));
        _valProvider = valProvider;
        _evalData = null;
    }

    public Task StartAsync(TrainOptions options, Action<TrainEpochResult> onEpoch, CancellationToken externalCt, bool useCheckpoint = false)
    {
        if (!IsConfigured) throw new InvalidOperationException("Training pipeline is not configured.");
        if (IsRunning) throw new InvalidOperationException("Training is already running.");
        if (onEpoch == null) throw new ArgumentNullException(nameof(onEpoch));

        _stopRequested = false;
        _cts = CancellationTokenSource.CreateLinkedTokenSource(externalCt);

        var callbacks = BuildCallbacks(options.Callbacks, onEpoch);
        var effectiveOptions = CloneOptions(options, callbacks);

        _runTask = Task.Factory.StartNew(() =>
        {
            _currentNetwork = useCheckpoint ? _currentNetwork ?? _preloadedNetwork ?? _networkFactory!() : _preloadedNetwork ?? _networkFactory!();
            _preloadedNetwork = null;

            var trainer = _trainerFactory!(_currentNetwork);
            var trainData = _trainProvider!().ToArray();
            var valData = _valProvider?.Invoke()?.ToArray();

            _evalData = valData ?? trainData;

            // override validation if provided at runtime
            if (valData != null)
            {
                effectiveOptions = new TrainOptions
                {
                    Epochs = effectiveOptions.Epochs,
                    BatchSize = effectiveOptions.BatchSize,
                    Shuffle = effectiveOptions.Shuffle,
                    DropLast = effectiveOptions.DropLast,
                    GradClipNorm = effectiveOptions.GradClipNorm,
                    GradientAccumulationSteps = effectiveOptions.GradientAccumulationSteps,
                    Seed = effectiveOptions.Seed,
                    Validation = valData,
                    Callbacks = effectiveOptions.Callbacks
                };
            }

            trainer.Train(trainData, effectiveOptions, _cts.Token);
        }, _cts.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default);

        return _runTask;
    }

    public void Stop()
    {
        _stopRequested = true;
        _cts?.Cancel();
    }

    public void PrepareResume()
    {
        if (_currentNetwork != null)
            _preloadedNetwork = _currentNetwork;
    }

    public double? ComputeAccuracy(CancellationToken ct = default)
    {
        var data = _evalData;
        var model = _currentNetwork;
        if (data == null || data.Length == 0 || model == null)
            return null;

        int ok = 0;
        int total = data.Length;

        foreach (var (x, y) in data)
        {
            if (ct.IsCancellationRequested)
                throw new OperationCanceledException(ct);

            var p = model.Forward(x, training: false);
            int pred = ArgMax(p);
            if (pred == y) ok++;
        }

        return total == 0 ? null : (double)ok / total;
    }

    private static int ArgMax(double[] v)
    {
        int idx = 0;
        double max = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > max) { max = v[i]; idx = i; }
        return idx;
    }

    public void SaveModel(string modelName, string? filePath = null)
    {
        if (_currentNetwork == null)
            _currentNetwork = _preloadedNetwork ?? _networkFactory?.Invoke();

        if (_currentNetwork == null)
            throw new InvalidOperationException("Нет активной модели для сохранения. Запустите обучение или загрузите модель.");

        if (!string.IsNullOrWhiteSpace(filePath))
            ModelStore.SaveToFile(filePath, _currentNetwork);
        else
            ModelStore.Save(modelName, _currentNetwork);
    }

    public void LoadModel(string modelName, string? filePath = null)
    {
        _preloadedNetwork = !string.IsNullOrWhiteSpace(filePath)
            ? ModelStore.LoadFromFile(filePath)
            : ModelStore.Load(modelName);
        _currentNetwork = _preloadedNetwork;
        _evalData = null;
    }

    private IEnumerable<ITrainCallback> BuildCallbacks(IEnumerable<ITrainCallback>? existing, Action<TrainEpochResult> onEpoch)
    {
        var list = existing?.ToList() ?? new List<ITrainCallback>();
        list.Add(new ManualStopCallback(() => _stopRequested || _cts?.IsCancellationRequested == true));
        list.Add(new ProgressCallback(onEpoch));
        return list;
    }

    private static TrainOptions CloneOptions(TrainOptions src, IEnumerable<ITrainCallback> callbacks)
    {
        return new TrainOptions
        {
            Epochs = src.Epochs,
            BatchSize = src.BatchSize,
            Shuffle = src.Shuffle,
            DropLast = src.DropLast,
            GradClipNorm = src.GradClipNorm,
            GradientAccumulationSteps = src.GradientAccumulationSteps,
            Seed = src.Seed,
            Validation = src.Validation,
            Callbacks = callbacks
        };
    }
}
