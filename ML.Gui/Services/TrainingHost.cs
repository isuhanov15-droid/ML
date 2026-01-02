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

    public bool IsConfigured => _trainerFactory != null && _trainProvider != null && _networkFactory != null;
    public bool IsRunning => _runTask is { IsCompleted: false };
    public Network? CurrentNetwork => _currentNetwork;

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

    public Task StartAsync(TrainOptions options, Action<TrainEpochResult> onEpoch, CancellationToken externalCt)
    {
        if (!IsConfigured) throw new InvalidOperationException("Training pipeline is not configured.");
        if (IsRunning) throw new InvalidOperationException("Training is already running.");
        if (onEpoch == null) throw new ArgumentNullException(nameof(onEpoch));

        _stopRequested = false;
        _cts = CancellationTokenSource.CreateLinkedTokenSource(externalCt);

        var callbacks = BuildCallbacks(options.Callbacks, onEpoch);
        var effectiveOptions = CloneOptions(options, callbacks);

        _runTask = Task.Run(() =>
        {
            _currentNetwork = _preloadedNetwork ?? _networkFactory!();
            _preloadedNetwork = null;

            var trainer = _trainerFactory!(_currentNetwork);
            var trainData = _trainProvider!();
            var valData = _valProvider?.Invoke();

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

            trainer.Train(trainData, effectiveOptions);
        }, _cts.Token);

        return _runTask;
    }

    public void Stop()
    {
        _stopRequested = true;
        _cts?.Cancel();
    }

    public void SaveModel(string modelName, string? filePath = null)
    {
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
