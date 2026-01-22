using System.Collections.Generic;
using System.Diagnostics;
using System.Text.Json;
using ML.Core;
using ML.Core.Abstractions;
using ML.Core.Data;
using ML.Core.Layers;
using ML.Core.Losses;
using ML.Core.Optimizers;
using ML.Core.Training;
using ML.Core.Training.Callbacks;
using ML.Shared.Protocol.Training;

namespace ML.Host.Services;

public sealed class TrainingService
{
    private readonly object _lock = new();
    private CancellationTokenSource? _cts;
    private Task? _runTask;
    private Network? _currentNetwork;
    private IOptimizer? _currentOptimizer;
    private (double[] x, int y)[]? _trainData;
    private (double[] x, int y)[]? _valData;
    private string? _runId;

    private readonly Stopwatch _stopwatch = new();

    public event Action<string>? Log;
    public event Action<TrainingState, string?>? StateChanged;
    public event Action<string?>? RunIdChanged;
    public event Action<TrainingMetrics>? Metrics;

    public bool IsRunning => _runTask is { IsCompleted: false };

    public string? CurrentRunId => _runId;
    public Network? CurrentNetwork => _currentNetwork;

    public void SetNetwork(Network network)
    {
        _currentNetwork = network;
        _currentOptimizer = null;
    }

    public void Start(TrainStartConfig config)
    {
        if (config == null) throw new ArgumentNullException(nameof(config));

        lock (_lock)
        {
            if (IsRunning)
                throw new InvalidOperationException("Training is already running.");

            _cts = new CancellationTokenSource();
        }

        try
        {
            BuildRuntime(config);
        }
        catch
        {
            lock (_lock)
            {
                _cts?.Dispose();
                _cts = null;
            }
            throw;
        }

        lock (_lock)
        {
            _runId = Guid.NewGuid().ToString("N");
            RunIdChanged?.Invoke(_runId);
        }
        StateChanged?.Invoke(TrainingState.Running, null);
        var every = config.Train.UiEveryNEpochs <= 0 ? 1 : config.Train.UiEveryNEpochs;
        Log?.Invoke($"train.start runId={_runId} epochs={config.Train.Epochs} uiEvery={every}");

        _stopwatch.Restart();
        _runTask = Task.Run(() => TrainLoop(config, _cts!.Token));
    }

    public void Stop()
    {
        lock (_lock)
        {
            if (_cts == null) return;
            StateChanged?.Invoke(TrainingState.Stopping, null);
            _cts.Cancel();
        }
    }

    private void BuildRuntime(TrainStartConfig config)
    {
        if (config.Resume && _currentNetwork != null)
        {
            // reuse current network/optimizer
            if (_currentOptimizer == null)
                _currentOptimizer = new AdamOptimizer(config.Train.LearningRate);
        }
        else
        {
            _currentNetwork = BuildNetwork(config.Network);
            _currentOptimizer = new AdamOptimizer(config.Train.LearningRate);
        }

        _trainData = LoadDataset(config.Data).ToArray();
        _valData = _trainData;
    }

    private void TrainLoop(TrainStartConfig config, CancellationToken ct)
    {
        try
        {
            if (_currentNetwork == null || _currentOptimizer == null)
                throw new InvalidOperationException("Network not initialized.");

        var loss = new CrossEntropyLoss();

        var callbacks = new List<ITrainCallback>
        {
            new ProgressCallback(r => OnEpoch(r, config))
        };

        int patience = config.Train.EarlyStopPatience ?? 200;
        double minDelta = config.Train.EarlyStopMinDelta ?? 1e-6;
        if (patience > 0)
        {
            callbacks.Add(new EarlyStoppingByLoss(
                metric: r => r.ValLoss ?? r.TrainLoss,
                patience: patience,
                minDelta: minDelta,
                onStop: () => Log?.Invoke("train.early_stop")));
        }

        var options = new TrainOptions
        {
            Epochs = config.Train.Epochs,
            BatchSize = config.Train.BatchSize,
            Shuffle = config.Train.Shuffle,
            DropLast = config.Train.DropLast,
            GradClipNorm = config.Train.GradClipNorm,
            GradientAccumulationSteps = config.Train.AccumulationSteps,
            Seed = config.Network.Seed,
            Validation = _valData,
            Callbacks = callbacks
        };

            var trainer = new Trainer(_currentNetwork, _currentOptimizer, loss);
            trainer.Train(_trainData!, options, ct);

            StateChanged?.Invoke(TrainingState.Finished, null);
            Log?.Invoke("train.completed");
        }
        catch (OperationCanceledException)
        {
            StateChanged?.Invoke(TrainingState.Stopped, null);
            Log?.Invoke("train.stopped");
        }
        catch (Exception ex)
        {
            StateChanged?.Invoke(TrainingState.Error, ex.Message);
            Log?.Invoke($"train.error: {ex.Message}");
        }
    }

    private void OnEpoch(TrainEpochResult result, TrainStartConfig config)
    {
        var every = config.Train.UiEveryNEpochs <= 0 ? 1 : config.Train.UiEveryNEpochs;
        bool shouldSend = result.Epoch == 1 ||
                          result.Epoch % every == 0 ||
                          result.Epoch == config.Train.Epochs;
        if (!shouldSend)
            return;

        double? acc = null;
        if (result.Epoch % 10 == 0 && _currentNetwork != null && _trainData != null)
            acc = ComputeAccuracy(_currentNetwork, _trainData);

        Metrics?.Invoke(new TrainingMetrics
        {
            RunId = _runId,
            Epoch = result.Epoch,
            Loss = result.TrainLoss,
            ValLoss = result.ValLoss,
            Acc = acc,
            LearningRate = config.Train.LearningRate,
            ElapsedMs = _stopwatch.ElapsedMilliseconds
        });
    }

    private static IEnumerable<(double[] x, int y)> LoadDataset(TrainDataConfig data)
    {
        return data.Preset.ToUpperInvariant() switch
        {
            "XOR" => DemoDatasets.Xor,
            "AND" => DemoDatasets.And,
            "FILE" => DatasetLoader.LoadClassification(data.DatasetPath ?? throw new InvalidOperationException("datasetPath required")),
            _ => throw new InvalidOperationException($"Unknown preset: {data.Preset}")
        };
    }

    private static Network BuildNetwork(TrainNetworkConfig cfg)
    {
        var net = new Network();
        int input = cfg.InputSize;
        int seed = cfg.Seed ?? 123;

        foreach (var hidden in cfg.Hidden)
        {
            net.Add(new LinearLayer(input, hidden, seed: seed));
            net.Add(new ActivationLayer(hidden, ParseActivation(cfg.Activation)));
            input = hidden;
            seed++;
        }

        net.Add(new LinearLayer(input, cfg.OutputSize, seed: seed));
        net.Add(new SoftmaxLayer(cfg.OutputSize));
        return net;
    }

    private static ActivationType ParseActivation(string? activation)
    {
        if (string.IsNullOrWhiteSpace(activation))
            return ActivationType.ReLu;

        return Enum.TryParse<ActivationType>(activation, ignoreCase: true, out var act)
            ? act
            : ActivationType.ReLu;
    }

    private static double ComputeAccuracy(Network net, (double[] x, int y)[] data)
    {
        int correct = 0;
        for (int i = 0; i < data.Length; i++)
        {
            var (x, y) = data[i];
            var p = net.Forward(x, training: false);
            int pred = ArgMax(p);
            if (pred == y) correct++;
        }

        return data.Length == 0 ? 0 : (double)correct / data.Length;
    }

    private static int ArgMax(double[] v)
    {
        int idx = 0;
        double max = v[0];
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > max)
            {
                max = v[i];
                idx = i;
            }
        }
        return idx;
    }
}

public enum TrainingState
{
    Idle,
    Running,
    Stopping,
    Stopped,
    Finished,
    Error
}

public sealed class TrainingMetrics
{
    public string? RunId { get; set; }
    public int Epoch { get; set; }
    public double Loss { get; set; }
    public double? ValLoss { get; set; }
    public double? Acc { get; set; }
    public double? LearningRate { get; set; }
    public long ElapsedMs { get; set; }
}
