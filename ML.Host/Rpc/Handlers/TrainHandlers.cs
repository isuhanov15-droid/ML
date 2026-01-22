using System.Diagnostics;
using System.Text.Json;
using ML.Core.Serialization;
using ML.Host.Rpc;
using ML.Host.Services;
using ML.Host.Storage;
using ML.Shared.Protocol;
using ML.Shared.Protocol.Training;

namespace ML.Host.Rpc.Handlers;

public sealed class TrainHandlers
{
    private readonly TrainingService _training;
    private readonly EventBus _events;
    private readonly ExperimentStore _experiments;
    private readonly RunStore _runs;
    private readonly InferenceService? _inference;
    private readonly Stopwatch _metricsStopwatch = Stopwatch.StartNew();
    private long _lastMetricsTicks;
    private readonly object _logLock = new();
    private readonly Queue<string> _logBuffer = new();
    private const int MaxLogBuffer = 5000;
    private string? _currentRunId;
    private string? _currentExperimentId;
    private string? _currentProjectId;
    private string? _currentModelPath;

    public TrainHandlers(TrainingService training, EventBus events, ExperimentStore experiments, RunStore runs, InferenceService? inference = null)
    {
        _training = training;
        _events = events;
        _experiments = experiments;
        _runs = runs;
        _inference = inference;

        _training.Log += OnLog;
        _training.StateChanged += OnStateChanged;
        _training.Metrics += OnMetrics;
    }

    public Task<object?> StartAsync(RpcRequest req)
    {
        var config = ParseConfig(req);
        string experimentId = GetRequiredString(req, "experimentId");
        string? projectId = GetOptionalString(req, "projectId");
        if (string.IsNullOrWhiteSpace(projectId))
        {
            var experiment = _experiments.Get(experimentId);
            if (experiment == null)
                throw new InvalidOperationException("Experiment not found.");
            projectId = experiment.projectId;
        }
        _currentModelPath = config.ModelPath;
        if (config.Resume && !string.IsNullOrWhiteSpace(config.ModelPath))
        {
            if (File.Exists(config.ModelPath))
            {
                var loaded = ModelStore.LoadFromFile(config.ModelPath);
                _training.SetNetwork(loaded);
                OnLog($"model.loaded: {config.ModelPath}");
            }
            else
            {
                OnLog($"model.resume.missing: {config.ModelPath}");
            }
        }
        _training.Start(config);
        _currentRunId = _training.CurrentRunId;
        _currentExperimentId = experimentId;
        _currentProjectId = projectId;
        _inference?.RegisterModelPath(_currentProjectId, _currentRunId, _currentModelPath);
        if (_currentRunId != null && projectId != null)
            _runs.Create(_currentRunId, projectId, experimentId);
        var result = new { runId = _currentRunId };
        return Task.FromResult<object?>(result);
    }

    public Task<object?> StopAsync(RpcRequest req)
    {
        _training.Stop();
        return Task.FromResult<object?>(new { ok = true });
    }

    private TrainStartConfig ParseConfig(RpcRequest req)
    {
        if (req.@params is JsonElement el)
        {
            if (el.ValueKind == JsonValueKind.Object && el.TryGetProperty("config", out var cfgEl))
                return Deserialize<TrainStartConfig>(cfgEl);

            return Deserialize<TrainStartConfig>(el);
        }

        return new TrainStartConfig();
    }

    private static T Deserialize<T>(JsonElement element)
    {
        var value = JsonSerializer.Deserialize<T>(element.GetRawText(), ML.Shared.Protocol.JsonOptions.Default);
        if (value == null)
            throw new InvalidOperationException("Invalid payload");
        return value;
    }

    private void OnLog(string message)
    {
        lock (_logLock)
        {
            _logBuffer.Enqueue(message);
            while (_logBuffer.Count > MaxLogBuffer)
                _logBuffer.Dequeue();
        }

        var evt = new RpcEvent(
            v: Protocol.ProtocolVersion,
            type: "log.append",
            data: new { level = "info", message, utc = DateTimeOffset.UtcNow });
        _ = _events.BroadcastAsync(evt);
    }

    private void OnStateChanged(TrainingState state, string? message)
    {
        string? runId = _training.CurrentRunId ?? _currentRunId;
        if (!string.IsNullOrWhiteSpace(runId))
            _runs.UpdateState(runId, MapState(state), message);

        var evt = new RpcEvent(
            v: Protocol.ProtocolVersion,
            type: "train.stateChanged",
            data: new
            {
                runId = runId,
                state = MapState(state),
                message,
                utc = DateTimeOffset.UtcNow
            });
        _ = _events.BroadcastAsync(evt);

        if (state == TrainingState.Finished || state == TrainingState.Stopped)
            SaveModelIfNeeded();
    }

    private void OnMetrics(TrainingMetrics metrics)
    {
        long nowTicks = _metricsStopwatch.ElapsedTicks;
        long minTicks = Stopwatch.Frequency / 10;
        bool hasExtraMetrics = metrics.Acc.HasValue || metrics.ValLoss.HasValue;
        if (!hasExtraMetrics && nowTicks - _lastMetricsTicks < minTicks)
            return;

        _lastMetricsTicks = nowTicks;
        if (!string.IsNullOrWhiteSpace(metrics.RunId))
        {
            var point = new RunMetricsPointDto(
                metrics.RunId,
                metrics.Epoch,
                metrics.Loss,
                metrics.Acc,
                DateTime.UtcNow);
            _runs.AppendMetric(point);
            _runs.UpdateLast(metrics.RunId, metrics.Epoch, metrics.Loss);
        }
        var evt = new RpcEvent(
            v: Protocol.ProtocolVersion,
            type: "metrics.update",
            data: new
            {
                runId = metrics.RunId,
                epoch = metrics.Epoch,
                loss = metrics.Loss,
                valLoss = metrics.ValLoss,
                accuracy = metrics.Acc,
                utc = DateTimeOffset.UtcNow
            });
        _ = _events.BroadcastAsync(evt);
    }

    private void SaveModelIfNeeded()
    {
        try
        {
            if (string.IsNullOrWhiteSpace(_currentModelPath))
                return;
            if (_training.CurrentNetwork == null)
                return;
            ModelStore.SaveToFile(_currentModelPath, _training.CurrentNetwork);
            OnLog($"model.saved: {_currentModelPath}");
            _inference?.RegisterModelPath(_currentProjectId, _currentRunId, _currentModelPath);
        }
        catch (Exception ex)
        {
            OnLog($"model.save.error: {ex.Message}");
        }
    }

    private static string GetRequiredString(RpcRequest req, string name)
    {
        var value = GetOptionalString(req, name);
        if (string.IsNullOrWhiteSpace(value))
            throw new InvalidOperationException($"{name} is required");
        return value;
    }

    private static string? GetOptionalString(RpcRequest req, string name)
    {
        if (req.@params is JsonElement el && el.ValueKind == JsonValueKind.Object)
        {
            if (el.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.String)
                return prop.GetString();
        }

        return null;
    }

    private static string MapState(TrainingState state)
    {
        return state switch
        {
            TrainingState.Idle => "queued",
            TrainingState.Running => "running",
            TrainingState.Stopping => "running",
            TrainingState.Stopped => "stopped",
            TrainingState.Finished => "finished",
            TrainingState.Error => "failed",
            _ => "running"
        };
    }
}
