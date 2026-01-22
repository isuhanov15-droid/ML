using System.Text.Json;
using ML.Gui.Models;
using ML.Gui.Net;

namespace ML.Gui.Services;

public sealed class TrainingRemoteService
{
    private readonly CoreTcpClient _client;
    private readonly ClientRouter _router;

    public event Action<MetricPoint>? MetricsReceived;
    public event Action<TrainingState, string?>? TrainStateChanged;
    public event Action<string>? LogReceived;
    public event Action<string?>? RunIdChanged;

    public string? CurrentRunId { get; private set; }

    public TrainingRemoteService(CoreTcpClient client, ClientRouter router)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _router = router ?? throw new ArgumentNullException(nameof(router));

        _router.On("train.started", HandleTrainStarted);
        _router.On("train.stopped", HandleTrainStopped);
        _router.On("train.completed", HandleTrainCompleted);
        _router.On("train.state", HandleTrainState);
        _router.On("train.metrics", HandleMetrics);
        _router.On("train.log", HandleLog);
    }

    public Task StartAsync(TrainStartConfig config, CancellationToken ct = default)
    {
        if (config == null) throw new ArgumentNullException(nameof(config));
        return _client.SendAsync("train.start", new { config }, NewRid(), ct);
    }

    public Task StopAsync(string? runId = null, CancellationToken ct = default)
    {
        return _client.SendAsync("train.stop", new { runId }, NewRid(), ct);
    }

    public Task ContinueAsync(string? runId = null, CancellationToken ct = default)
    {
        return _client.SendAsync("train.continue", new { runId }, NewRid(), ct);
    }

    private void HandleTrainStarted(string? rid, JsonElement data)
    {
        CurrentRunId = data.ValueKind == JsonValueKind.Object && data.TryGetProperty("runId", out var id)
            ? id.GetString()
            : null;

        RunIdChanged?.Invoke(CurrentRunId);
        TrainStateChanged?.Invoke(TrainingState.Running, null);
    }

    private void HandleTrainStopped(string? rid, JsonElement data)
    {
        CurrentRunId = data.ValueKind == JsonValueKind.Object && data.TryGetProperty("runId", out var id)
            ? id.GetString()
            : CurrentRunId;

        RunIdChanged?.Invoke(CurrentRunId);
        TrainStateChanged?.Invoke(TrainingState.Stopped, null);
    }

    private void HandleTrainCompleted(string? rid, JsonElement data)
    {
        CurrentRunId = data.ValueKind == JsonValueKind.Object && data.TryGetProperty("runId", out var id)
            ? id.GetString()
            : CurrentRunId;

        RunIdChanged?.Invoke(CurrentRunId);
        TrainStateChanged?.Invoke(TrainingState.Finished, null);
    }

    private void HandleTrainState(string? rid, JsonElement data)
    {
        var stateStr = data.ValueKind == JsonValueKind.Object && data.TryGetProperty("state", out var s)
            ? s.GetString()
            : null;
        var message = data.ValueKind == JsonValueKind.Object && data.TryGetProperty("message", out var m)
            ? m.GetString()
            : null;

        var state = stateStr?.ToLowerInvariant() switch
        {
            "running" => TrainingState.Running,
            "stopping" => TrainingState.Stopping,
            "stopped" => TrainingState.Stopped,
            "finished" => TrainingState.Finished,
            "error" => TrainingState.Error,
            _ => TrainingState.Idle
        };

        TrainStateChanged?.Invoke(state, message);
    }

    private void HandleMetrics(string? rid, JsonElement data)
    {
        if (data.ValueKind != JsonValueKind.Object) return;

        int epoch = data.TryGetProperty("epoch", out var epochEl) && epochEl.ValueKind == JsonValueKind.Number
            ? epochEl.GetInt32()
            : 0;

        if (!data.TryGetProperty("loss", out var lossEl) || lossEl.ValueKind != JsonValueKind.Number)
            return;
        double loss = lossEl.GetDouble();

        double? valLoss = data.TryGetProperty("valLoss", out var valLossEl) && valLossEl.ValueKind == JsonValueKind.Number
            ? valLossEl.GetDouble()
            : null;

        double? acc = data.TryGetProperty("acc", out var accEl) && accEl.ValueKind == JsonValueKind.Number
            ? accEl.GetDouble()
            : null;

        double? lr = data.TryGetProperty("lr", out var lrEl) && lrEl.ValueKind == JsonValueKind.Number
            ? lrEl.GetDouble()
            : null;

        long? elapsed = data.TryGetProperty("elapsedMs", out var elapsedEl) && elapsedEl.ValueKind == JsonValueKind.Number
            ? elapsedEl.GetInt64()
            : null;

        string? runId = data.TryGetProperty("runId", out var runEl) ? runEl.GetString() : null;

        MetricsReceived?.Invoke(new MetricPoint(epoch, loss, valLoss, acc, lr, elapsed, runId));
    }

    private void HandleLog(string? rid, JsonElement data)
    {
        if (data.ValueKind != JsonValueKind.Object)
        {
            LogReceived?.Invoke(data.ToString());
            return;
        }

        var level = data.TryGetProperty("level", out var lvl) ? lvl.GetString() : "info";
        var message = data.TryGetProperty("message", out var msg) ? msg.GetString() : data.ToString();

        LogReceived?.Invoke($"[{level}] {message}");
    }

    private static string NewRid() => Guid.NewGuid().ToString("N");
}
