using System.Collections.Concurrent;
using System.Text.Json;
using ML.Gui.Models;
using ML.Gui.Net;

namespace ML.Gui.Services;

public sealed class InferenceRemoteService
{
    private readonly CoreTcpClient _client;
    private readonly ClientRouter _router;
    private readonly ConcurrentDictionary<string, TaskCompletionSource<PredictResult>> _predictWaiters = new();
    private readonly ConcurrentDictionary<string, TaskCompletionSource<RemoteResult>> _loadWaiters = new();

    public InferenceRemoteService(CoreTcpClient client, ClientRouter router)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _router = router ?? throw new ArgumentNullException(nameof(router));

        _router.On("infer.load.result", HandleLoadResult);
        _router.On("infer.predict.result", HandlePredictResult);
    }

    public Task<RemoteResult> LoadAsync(string? modelName, string? filePath, CancellationToken ct = default)
    {
        var rid = NewRid();
        var tcs = new TaskCompletionSource<RemoteResult>(TaskCreationOptions.RunContinuationsAsynchronously);
        _loadWaiters[rid] = tcs;

        ct.Register(() =>
        {
            _loadWaiters.TryRemove(rid, out _);
            tcs.TrySetCanceled(ct);
        });
        _client.SendAsync("infer.load", new { modelName, filePath }, rid, ct)
            .ContinueWith(task => { if (task.IsFaulted) tcs.TrySetException(task.Exception!); }, TaskScheduler.Default);

        return tcs.Task;
    }

    public Task<PredictResult> PredictAsync(double[] input, CancellationToken ct = default)
    {
        var rid = NewRid();
        var tcs = new TaskCompletionSource<PredictResult>(TaskCreationOptions.RunContinuationsAsynchronously);
        _predictWaiters[rid] = tcs;

        ct.Register(() =>
        {
            _predictWaiters.TryRemove(rid, out _);
            tcs.TrySetCanceled(ct);
        });
        _client.SendAsync("infer.predict", new { input }, rid, ct)
            .ContinueWith(task => { if (task.IsFaulted) tcs.TrySetException(task.Exception!); }, TaskScheduler.Default);

        return tcs.Task;
    }

    private void HandleLoadResult(string? rid, JsonElement data)
    {
        if (rid == null || !_loadWaiters.TryRemove(rid, out var tcs))
            return;

        bool ok = data.TryGetProperty("ok", out var okEl) && okEl.ValueKind == JsonValueKind.True;
        string? message = data.TryGetProperty("message", out var msg) ? msg.GetString() : null;
        tcs.TrySetResult(new RemoteResult(ok, message));
    }

    private void HandlePredictResult(string? rid, JsonElement data)
    {
        if (rid == null || !_predictWaiters.TryRemove(rid, out var tcs))
            return;

        bool ok = data.TryGetProperty("ok", out var okEl) && okEl.ValueKind == JsonValueKind.True;
        string? message = data.TryGetProperty("message", out var msg) ? msg.GetString() : null;
        long? latency = data.TryGetProperty("latencyMs", out var lat) && lat.ValueKind == JsonValueKind.Number
            ? lat.GetInt64()
            : null;

        double[] output = Array.Empty<double>();
        if (data.TryGetProperty("output", out var outEl) && outEl.ValueKind == JsonValueKind.Array)
        {
            output = outEl.EnumerateArray().Where(x => x.ValueKind == JsonValueKind.Number).Select(x => x.GetDouble()).ToArray();
        }

        tcs.TrySetResult(new PredictResult(ok, output, latency, message));
    }

    private static string NewRid() => Guid.NewGuid().ToString("N");
}

public sealed record PredictResult(bool Ok, double[] Output, long? LatencyMs, string? Message);
