using System.Collections.Concurrent;
using System.Text.Json;
using ML.Gui.Models;
using ML.Gui.Net;

namespace ML.Gui.Services;

public sealed class ModelStoreRemoteService
{
    private readonly CoreTcpClient _client;
    private readonly ClientRouter _router;
    private readonly ConcurrentDictionary<string, TaskCompletionSource<RemoteResult>> _saveWaiters = new();
    private readonly ConcurrentDictionary<string, TaskCompletionSource<RemoteResult>> _loadWaiters = new();

    public ModelStoreRemoteService(CoreTcpClient client, ClientRouter router)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _router = router ?? throw new ArgumentNullException(nameof(router));

        _router.On("model.save.result", HandleSaveResult);
        _router.On("model.load.result", HandleLoadResult);
    }

    public Task<RemoteResult> SaveAsync(string? modelName, string? filePath, bool? includeOptimizerState = null, CancellationToken ct = default)
    {
        var rid = NewRid();
        var tcs = new TaskCompletionSource<RemoteResult>(TaskCreationOptions.RunContinuationsAsynchronously);
        _saveWaiters[rid] = tcs;

        ct.Register(() =>
        {
            _saveWaiters.TryRemove(rid, out _);
            tcs.TrySetCanceled(ct);
        });
        _client.SendAsync("model.save", new { modelName, filePath, includeOptimizerState }, rid, ct)
            .ContinueWith(task => { if (task.IsFaulted) tcs.TrySetException(task.Exception!); }, TaskScheduler.Default);

        return tcs.Task;
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
        _client.SendAsync("model.load", new { modelName, filePath }, rid, ct)
            .ContinueWith(task => { if (task.IsFaulted) tcs.TrySetException(task.Exception!); }, TaskScheduler.Default);

        return tcs.Task;
    }

    private void HandleSaveResult(string? rid, JsonElement data)
    {
        if (rid == null || !_saveWaiters.TryRemove(rid, out var tcs))
            return;

        bool ok = data.TryGetProperty("ok", out var okEl) && okEl.ValueKind == JsonValueKind.True;
        string? message = data.TryGetProperty("message", out var msg) ? msg.GetString() : null;
        tcs.TrySetResult(new RemoteResult(ok, message));
    }

    private void HandleLoadResult(string? rid, JsonElement data)
    {
        if (rid == null || !_loadWaiters.TryRemove(rid, out var tcs))
            return;

        bool ok = data.TryGetProperty("ok", out var okEl) && okEl.ValueKind == JsonValueKind.True;
        string? message = data.TryGetProperty("message", out var msg) ? msg.GetString() : null;
        tcs.TrySetResult(new RemoteResult(ok, message));
    }

    private static string NewRid() => Guid.NewGuid().ToString("N");
}
