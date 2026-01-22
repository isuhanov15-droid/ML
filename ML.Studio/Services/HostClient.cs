using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Text.Json;
using ML.Shared.Protocol;

namespace ML.Studio.Services;

public sealed class HostClient : IAsyncDisposable
{
    private TcpClient? _client;
    private NetworkStream? _stream;
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private readonly ConcurrentDictionary<string, TaskCompletionSource<RpcResponse>> _pending = new();
    private CancellationTokenSource? _cts;
    private Task? _recvTask;

    public bool IsConnected => _client != null && _client.Connected;

    public event Action<RpcEvent>? EventReceived;
    public event Action? Disconnected;

    public async Task ConnectAsync(string host, int port, CancellationToken ct = default)
    {
        if (IsConnected)
            return;

        _client = new TcpClient();
        await _client.ConnectAsync(host, port, ct);
        _client.NoDelay = true;
        _stream = _client.GetStream();
        _cts = new CancellationTokenSource();
        _recvTask = Task.Run(() => RecvLoopAsync(_cts.Token), CancellationToken.None);
    }

    public async Task DisconnectAsync()
    {
        await DisposeAsync();
    }

    public async Task<T> CallAsync<T>(string method, object? paramsObj, CancellationToken ct = default)
    {
        if (_stream == null)
            throw new InvalidOperationException("Not connected.");

        string id = Guid.NewGuid().ToString("N");
        var req = new RpcRequest(Protocol.ProtocolVersion, id, method, paramsObj);
        var tcs = new TaskCompletionSource<RpcResponse>(TaskCreationOptions.RunContinuationsAsynchronously);
        _pending[id] = tcs;

        string json = JsonSerializer.Serialize(req, JsonOptions.Default);
        await SendJsonAsync(json, ct);

        using (ct.Register(() =>
        {
            _pending.TryRemove(id, out _);
            tcs.TrySetCanceled(ct);
        }))
        {
            RpcResponse resp = await tcs.Task;
            if (!resp.ok)
            {
                string message = resp.error?.message ?? "Unknown error";
                throw new InvalidOperationException(message);
            }

            if (resp.result is JsonElement el)
            {
                var value = JsonSerializer.Deserialize<T>(el.GetRawText(), JsonOptions.Default);
                if (value == null)
                    throw new InvalidOperationException("Invalid response payload.");
                return value;
            }

            return (T)resp.result!;
        }
    }

    private async Task SendJsonAsync(string json, CancellationToken ct)
    {
        if (_stream == null)
            throw new InvalidOperationException("Not connected.");

        await _sendLock.WaitAsync(ct);
        try
        {
            await LengthPrefixedStream.WriteAsync(_stream, json, ct);
        }
        finally
        {
            _sendLock.Release();
        }
    }

    private async Task RecvLoopAsync(CancellationToken ct)
    {
        try
        {
            if (_stream == null)
                return;

            while (!ct.IsCancellationRequested)
            {
                string? json = await LengthPrefixedStream.ReadAsync(_stream, ct);
                if (json == null)
                    break;

                if (!TryDispatchMessage(json))
                    continue;
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch
        {
        }
        finally
        {
            foreach (var pending in _pending.Values)
                pending.TrySetException(new IOException("Disconnected."));
            _pending.Clear();

            Disconnected?.Invoke();
        }
    }

    private bool TryDispatchMessage(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (root.TryGetProperty("type", out _))
        {
            var evt = JsonSerializer.Deserialize<RpcEvent>(json, JsonOptions.Default);
            if (evt != null)
            {
                EventReceived?.Invoke(evt);
                return true;
            }
        }

        if (root.TryGetProperty("ok", out _) && root.TryGetProperty("id", out _))
        {
            var resp = JsonSerializer.Deserialize<RpcResponse>(json, JsonOptions.Default);
            if (resp != null)
            {
                if (_pending.TryRemove(resp.id, out var tcs))
                    tcs.TrySetResult(resp);
                return true;
            }
        }

        return false;
    }

    public async ValueTask DisposeAsync()
    {
        if (_cts != null)
        {
            try { _cts.Cancel(); } catch { }
        }

        if (_recvTask != null)
        {
            try { await _recvTask; } catch { }
        }

        try { _stream?.Close(); } catch { }
        try { _client?.Close(); } catch { }

        foreach (var pending in _pending.Values)
            pending.TrySetException(new IOException("Disconnected."));
        _pending.Clear();

        _stream = null;
        _client = null;
        if (_cts != null)
        {
            _cts.Dispose();
            _cts = null;
        }
    }
}
