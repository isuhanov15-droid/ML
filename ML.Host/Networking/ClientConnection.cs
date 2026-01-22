using System.Net.Sockets;
using System.Text.Json;
using ML.Shared.Protocol;

namespace ML.Host.Networking;

public sealed class ClientConnection : IAsyncDisposable
{
    private readonly TcpClient _client;
    private readonly NetworkStream _stream;
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private readonly CancellationTokenSource _cts = new();
    private Task? _recvTask;

    public event Action<ClientConnection, RpcRequest>? RequestReceived;
    public event Action<ClientConnection>? Disconnected;

    public ClientConnection(TcpClient client)
    {
        _client = client;
        _client.NoDelay = true;
        _stream = client.GetStream();
    }

    public void Start()
    {
        _recvTask = Task.Run(() => RecvLoopAsync(_cts.Token), CancellationToken.None);
    }

    public async Task SendResponseAsync(RpcResponse response, CancellationToken ct = default)
    {
        string json = JsonSerializer.Serialize(response, JsonOptions.Default);
        await SendJsonAsync(json, ct);
    }

    public async Task SendEventAsync(RpcEvent evt, CancellationToken ct = default)
    {
        string json = JsonSerializer.Serialize(evt, JsonOptions.Default);
        await SendJsonAsync(json, ct);
    }

    private async Task SendJsonAsync(string json, CancellationToken ct)
    {
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
            while (!ct.IsCancellationRequested)
            {
                string? json = await LengthPrefixedStream.ReadAsync(_stream, ct);
                if (json == null)
                    break;

                RpcRequest? req;
                try
                {
                    req = JsonSerializer.Deserialize<RpcRequest>(json, JsonOptions.Default);
                }
                catch
                {
                    continue;
                }

                if (req == null)
                    continue;

                RequestReceived?.Invoke(this, req);
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
            Disconnected?.Invoke(this);
        }
    }

    public async ValueTask DisposeAsync()
    {
        try { _cts.Cancel(); } catch { }

        if (_recvTask != null)
        {
            try { await _recvTask; } catch { }
        }

        try { _stream.Close(); } catch { }
        try { _client.Close(); } catch { }
        _cts.Dispose();
        _sendLock.Dispose();
    }
}
