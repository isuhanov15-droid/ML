using System.Net.Sockets;
using System.Text.Json;
using ML.Shared.Protocol;

namespace ML.Host.Net;

public sealed class ClientConnection : IAsyncDisposable
{
    private readonly TcpClient _client;
    private readonly NetworkStream _stream;
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private readonly CancellationTokenSource _cts = new();
    private Task? _recvTask;

    private static readonly JsonSerializerOptions JsonOptions = ProtocolJson.Options;

    public event Action<ClientConnection, Envelope>? MessageReceived;
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

    public async Task SendAsync(string type, object? data = null, string? rid = null, CancellationToken ct = default)
    {
        var env = new OutgoingEnvelope
        {
            Type = type,
            Rid = rid,
            Ts = DateTimeOffset.UtcNow,
            Data = data
        };

        byte[] payload = JsonSerializer.SerializeToUtf8Bytes(env, JsonOptions);

        await _sendLock.WaitAsync(ct);
        try
        {
            await TcpFraming.WriteFrameAsync(_stream, payload, ct);
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
                var payload = await TcpFraming.ReadFrameAsync(_stream, ct);
                if (payload == null)
                    break;

                Envelope? env;
                try
                {
                    env = JsonSerializer.Deserialize<Envelope>(payload, JsonOptions);
                }
                catch
                {
                    continue;
                }

                if (env == null || string.IsNullOrWhiteSpace(env.Type))
                    continue;

                MessageReceived?.Invoke(this, env);
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
