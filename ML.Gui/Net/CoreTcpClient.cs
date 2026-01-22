using System.Buffers.Binary;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace ML.Gui.Net;

public sealed class CoreTcpClient : IAsyncDisposable
{
    public event Action<string, string?, JsonElement>? MessageReceived;
    public event Action? Disconnected;
    public event Action<Exception>? Error;

    private TcpClient? _tcp;
    private NetworkStream? _stream;
    private Task? _rxLoop;
    private CancellationTokenSource? _cts;
    private readonly object _gate = new();

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        PropertyNameCaseInsensitive = true
    };

    public bool IsConnected => _tcp?.Connected == true;

    public async Task ConnectAsync(string host, int port)
    {
        await DisconnectAsync();

        _tcp = new TcpClient();
        _tcp.NoDelay = true;
        await _tcp.ConnectAsync(host, port);
        _stream = _tcp.GetStream();
        _cts = new CancellationTokenSource();
        _rxLoop = Task.Run(() => RxLoopAsync(_cts.Token));
    }

    public async Task DisconnectAsync()
    {
        Task? rxLoop = null;
        CancellationTokenSource? cts = null;
        NetworkStream? stream = null;
        TcpClient? tcp = null;

        lock (_gate)
        {
            cts = _cts;
            rxLoop = _rxLoop;
            stream = _stream;
            tcp = _tcp;
            _cts = null;
            _rxLoop = null;
            _stream = null;
            _tcp = null;
        }

        try { cts?.Cancel(); } catch { }
        try { if (rxLoop != null) await rxLoop; } catch { }
        try { stream?.Close(); } catch { }
        try { tcp?.Close(); } catch { }
        try { cts?.Dispose(); } catch { }
    }

    public async ValueTask DisposeAsync() => await DisconnectAsync();

    public async Task SendAsync(string type, object? data = null, string? rid = null, CancellationToken ct = default)
    {
        if (_stream == null) throw new InvalidOperationException("Not connected");

        var env = new OutgoingEnvelope
        {
            Type = type,
            Rid = rid,
            Ts = DateTimeOffset.UtcNow,
            Data = data
        };

        var json = JsonSerializer.Serialize(env, JsonOpts);
        var payload = Encoding.UTF8.GetBytes(json);

        if (payload.Length <= 0 || payload.Length > TcpFraming.MaxFrameBytes)
            throw new InvalidDataException($"Invalid payload length: {payload.Length}");

        await TcpFraming.WriteFrameAsync(_stream, payload, ct);
    }

    private async Task RxLoopAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested && _stream != null)
            {
                var payload = await TcpFraming.ReadFrameAsync(_stream, ct);
                if (payload == null) return;

                Envelope? env;
                try
                {
                    env = JsonSerializer.Deserialize<Envelope>(payload, JsonOpts);
                }
                catch (Exception ex)
                {
                    Error?.Invoke(ex);
                    continue;
                }

                if (env == null || string.IsNullOrWhiteSpace(env.Type))
                    continue;

                MessageReceived?.Invoke(env.Type, env.Rid, env.Data);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Error?.Invoke(ex);
        }
        finally
        {
            Disconnected?.Invoke();
        }
    }

    private static class TcpFraming
    {
        public const int MaxFrameBytes = 50_000_000;

        public static async Task WriteFrameAsync(NetworkStream stream, ReadOnlyMemory<byte> payload, CancellationToken ct)
        {
            int len = payload.Length;
            var lenBytes = new byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(lenBytes.AsSpan(), len);
            await stream.WriteAsync(lenBytes, ct);
            await stream.WriteAsync(payload, ct);
            await stream.FlushAsync(ct);
        }

        public static async Task<byte[]?> ReadFrameAsync(NetworkStream stream, CancellationToken ct)
        {
            var lenBuf = new byte[4];
            if (!await ReadExactAsync(stream, lenBuf, ct))
                return null;

            int len = BinaryPrimitives.ReadInt32LittleEndian(lenBuf.AsSpan());
            if (len <= 0 || len > MaxFrameBytes)
                throw new InvalidDataException($"Invalid frame length: {len}");

            var payload = new byte[len];
            if (!await ReadExactAsync(stream, payload, ct))
                return null;

            return payload;
        }

        private static async Task<bool> ReadExactAsync(NetworkStream stream, byte[] buffer, CancellationToken ct)
        {
            int readTotal = 0;
            while (readTotal < buffer.Length)
            {
                int n = await stream.ReadAsync(buffer.AsMemory(readTotal, buffer.Length - readTotal), ct);
                if (n == 0) return false;
                readTotal += n;
            }
            return true;
        }
    }
}
