using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Net.Sockets;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using Avalonia.Threading;

namespace ML.Gui.Net;

public sealed class TcpTransport : IAsyncDisposable
{
    private TcpClient? _client;
    private NetworkStream? _stream;

    private readonly Channel<byte[]> _outgoing = Channel.CreateUnbounded<byte[]>(new UnboundedChannelOptions
    {
        SingleReader = true,
        SingleWriter = false,
        AllowSynchronousContinuations = false
    });

    private readonly CancellationTokenSource _cts = new();
    private Task? _recvTask;
    private Task? _sendTask;

    public bool IsConnected => _client?.Connected == true;

    public event Action? Connected;
    public event Action? Disconnected;
    public event Action<Exception>? Error;
    public event Action<Envelope>? Message;

    // ---------- Public API ----------

    public async Task ConnectAsync(string host, int port, CancellationToken ct = default)
    {
        if (IsConnected) return;

        _client = new TcpClient();
        _client.NoDelay = true;

        await _client.ConnectAsync(host, port, ct);
        _stream = _client.GetStream();

        _sendTask = Task.Run(() => SendLoopAsync(_cts.Token), CancellationToken.None);
        _recvTask = Task.Run(() => RecvLoopAsync(_cts.Token), CancellationToken.None);

        RaiseUI(() => Connected?.Invoke());
    }

    public async Task DisconnectAsync()
    {
        await DisposeAsync();
    }

    public async Task SendAsync(string type, object? data = null, string? requestId = null, CancellationToken ct = default)
    {
        var env = new OutgoingEnvelope
        {
            Type = type,
            RequestId = requestId,
            Timestamp = DateTimeOffset.UtcNow,
            Data = data
        };

        byte[] json = JsonSerializer.SerializeToUtf8Bytes(env, JsonDefaults.Serialize);
        await _outgoing.Writer.WriteAsync(json, ct);
    }

    // ---------- Loops ----------

    private async Task SendLoopAsync(CancellationToken ct)
    {
        try
        {
            while (await _outgoing.Reader.WaitToReadAsync(ct))
            {
                while (_outgoing.Reader.TryRead(out var payload))
                {
                    var stream = _stream;
                    if (stream is null) return;

                    await TcpFraming.WriteFrameAsync(stream, payload, ct);
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            RaiseUI(() => Error?.Invoke(ex));
        }
    }

    private async Task RecvLoopAsync(CancellationToken ct)
    {
        try
        {
            var stream = _stream;
            if (stream is null) return;

            while (!ct.IsCancellationRequested)
            {
                var payload = await TcpFraming.ReadFrameAsync(stream, ct);
                if (payload is null) break;

                Envelope? env;
                try
                {
                    env = JsonSerializer.Deserialize<Envelope>(payload, JsonDefaults.Deserialize);
                }
                catch (Exception ex)
                {
                    RaiseUI(() => Error?.Invoke(ex));
                    continue;
                }

                if (env is null || string.IsNullOrWhiteSpace(env.Type))
                    continue;

                RaiseUI(() => Message?.Invoke(env));
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            RaiseUI(() => Error?.Invoke(ex));
        }
        finally
        {
            RaiseUI(() => Disconnected?.Invoke());
        }
    }

    // ---------- UI marshal ----------

    private static void RaiseUI(Action action)
    {
        // Avalonia-safe: всегда дергаем UIThread (даже если мы уже на нём)
        Dispatcher.UIThread.Post(action);
    }

    // ---------- Dispose ----------

    public async ValueTask DisposeAsync()
    {
        try { _cts.Cancel(); } catch { }

        _outgoing.Writer.TryComplete();

        try { _stream?.Close(); } catch { }
        try { _client?.Close(); } catch { }

        if (_sendTask is not null)
        {
            try { await _sendTask; } catch { }
        }

        if (_recvTask is not null)
        {
            try { await _recvTask; } catch { }
        }

        _stream = null;
        _client = null;
    }

    // ---------- Envelopes ----------

    public sealed class Envelope
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = "";

        [JsonPropertyName("requestId")]
        public string? RequestId { get; set; }

        [JsonPropertyName("timestamp")]
        public DateTimeOffset Timestamp { get; set; }

        [JsonPropertyName("data")]
        public JsonElement Data { get; set; }
    }

    private sealed class OutgoingEnvelope
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = "";

        [JsonPropertyName("requestId")]
        public string? RequestId { get; set; }

        [JsonPropertyName("timestamp")]
        public DateTimeOffset Timestamp { get; set; }

        [JsonPropertyName("data")]
        public object? Data { get; set; }
    }

    // ---------- Framing ----------

    private static class TcpFraming
    {
        private const int MaxFrameBytes = 50_000_000;

        public static async Task WriteFrameAsync(NetworkStream stream, ReadOnlyMemory<byte> payload, CancellationToken ct)
        {
            int len = payload.Length;
            if (len <= 0 || len > MaxFrameBytes)
                throw new InvalidDataException($"Invalid payload length: {len}");

            byte[] lenBytes = ArrayPool<byte>.Shared.Rent(4);
            try
            {
                BinaryPrimitives.WriteInt32LittleEndian(lenBytes.AsSpan(0, 4), len);
                await stream.WriteAsync(lenBytes.AsMemory(0, 4), ct);
                await stream.WriteAsync(payload, ct);
                await stream.FlushAsync(ct);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(lenBytes);
            }
        }

        public static async Task<byte[]?> ReadFrameAsync(NetworkStream stream, CancellationToken ct)
        {
            byte[] lenBuf = ArrayPool<byte>.Shared.Rent(4);
            try
            {
                if (!await ReadExactAsync(stream, lenBuf.AsMemory(0, 4), ct))
                    return null;

                int len = BinaryPrimitives.ReadInt32LittleEndian(lenBuf.AsSpan(0, 4));
                if (len <= 0 || len > MaxFrameBytes)
                    throw new InvalidDataException($"Invalid frame length: {len}");

                byte[] payload = ArrayPool<byte>.Shared.Rent(len);
                try
                {
                    if (!await ReadExactAsync(stream, payload.AsMemory(0, len), ct))
                        return null;

                    var exact = new byte[len];
                    Buffer.BlockCopy(payload, 0, exact, 0, len);
                    return exact;
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(payload);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(lenBuf);
            }
        }

        private static async Task<bool> ReadExactAsync(NetworkStream stream, Memory<byte> buffer, CancellationToken ct)
        {
            int readTotal = 0;
            while (readTotal < buffer.Length)
            {
                int n = await stream.ReadAsync(buffer.Slice(readTotal), ct);
                if (n == 0) return false;
                readTotal += n;
            }
            return true;
        }
    }

    private static class JsonDefaults
    {
        public static readonly JsonSerializerOptions Serialize = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            WriteIndented = false
        };

        public static readonly JsonSerializerOptions Deserialize = new()
        {
            PropertyNameCaseInsensitive = true
        };
    }
}
