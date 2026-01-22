using System.Buffers.Binary;
using System.Net.Sockets;

namespace ML.Host.Net;

public static class TcpFraming
{
    public const int MaxFrameBytes = 50_000_000;

    public static async Task WriteFrameAsync(NetworkStream stream, ReadOnlyMemory<byte> payload, CancellationToken ct)
    {
        int len = payload.Length;
        if (len <= 0 || len > MaxFrameBytes)
            throw new InvalidDataException($"Invalid payload length: {len}");

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
