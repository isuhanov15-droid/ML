using System.Buffers.Binary;
using System.Text;

namespace ML.Shared.Protocol;

public static class LengthPrefixedStream
{
    private const int MaxFrameBytes = 16 * 1024 * 1024;

    public static async Task WriteAsync(Stream stream, string json, CancellationToken ct)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));
        if (json == null) throw new ArgumentNullException(nameof(json));

        byte[] payload = Encoding.UTF8.GetBytes(json);
        int len = payload.Length;
        if (len <= 0 || len > MaxFrameBytes)
            throw new InvalidDataException($"Invalid payload length: {len}");

        var lenBytes = new byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(lenBytes.AsSpan(), len);
        await stream.WriteAsync(lenBytes, ct);
        await stream.WriteAsync(payload, ct);
        await stream.FlushAsync(ct);
    }

    public static async Task<string?> ReadAsync(Stream stream, CancellationToken ct)
    {
        if (stream == null) throw new ArgumentNullException(nameof(stream));

        var lenBuf = new byte[4];
        if (!await ReadExactAsync(stream, lenBuf, ct))
            return null;

        int len = BinaryPrimitives.ReadInt32LittleEndian(lenBuf.AsSpan());
        if (len <= 0 || len > MaxFrameBytes)
            throw new InvalidDataException($"Invalid frame length: {len}");

        var payload = new byte[len];
        if (!await ReadExactAsync(stream, payload, ct))
            return null;

        return Encoding.UTF8.GetString(payload);
    }

    private static async Task<bool> ReadExactAsync(Stream stream, byte[] buffer, CancellationToken ct)
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
