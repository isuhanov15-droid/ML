using System.Collections.Concurrent;
using System.Text.Json;
using Avalonia.Threading;

namespace ML.Gui.Net;

public sealed class ClientRouter
{
    private readonly ConcurrentDictionary<string, Action<string?, JsonElement>> _map = new(StringComparer.Ordinal);

    public void On(string type, Action<string?, JsonElement> handler)
    {
        if (string.IsNullOrWhiteSpace(type))
            throw new ArgumentException("Type is required", nameof(type));
        _map[type] = handler ?? throw new ArgumentNullException(nameof(handler));
    }

    public bool Dispatch(string type, string? rid, JsonElement data)
    {
        if (_map.TryGetValue(type, out var handler))
        {
            Dispatcher.UIThread.Post(() => handler(rid, data));
            return true;
        }

        return false;
    }
}
