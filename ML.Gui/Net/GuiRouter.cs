using System.Collections.Concurrent;

namespace ML.Gui.Net;

public sealed class GuiRouter
{
    private readonly ConcurrentDictionary<string, Action<TcpTransport.Envelope>> _map = new(StringComparer.Ordinal);

    public void On(string type, Action<TcpTransport.Envelope> handler) => _map[type] = handler;

    public bool TryDispatch(TcpTransport.Envelope env)
    {
        if (_map.TryGetValue(env.Type, out var h))
        {
            h(env);
            return true;
        }
        return false;
    }
}
