using ML.Host.Networking;
using ML.Shared.Protocol;

namespace ML.Host.Rpc;

public sealed class EventBus
{
    private readonly object _lock = new();
    private readonly List<ClientConnection> _clients = new();

    public void AddClient(ClientConnection conn)
    {
        lock (_lock)
            _clients.Add(conn);
    }

    public void RemoveClient(ClientConnection conn)
    {
        lock (_lock)
            _clients.Remove(conn);
    }

    public async Task BroadcastAsync(RpcEvent evt, CancellationToken ct = default)
    {
        ClientConnection[] clients;
        lock (_lock)
            clients = _clients.ToArray();

        foreach (var conn in clients)
        {
            try
            {
                await conn.SendEventAsync(evt, ct);
            }
            catch
            {
            }
        }
    }
}
