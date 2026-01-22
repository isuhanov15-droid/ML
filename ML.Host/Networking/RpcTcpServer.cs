using System.Net;
using System.Net.Sockets;
using ML.Host.Rpc;

namespace ML.Host.Networking;

public sealed class RpcTcpServer : IAsyncDisposable
{
    private readonly TcpListener _listener;
    private readonly CancellationTokenSource _cts = new();
    private readonly List<ClientConnection> _clients = new();
    private readonly object _lock = new();
    private Task? _acceptTask;

    public EventBus Events { get; }
    public RpcRouter Router { get; }

    public RpcTcpServer(IPAddress address, int port, RpcRouter router)
    {
        _listener = new TcpListener(address, port);
        Router = router;
        Events = new EventBus();
    }

    public void Start()
    {
        _listener.Start();
        _acceptTask = Task.Run(() => AcceptLoopAsync(_cts.Token), CancellationToken.None);
    }

    private async Task AcceptLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            TcpClient client;
            try
            {
                client = await _listener.AcceptTcpClientAsync(ct);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            var conn = new ClientConnection(client);
            conn.RequestReceived += OnRequest;
            conn.Disconnected += OnDisconnected;

            lock (_lock)
                _clients.Add(conn);

            Events.AddClient(conn);
            conn.Start();
        }
    }

    private void OnDisconnected(ClientConnection conn)
    {
        lock (_lock)
            _clients.Remove(conn);

        Events.RemoveClient(conn);
    }

    private void OnRequest(ClientConnection conn, ML.Shared.Protocol.RpcRequest req)
    {
        _ = Router.HandleAsync(conn, req);
    }

    public async ValueTask DisposeAsync()
    {
        try { _cts.Cancel(); } catch { }
        try { _listener.Stop(); } catch { }

        if (_acceptTask != null)
        {
            try { await _acceptTask; } catch { }
        }

        ClientConnection[] connections;
        lock (_lock)
            connections = _clients.ToArray();

        foreach (var conn in connections)
            await conn.DisposeAsync();

        _cts.Dispose();
    }
}
