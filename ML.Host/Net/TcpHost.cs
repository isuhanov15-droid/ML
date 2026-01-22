using System.Net;
using System.Net.Sockets;

namespace ML.Host.Net;

public sealed class TcpHost : IAsyncDisposable
{
    private readonly TcpListener _listener;
    private readonly List<ClientConnection> _clients = new();
    private readonly CancellationTokenSource _cts = new();
    private Task? _acceptTask;
    private readonly object _lock = new();

    public event Action<ClientConnection>? ClientConnected;
    public event Action<ClientConnection>? ClientDisconnected;

    public TcpHost(IPAddress address, int port)
    {
        _listener = new TcpListener(address, port);
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
            conn.Disconnected += OnDisconnected;

            lock (_lock)
                _clients.Add(conn);

            ClientConnected?.Invoke(conn);
            conn.Start();
        }
    }

    private void OnDisconnected(ClientConnection conn)
    {
        lock (_lock)
            _clients.Remove(conn);

        ClientDisconnected?.Invoke(conn);
    }

    public IReadOnlyList<ClientConnection> Clients
    {
        get
        {
            lock (_lock)
                return _clients.ToArray();
        }
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
