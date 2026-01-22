using System.Text.Json;
using ML.Gui.Net;
using Avalonia.Threading;

namespace ML.Gui.Services;

public sealed class ConnectionService : IAsyncDisposable
{
    private readonly CoreTcpClient _client;
    private readonly ClientRouter _router;
    private string _lastHost = "127.0.0.1";
    private int _lastPort = 5001;

    public event Action<bool>? ConnectionChanged;
    public event Action<string>? StatusChanged;
    public event Action<string>? CoreHelloReceived;
    public event Action<DateTimeOffset>? PongReceived;
    public event Action<string>? ErrorReceived;

    public bool IsConnected => _client.IsConnected;
    public string Status { get; private set; } = "Disconnected";
    public string LastHost => _lastHost;
    public int LastPort => _lastPort;

    public ConnectionService(CoreTcpClient client, ClientRouter router)
    {
        _client = client ?? throw new ArgumentNullException(nameof(client));
        _router = router ?? throw new ArgumentNullException(nameof(router));

        _client.MessageReceived += OnMessage;
        _client.Disconnected += OnDisconnected;
        _client.Error += ex => RaiseError(ex.Message);

        _router.On("core.hello", HandleCoreHello);
        _router.On("pong", HandlePong);
        _router.On("error", HandleError);
    }

    public async Task ConnectAsync(string host, int port)
    {
        _lastHost = host;
        _lastPort = port;
        UpdateStatus("Connecting...");
        try
        {
            await _client.ConnectAsync(host, port);
            RaiseConnectionChanged(true);
            UpdateStatus("Connected");
        }
        catch (Exception ex)
        {
            RaiseConnectionChanged(false);
            UpdateStatus("Disconnected");
            RaiseError(ex.Message);
        }
    }

    public async Task DisconnectAsync()
    {
        UpdateStatus("Disconnecting...");
        await _client.DisconnectAsync();
        RaiseConnectionChanged(false);
        UpdateStatus("Disconnected");
    }

    public Task PingAsync(CancellationToken ct = default)
    {
        return _client.SendAsync("ping", null, NewRid(), ct);
    }

    public CoreTcpClient Client => _client;
    public ClientRouter Router => _router;

    private void OnMessage(string type, string? rid, JsonElement data)
    {
        if (!_router.Dispatch(type, rid, data))
        {
            // ignore unknown messages
        }
    }

    private void OnDisconnected()
    {
        RaiseConnectionChanged(false);
        UpdateStatus("Disconnected");
    }

    private void HandleCoreHello(string? rid, JsonElement data)
    {
        if (data.ValueKind == JsonValueKind.Object && data.TryGetProperty("version", out var v))
            CoreHelloReceived?.Invoke(v.GetString() ?? "");
        else
            CoreHelloReceived?.Invoke(string.Empty);
    }

    private void HandlePong(string? rid, JsonElement data)
    {
        if (data.ValueKind == JsonValueKind.Object && data.TryGetProperty("serverTimeUtc", out var t))
        {
            if (t.ValueKind == JsonValueKind.String && DateTimeOffset.TryParse(t.GetString(), out var ts))
                PongReceived?.Invoke(ts);
        }
    }

    private void HandleError(string? rid, JsonElement data)
    {
        var msg = data.ValueKind == JsonValueKind.Object && data.TryGetProperty("message", out var m)
            ? m.GetString()
            : data.ToString();
        if (!string.IsNullOrWhiteSpace(msg))
            RaiseError(msg);
    }

    private void UpdateStatus(string status)
    {
        Dispatcher.UIThread.Post(() =>
        {
            Status = status;
            StatusChanged?.Invoke(status);
        });
    }

    private void RaiseError(string message)
    {
        Dispatcher.UIThread.Post(() => ErrorReceived?.Invoke(message));
    }

    private void RaiseConnectionChanged(bool connected)
    {
        Dispatcher.UIThread.Post(() => ConnectionChanged?.Invoke(connected));
    }

    private static string NewRid() => Guid.NewGuid().ToString("N");

    public ValueTask DisposeAsync() => _client.DisposeAsync();
}
