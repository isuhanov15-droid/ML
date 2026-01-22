using System.Threading.Tasks;
using ML.Gui.Net;
using ML.Gui.Services;
using ML.Gui.Utils;

namespace ML.Gui.ViewModels;

public sealed class MainViewModel : ViewModelBase, IAsyncDisposable
{
    private readonly CoreTcpClient _client;
    private readonly ClientRouter _router;
    private readonly ConnectionService _connection;
    private readonly TrainingRemoteService _trainingRemote;
    private readonly ModelStoreRemoteService _modelStore;
    private readonly InferenceRemoteService _inferenceRemote;
    private readonly SettingsService _settingsService;
    private readonly SettingsData _settings;
    private string _host = "127.0.0.1";
    private int _port = 5001;
    private string _connectionStatus = "Disconnected";
    private bool _isConnected;

    public MainViewModel()
    {
        _client = new CoreTcpClient();
        _router = new ClientRouter();
        _connection = new ConnectionService(_client, _router);
        _trainingRemote = new TrainingRemoteService(_client, _router);
        _modelStore = new ModelStoreRemoteService(_client, _router);
        _inferenceRemote = new InferenceRemoteService(_client, _router);
        _settingsService = new SettingsService();
        _settings = _settingsService.Load();

        _host = _settings.Host;
        _port = _settings.Port;

        Training = new TrainingViewModel(_connection, _trainingRemote, _modelStore, _settingsService);
        Inference = new InferenceViewModel(_connection, _inferenceRemote, _settingsService);

        ConnectCommand = new AsyncRelayCommand(ConnectAsync, () => !IsConnected);
        DisconnectCommand = new AsyncRelayCommand(DisconnectAsync, () => IsConnected);

        _connection.ConnectionChanged += connected =>
        {
            IsConnected = connected;
            Training.AddLogLine(connected ? "connected" : "disconnected");
        };
        _connection.StatusChanged += status => ConnectionStatus = status;
        _connection.CoreHelloReceived += version => Training.AddLogLine($"core.hello: {version}");
        _connection.PongReceived += ts => Training.AddLogLine($"pong: {ts:HH:mm:ss}");
        _connection.ErrorReceived += msg =>
        {
            Training.AddLogLine($"error: {msg}");
            ConnectionStatus = "Disconnected";
        };

        _ = _connection.ConnectAsync(_host, _port);
    }

    public TrainingViewModel Training { get; }
    public InferenceViewModel Inference { get; }
    public AsyncRelayCommand ConnectCommand { get; }
    public AsyncRelayCommand DisconnectCommand { get; }

    public string Host
    {
        get => _host;
        set
        {
            if (SetField(ref _host, value))
            {
                _settings.Host = value;
                _settingsService.Save(_settings);
            }
        }
    }

    public int Port
    {
        get => _port;
        set
        {
            if (SetField(ref _port, value))
            {
                _settings.Port = value;
                _settingsService.Save(_settings);
            }
        }
    }

    public string ConnectionStatus
    {
        get => _connectionStatus;
        private set => SetField(ref _connectionStatus, value);
    }

    public bool IsConnected
    {
        get => _isConnected;
        private set
        {
            if (SetField(ref _isConnected, value))
            {
                ConnectCommand.RaiseCanExecuteChanged();
                DisconnectCommand.RaiseCanExecuteChanged();
            }
        }
    }

    private async Task ConnectAsync()
    {
        ConnectionStatus = "Connecting...";
        await _connection.ConnectAsync(Host, Port);
        ConnectionStatus = _connection.Status;
    }

    private async Task DisconnectAsync()
    {
        ConnectionStatus = "Disconnecting...";
        await _connection.DisconnectAsync();
        ConnectionStatus = _connection.Status;
    }

    public async ValueTask DisposeAsync()
    {
        await _connection.DisposeAsync();
        Training.Dispose();
    }
}
