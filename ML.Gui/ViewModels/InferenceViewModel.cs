using System;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Threading;
using ML.Gui.Services;
using ML.Gui.Utils;

namespace ML.Gui.ViewModels;

public sealed class InferenceViewModel : ViewModelBase
{
    private readonly ConnectionService _connection;
    private readonly InferenceRemoteService _inference;
    private readonly SettingsService _settingsService;
    private readonly SettingsData _settings;
    private DispatcherTimer? _timer;
    private EventHandler? _tickHandler;
    private bool _inferenceRunning;

    private string _modelPath = "";
    private string _inputText = "0,0";
    private string _outputText = "";
    private string _latencyText = "—";
    private string _status = "Готов к инференсу";
    private bool _isMonitoring;
    private bool _isLoaded;

    public InferenceViewModel(ConnectionService connection, InferenceRemoteService inference, SettingsService settingsService)
    {
        _connection = connection ?? throw new ArgumentNullException(nameof(connection));
        _inference = inference ?? throw new ArgumentNullException(nameof(inference));
        _settingsService = settingsService ?? throw new ArgumentNullException(nameof(settingsService));
        _settings = _settingsService.Load();

        _modelPath = _settings.LastLoadPath ?? "";

        LoadModelCommand = new AsyncRelayCommand(LoadModelAsync);
        RunOnceCommand = new AsyncRelayCommand(RunOnceAsync, () => _isLoaded && !_isMonitoring);
        StartMonitorCommand = new RelayCommand(StartMonitor, () => _isLoaded && !_isMonitoring);
        StopMonitorCommand = new RelayCommand(StopMonitor, () => _isMonitoring);
    }

    public string ModelPath
    {
        get => _modelPath;
        set => SetField(ref _modelPath, value);
    }

    public string InputText
    {
        get => _inputText;
        set => SetField(ref _inputText, value);
    }

    public string OutputText
    {
        get => _outputText;
        private set => SetField(ref _outputText, value);
    }

    public string LatencyText
    {
        get => _latencyText;
        private set => SetField(ref _latencyText, value);
    }

    public string Status
    {
        get => _status;
        private set => SetField(ref _status, value);
    }

    public bool IsMonitoring
    {
        get => _isMonitoring;
        private set
        {
            if (SetField(ref _isMonitoring, value))
            {
                StartMonitorCommand.RaiseCanExecuteChanged();
                StopMonitorCommand.RaiseCanExecuteChanged();
                RunOnceCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public AsyncRelayCommand LoadModelCommand { get; }
    public AsyncRelayCommand RunOnceCommand { get; }
    public RelayCommand StartMonitorCommand { get; }
    public RelayCommand StopMonitorCommand { get; }

    private async Task LoadModelAsync()
    {
        if (!await EnsureConnectedAsync())
        {
            Status = "Нет подключения к ML.Host.";
            return;
        }

        try
        {
            var result = await _inference.LoadAsync(null, string.IsNullOrWhiteSpace(ModelPath) ? null : ModelPath);
            if (result.Ok)
            {
                _isLoaded = true;
                Status = "Модель загружена.";
                if (!string.IsNullOrWhiteSpace(ModelPath))
                {
                    _settings.LastLoadPath = ModelPath;
                    _settingsService.Save(_settings);
                }
            }
            else
            {
                Status = $"Ошибка загрузки: {result.Message}";
                _isLoaded = false;
            }
        }
        catch (Exception ex)
        {
            Status = $"Ошибка загрузки: {ex.Message}";
            _isLoaded = false;
        }
        finally
        {
            StartMonitorCommand.RaiseCanExecuteChanged();
            RunOnceCommand.RaiseCanExecuteChanged();
        }
    }

    private Task RunOnceAsync() => RunInferenceAsync();

    private void StartMonitor()
    {
        if (!_isLoaded)
        {
            Status = "Сначала загрузите модель.";
            return;
        }

        StopMonitor();
        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _tickHandler = async (_, _) => await RunInferenceAsync();
        _timer.Tick += _tickHandler;
        _timer.Start();
        IsMonitoring = true;
        Status = "Мониторинг запущен.";
    }

    private void StopMonitor()
    {
        if (_timer != null)
        {
            _timer.Stop();
            if (_tickHandler != null)
                _timer.Tick -= _tickHandler;
            _timer = null;
            _tickHandler = null;
        }

        if (IsMonitoring)
            Status = "Мониторинг остановлен.";
        IsMonitoring = false;
    }

    private async Task RunInferenceAsync()
    {
        if (_inferenceRunning) return;
        _inferenceRunning = true;

        double[]? input = ParseInput(InputText);
        if (input == null)
        {
            Status = "Некорректный ввод (ожидаются числа через запятую или пробел).";
            _inferenceRunning = false;
            return;
        }

        try
        {
            if (!await EnsureConnectedAsync())
            {
                Status = "Нет подключения к ML.Host.";
                return;
            }

            var result = await _inference.PredictAsync(input);
            if (!result.Ok)
            {
                Status = $"Ошибка инференса: {result.Message}";
                _inferenceRunning = false;
                return;
            }

            Dispatcher.UIThread.Post(() =>
            {
                OutputText = string.Join(", ", result.Output.Select(v => v.ToString("F4", CultureInfo.InvariantCulture)));
                LatencyText = result.LatencyMs.HasValue ? $"{result.LatencyMs.Value} мс" : "—";
                Status = "Инференс обновлён.";
            });
        }
        catch (Exception ex)
        {
            Status = $"Ошибка инференса: {ex.Message}";
            StopMonitor();
        }
        finally
        {
            _inferenceRunning = false;
        }
    }

    private static double[]? ParseInput(string input)
    {
        if (string.IsNullOrWhiteSpace(input)) return null;
        var tokens = input.Split(new[] { ',', ';', ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        var values = new double[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            if (!double.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                return null;
            values[i] = v;
        }

        return values.Length == 0 ? null : values;
    }

    private async Task<bool> EnsureConnectedAsync()
    {
        if (_connection.IsConnected)
            return true;

        await _connection.ConnectAsync(_connection.LastHost, _connection.LastPort);
        return _connection.IsConnected;
    }
}
