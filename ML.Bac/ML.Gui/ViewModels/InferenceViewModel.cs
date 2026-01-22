using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using ML.Core.Inference;
using ML.Gui.Utils;

namespace ML.Gui.ViewModels;

public sealed class InferenceViewModel : ViewModelBase
{
    private readonly IInferenceSession _session = new InferenceSession();
    private DispatcherTimer? _timer;
    private CancellationTokenSource? _cts;
    private bool _inferenceRunning;

    private string _modelPath = "";
    private string _inputText = "0,0";
    private string _outputText = "";
    private string _latencyText = "—";
    private string _status = "Готов к инференсу";
    private bool _isMonitoring;
    private bool _isLoaded;

    public InferenceViewModel()
    {
        LoadModelCommand = new RelayCommand(LoadModel);
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

    public RelayCommand LoadModelCommand { get; }
    public AsyncRelayCommand RunOnceCommand { get; }
    public RelayCommand StartMonitorCommand { get; }
    public RelayCommand StopMonitorCommand { get; }

    private void LoadModel()
    {
        try
        {
            _session.LoadFromFile(ModelPath);
            _isLoaded = true;
            Status = "Модель загружена.";
            StartMonitorCommand.RaiseCanExecuteChanged();
            RunOnceCommand.RaiseCanExecuteChanged();
        }
        catch (Exception ex)
        {
            Status = $"Ошибка загрузки: {ex.Message}";
            _isLoaded = false;
        }
    }

    private async Task RunOnceAsync()
    {
        await RunInferenceAsync(CancellationToken.None);
    }

    private void StartMonitor()
    {
        if (!_isLoaded)
        {
            Status = "Сначала загрузите модель.";
            return;
        }

        StopMonitor();
        _cts = new CancellationTokenSource();
        _timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
        _timer.Tick += async (_, _) => await RunInferenceAsync(_cts.Token);
        _timer.Start();
        IsMonitoring = true;
        Status = "Мониторинг запущен.";
    }

    private void StopMonitor()
    {
        _timer?.Stop();
        _timer = null;
        _cts?.Cancel();
        _cts = null;
        if (IsMonitoring)
            Status = "Мониторинг остановлен.";
        IsMonitoring = false;
    }

    private async Task RunInferenceAsync(CancellationToken ct)
    {
        if (_inferenceRunning) return;
        _inferenceRunning = true;
        if (ct.IsCancellationRequested) return;

        double[]? input = ParseInput(InputText);
        if (input == null)
        {
            Status = "Некорректный ввод (ожидаются числа через запятую или пробел).";
            _inferenceRunning = false;
            return;
        }

        try
        {
            var sw = Stopwatch.StartNew();
            var output = await Task.Run(() => _session.Predict(input), ct);
            sw.Stop();

            Dispatcher.UIThread.Post(() =>
            {
                if (ct.IsCancellationRequested) return;
                OutputText = string.Join(", ", output.Select(v => v.ToString("F4", CultureInfo.InvariantCulture)));
                LatencyText = $"{sw.ElapsedMilliseconds} мс";
                Status = "Инференс обновлён.";
            });
        }
        catch (OperationCanceledException)
        {
            Status = "Мониторинг остановлен.";
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
}
