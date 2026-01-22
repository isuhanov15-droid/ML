using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Avalonia.Threading;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.SkiaSharpView;
using ML.Core;
using ML.Core.Examples;
using ML.Core.Path;
using ML.Gui.Models;
using ML.Gui.Services;
using ML.Gui.Utils;

namespace ML.Gui.ViewModels;

public sealed class TrainingViewModel : ViewModelBase, IDisposable
{
    private readonly ConnectionService _connection;
    private readonly TrainingRemoteService _training;
    private readonly ModelStoreRemoteService _modelStore;
    private readonly SettingsService _settingsService;
    private readonly SettingsData _settings;

    private TrainingState _state = TrainingState.Idle;
    private string _status = "Готов";
    private string _datasetPath = "";
    private int _epochs = 50;
    private int _batchSize = 4;
    private int _accumulationSteps = 1;
    private bool _shuffle = true;
    private bool _dropLast;
    private double? _gradClipNorm;
    private int? _seed = 42;

    private int _inputSize = 2;
    private int _outputSize = 2;
    private string _hiddenSizes = "4";
    private ActivationType _activation = ActivationType.ReLu;
    private double _learningRate = 0.01;
    private string _modelName = "demo_model";
    private string _savePath = "";
    private string _loadPath = "";
    private string _configSavePath = "";
    private string _configLoadPath = "";
    private string _metricsExportPath = "";
    private int _epochDisplayEvery = 1;
    private ExamplePreset? _selectedPreset;

    private readonly ObservableCollection<ObservablePoint> _trainLossPoints = new();
    private readonly ObservableCollection<ObservablePoint> _valLossPoints = new();
    private readonly ObservableCollection<ObservablePoint> _accPoints = new();
    private LineSeries<ObservablePoint>? _trainSeries;
    private LineSeries<ObservablePoint>? _valSeries;
    private LineSeries<ObservablePoint>? _accSeries;
    private int? _lastTrainEpoch;
    private int? _lastValEpoch;
    private int? _lastAccEpoch;
    private const int MaxEpochRows = 500;
    private int _maxChartPoints = 500;
    private const int UiPumpIntervalMs = 100;
    private readonly Stopwatch _stopwatch = new();
    private DispatcherTimer? _elapsedTimer;
    private DispatcherTimer? _uiPumpTimer;
    private string _elapsedText = "00:00:00";
    private bool _autoExportArtifacts = true;
    private readonly ConcurrentQueue<MetricPoint> _metricQueue = new();
    private readonly ConcurrentQueue<LogLineVm> _logQueue = new();
    private readonly List<TrainingMetricsSnapshot> _allSnapshots = new();
    private readonly Dictionary<int, int> _epochIndex = new();
    private readonly List<LogLineVm> _logWindow = new();
    private const int MaxDrainPerTick = 5000;
    private const int DefaultMaxLogLines = 20000;
    private bool _chartCappedLogged;
    private double? _lastAccuracy;
    private int _updateUiEveryNEpochs = 10;
    private int _maxLogLines = DefaultMaxLogLines;
    private bool _isAutoScrollEnabled = true;
    private int _lastStopEpoch;
    private string _stopReason = "";
    private bool _hasCheckpoint;
    private int _epochCurrent;
    private int _epochStart;
    private int _runPlannedEpochs;

    public TrainingViewModel(
        ConnectionService connection,
        TrainingRemoteService training,
        ModelStoreRemoteService modelStore,
        SettingsService settingsService)
    {
        _connection = connection ?? throw new ArgumentNullException(nameof(connection));
        _training = training ?? throw new ArgumentNullException(nameof(training));
        _modelStore = modelStore ?? throw new ArgumentNullException(nameof(modelStore));
        _settingsService = settingsService ?? throw new ArgumentNullException(nameof(settingsService));
        _settings = _settingsService.Load();

        StartCommand = new AsyncRelayCommand(StartAsync, () => CanStart || CanResume || CanStartNew);
        StopCommand = new RelayCommand(Stop, () => CanStop);
        SaveModelCommand = new AsyncRelayCommand(SaveModelAsync, () => !IsBusy);
        LoadModelCommand = new AsyncRelayCommand(LoadModelAsync, () => !IsBusy);
        NewStartCommand = new AsyncRelayCommand(StartFreshAsync, () => CanStartNew);
        ApplyPresetCommand = new RelayCommand(() => { if (SelectedPreset != null) ApplyPreset(SelectedPreset); });

        _trainSeries = new LineSeries<ObservablePoint>
        {
            Name = "Train loss",
            Values = _trainLossPoints,
            GeometrySize = 0,
            GeometryStroke = null,
            GeometryFill = null,
            Fill = null
        };
        _valSeries = new LineSeries<ObservablePoint>
        {
            Name = "Val loss",
            Values = _valLossPoints,
            GeometrySize = 0,
            GeometryStroke = null,
            GeometryFill = null,
            Fill = null
        };
        _accSeries = new LineSeries<ObservablePoint>
        {
            Name = "Accuracy",
            Values = _accPoints,
            GeometrySize = 0,
            GeometryStroke = null,
            GeometryFill = null,
            Fill = null,
            IsVisible = false
        };

        Series = new ISeries[] { _trainSeries, _valSeries, _accSeries };

        _training.MetricsReceived += OnMetric;
        _training.LogReceived += EnqueueLog;
        _training.TrainStateChanged += OnRemoteStateChanged;
        _training.RunIdChanged += _ =>
        {
            OnPropertyChanged(nameof(CanResume));
            OnPropertyChanged(nameof(StartButtonText));
            StartCommand.RaiseCanExecuteChanged();
        };

        _connection.ConnectionChanged += _ =>
        {
            OnPropertyChanged(nameof(CanStart));
            OnPropertyChanged(nameof(CanStartNew));
            OnPropertyChanged(nameof(CanResume));
            OnPropertyChanged(nameof(CanStop));
            StartCommand.RaiseCanExecuteChanged();
            StopCommand.RaiseCanExecuteChanged();
            NewStartCommand.RaiseCanExecuteChanged();
        };

        SavePath = _settings.LastSavePath ?? "";
        LoadPath = _settings.LastLoadPath ?? "";

        // default preset
        SelectedPreset = Presets.FirstOrDefault();
        if (!string.IsNullOrWhiteSpace(_settings.LastPreset))
        {
            var preset = Presets.FirstOrDefault(p => p.Name == _settings.LastPreset);
            if (preset != null)
                SelectedPreset = preset;
        }
    }

    public ObservableCollection<EpochViewModel> Epochs { get; } = new();
    public ObservableCollection<LogLineVm> LogLines { get; } = new();
    public ObservableCollection<ObservablePoint> TrainLossPoints => _trainLossPoints;
    public ObservableCollection<ObservablePoint> ValLossPoints => _valLossPoints;
    public ObservableCollection<ObservablePoint> AccuracyPoints => _accPoints;
    public bool AutoExportArtifacts
    {
        get => _autoExportArtifacts;
        set => SetField(ref _autoExportArtifacts, value);
    }

    public bool IsAutoScrollEnabled
    {
        get => _isAutoScrollEnabled;
        set => SetField(ref _isAutoScrollEnabled, value);
    }
    public IReadOnlyList<ExamplePreset> Presets { get; } = ExampleRegistry.Presets;
    public ExamplePreset? SelectedPreset
    {
        get => _selectedPreset;
        set
        {
            if (SetField(ref _selectedPreset, value) && value != null)
            {
                ApplyPreset(value);
                _settings.LastPreset = value.Name;
                _settingsService.Save(_settings);
            }
        }
    }

    public bool RequiresDatasetPath => SelectedPreset?.RequiresDatasetPath == true;

    public TrainingState State
    {
        get => _state;
        private set
        {
            if (SetField(ref _state, value))
            {
                OnPropertyChanged(nameof(IsBusy));
                OnPropertyChanged(nameof(IsNotBusy));
                OnPropertyChanged(nameof(CanEdit));
                OnPropertyChanged(nameof(CanStart));
                OnPropertyChanged(nameof(CanStartNew));
                OnPropertyChanged(nameof(CanResume));
                OnPropertyChanged(nameof(CanStop));
                OnPropertyChanged(nameof(StartButtonText));
                StartCommand.RaiseCanExecuteChanged();
                StopCommand.RaiseCanExecuteChanged();
                SaveModelCommand.RaiseCanExecuteChanged();
                LoadModelCommand.RaiseCanExecuteChanged();
                NewStartCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public bool IsBusy => State is TrainingState.Running or TrainingState.Stopping;
    public bool IsNotBusy => !IsBusy;
    public bool CanEdit => State is TrainingState.Idle or TrainingState.Finished or TrainingState.Error or TrainingState.Stopped;
    public bool CanStart => _connection.IsConnected && (State is TrainingState.Idle or TrainingState.Finished or TrainingState.Error);
    public bool CanStartNew => _connection.IsConnected && (CanStart || State is TrainingState.Stopped);
    public bool CanResume => _connection.IsConnected && State is TrainingState.Stopped && HasCheckpointInMemory;
    public bool CanStop => _connection.IsConnected && State is TrainingState.Running or TrainingState.Stopping;

    public string Status
    {
        get => _status;
        private set => SetField(ref _status, value);
    }

    public string ElapsedText
    {
        get => _elapsedText;
        private set => SetField(ref _elapsedText, value);
    }

    public string DatasetPath
    {
        get => _datasetPath;
        set => SetField(ref _datasetPath, value);
    }

    public int EpochsPlanned
    {
        get => _epochs;
        set => EpochsCount = value;
    }

    public int EpochCurrent
    {
        get => _epochCurrent;
        private set => SetField(ref _epochCurrent, value);
    }

    public int EpochStart
    {
        get => _epochStart;
        private set => SetField(ref _epochStart, value);
    }

    public int LastStopEpoch
    {
        get => _lastStopEpoch;
        private set => SetField(ref _lastStopEpoch, value);
    }

    public string StopReason
    {
        get => _stopReason;
        private set => SetField(ref _stopReason, value);
    }

    public bool HasCheckpointInMemory
    {
        get => _hasCheckpoint;
        private set
        {
            if (SetField(ref _hasCheckpoint, value))
            {
                OnPropertyChanged(nameof(CanResume));
                OnPropertyChanged(nameof(StartButtonText));
                StartCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public string StartButtonText => CanResume ? "Продолжить" : "Старт";

    public int EpochsCount
    {
        get => _epochs;
        set => SetField(ref _epochs, value);
    }

    public int BatchSize
    {
        get => _batchSize;
        set => SetField(ref _batchSize, value);
    }

    public int AccumulationSteps
    {
        get => _accumulationSteps;
        set => SetField(ref _accumulationSteps, value);
    }

    public bool Shuffle
    {
        get => _shuffle;
        set => SetField(ref _shuffle, value);
    }

    public bool DropLast
    {
        get => _dropLast;
        set => SetField(ref _dropLast, value);
    }

    public double? GradClipNorm
    {
        get => _gradClipNorm;
        set => SetField(ref _gradClipNorm, value);
    }

    public int? Seed
    {
        get => _seed;
        set => SetField(ref _seed, value);
    }

    public int InputSize
    {
        get => _inputSize;
        set => SetField(ref _inputSize, value);
    }

    public int OutputSize
    {
        get => _outputSize;
        set => SetField(ref _outputSize, value);
    }

    public string HiddenSizes
    {
        get => _hiddenSizes;
        set => SetField(ref _hiddenSizes, value);
    }

    public ActivationType Activation
    {
        get => _activation;
        set => SetField(ref _activation, value);
    }

    public ActivationType[] ActivationOptions { get; } = Enum.GetValues<ActivationType>();

    public double LearningRate
    {
        get => _learningRate;
        set => SetField(ref _learningRate, value);
    }

    public string ModelName
    {
        get => _modelName;
        set => SetField(ref _modelName, value);
    }

    public string SavePath
    {
        get => _savePath;
        set
        {
            if (SetField(ref _savePath, value))
            {
                _settings.LastSavePath = value;
                _settingsService.Save(_settings);
            }
        }
    }

    public string LoadPath
    {
        get => _loadPath;
        set
        {
            if (SetField(ref _loadPath, value))
            {
                _settings.LastLoadPath = value;
                _settingsService.Save(_settings);
            }
        }
    }

    public void SetSavePath(string path) => SavePath = path;
    public void SetLoadPath(string path) => LoadPath = path;

    public string ConfigSavePath
    {
        get => _configSavePath;
        set => SetField(ref _configSavePath, value);
    }

    public string ConfigLoadPath
    {
        get => _configLoadPath;
        set => SetField(ref _configLoadPath, value);
    }

    public string MetricsExportPath
    {
        get => _metricsExportPath;
        set => SetField(ref _metricsExportPath, value);
    }

    public int EpochDisplayEvery
    {
        get => _epochDisplayEvery;
        set
        {
            var v = value <= 0 ? 1 : value;
            if (SetField(ref _epochDisplayEvery, v))
            {
                if (_updateUiEveryNEpochs != v)
                {
                    _updateUiEveryNEpochs = v;
                    OnPropertyChanged(nameof(UpdateUiEveryNEpochs));
                }
            }
        }
    }

    public int UpdateUiEveryNEpochs
    {
        get => _updateUiEveryNEpochs;
        set
        {
            var v = value <= 0 ? 1 : value;
            if (SetField(ref _updateUiEveryNEpochs, v))
            {
                if (_epochDisplayEvery != v)
                {
                    _epochDisplayEvery = v;
                    OnPropertyChanged(nameof(EpochDisplayEvery));
                }
            }
        }
    }

    public AsyncRelayCommand StartCommand { get; }
    public RelayCommand StopCommand { get; }
    public AsyncRelayCommand SaveModelCommand { get; }
    public AsyncRelayCommand LoadModelCommand { get; }
    public AsyncRelayCommand NewStartCommand { get; }
    public RelayCommand ApplyPresetCommand { get; }

    public ISeries[] Series { get; }

    private void ApplyPreset(ExamplePreset preset)
    {
        InputSize = preset.InputSize;
        OutputSize = preset.OutputSize;
        HiddenSizes = preset.HiddenSizes;
        Activation = preset.Activation;
        LearningRate = preset.LearningRate;
        EpochsCount = preset.Epochs;
        BatchSize = preset.BatchSize;
        AccumulationSteps = preset.Accumulation;
        Shuffle = preset.Shuffle;
        DropLast = preset.DropLast;
        Seed = Seed; // keep user seed
    }

    public TrainStartConfig BuildStartConfig(bool resume, int plannedEpochs)
    {
        var hidden = ParseHiddenSizes();
        var (presetKey, datasetPath) = ResolvePreset();

        return new TrainStartConfig
        {
            Resume = resume,
            Network = new TrainNetworkConfig
            {
                InputSize = InputSize,
                OutputSize = OutputSize,
                Hidden = hidden,
                Activation = Activation.ToString(),
                Seed = Seed
            },
            Train = new TrainTrainConfig
            {
                Epochs = plannedEpochs,
                BatchSize = BatchSize,
                Shuffle = Shuffle,
                DropLast = DropLast,
                GradClipNorm = GradClipNorm,
                AccumulationSteps = AccumulationSteps,
                LearningRate = LearningRate,
                UiEveryNEpochs = EpochDisplayEvery
            },
            Data = new TrainDataConfig
            {
                Preset = presetKey,
                DatasetPath = datasetPath
            }
        };
    }

    private int[] ParseHiddenSizes()
    {
        if (string.IsNullOrWhiteSpace(HiddenSizes))
            return Array.Empty<int>();

        var sizes = HiddenSizes.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(s => int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : 0)
            .Where(v => v > 0)
            .ToArray();

        return sizes.Length == 0 ? new[] { 4 } : sizes;
    }

    private (string presetKey, string? datasetPath) ResolvePreset()
    {
        if (RequiresDatasetPath)
            return ("FILE", DatasetPath);

        var name = SelectedPreset?.Name ?? "";
        if (name.Contains("XOR", StringComparison.OrdinalIgnoreCase))
            return ("XOR", null);
        if (name.Contains("AND", StringComparison.OrdinalIgnoreCase))
            return ("AND", null);

        return (string.Empty, null);
    }

    private Task StartAsync() => StartInternalAsync(resume: CanResume);

    private Task StartFreshAsync()
    {
        ResetProgress();
        HasCheckpointInMemory = false;
        EpochStart = 0;
        return StartInternalAsync(resume: false);
    }

    private async Task StartInternalAsync(bool resume)
    {
        if ((!CanStart && !resume && !CanStartNew) || IsBusy)
        {
            Status = "Уже идёт обучение.";
            return;
        }

        if (!await EnsureConnectedAsync())
        {
            Status = "Нет подключения к ML.Host.";
            return;
        }

        if (EpochsCount <= 0 || BatchSize <= 0 || AccumulationSteps <= 0 || InputSize <= 0 || OutputSize <= 0)
        {
            State = TrainingState.Error;
            Status = "Проверьте размеры и Epochs/BatchSize/AccumSteps (>0).";
            return;
        }

        if (RequiresDatasetPath && string.IsNullOrWhiteSpace(DatasetPath))
        {
            State = TrainingState.Error;
            Status = "Укажите путь к датасету (CSV/JSON).";
            return;
        }

        if (!resume)
            ResetProgress();

        Status = resume ? "Возобновление..." : "Запуск...";
        State = TrainingState.Running;
        StartElapsedTimer();
        StartUiPump();

        EpochStart = resume ? LastStopEpoch : 0;
        _runPlannedEpochs = resume && LastStopEpoch > 0 ? Math.Max(1, EpochsCount - LastStopEpoch) : EpochsCount;
        _updateUiEveryNEpochs = _runPlannedEpochs < EpochDisplayEvery ? 1 : EpochDisplayEvery;
        var expectedPoints = Math.Max(1, _runPlannedEpochs / Math.Max(1, _updateUiEveryNEpochs));
        _maxChartPoints = Math.Max(5000, expectedPoints + 2);
        _maxLogLines = Math.Max(DefaultMaxLogLines, expectedPoints + 10);

        var (presetKey, _) = ResolvePreset();
        if (string.IsNullOrWhiteSpace(presetKey))
        {
            State = TrainingState.Error;
            Status = "Выбранный пресет не поддерживается ML.Host.";
            StopElapsedTimer();
            StopUiPump();
            return;
        }

        try
        {
            var config = BuildStartConfig(resume, _runPlannedEpochs);
            AddLogLine($"Train started. epochs={config.Train.Epochs}, uiEvery={config.Train.UiEveryNEpochs}");
            await _training.StartAsync(config);
        }
        catch (Exception ex)
        {
            StopElapsedTimer();
            StopUiPump();
            Dispatcher.UIThread.Post(() =>
            {
                Status = $"Запуск не удался: {ex.GetBaseException().Message}";
                State = TrainingState.Error;
                AddLogLine($"Error: {ex.GetBaseException().Message}");
            });
        }
    }

    private void Stop()
    {
        if (!CanStop) return;
        Status = "Остановка...";
        State = TrainingState.Stopping;
        StopReason = "Остановлено пользователем";
        _ = _training.StopAsync();
    }

    private async Task SaveModelAsync()
    {
        try
        {
            if (!await EnsureConnectedAsync())
            {
                Status = "Нет подключения к ML.Host.";
                return;
            }

            var result = await _modelStore.SaveAsync(ModelName, string.IsNullOrWhiteSpace(SavePath) ? null : SavePath);
            Status = result.Ok
                ? $"Сохранено: {(string.IsNullOrWhiteSpace(SavePath) ? ModelName : SavePath)}"
                : $"Сохранение не удалось: {result.Message}";
        }
        catch (Exception ex)
        {
            Status = $"Сохранение не удалось: {ex.Message}";
        }
    }

    private async Task LoadModelAsync()
    {
        try
        {
            if (!await EnsureConnectedAsync())
            {
                Status = "Нет подключения к ML.Host.";
                return;
            }

            var result = await _modelStore.LoadAsync(ModelName, string.IsNullOrWhiteSpace(LoadPath) ? null : LoadPath);
            Status = result.Ok
                ? $"Загружена модель: {(string.IsNullOrWhiteSpace(LoadPath) ? ModelName : LoadPath)}"
                : $"Загрузка не удалась: {result.Message}";
        }
        catch (Exception ex)
        {
            Status = $"Загрузка не удалась: {ex.Message}";
        }
    }

    private void OnRemoteStateChanged(TrainingState state, string? message)
    {
        Dispatcher.UIThread.Post(() =>
        {
            State = state;
            if (!string.IsNullOrWhiteSpace(message))
                Status = message;

            switch (state)
            {
                case TrainingState.Finished:
                    Status = "Обучение завершено.";
                    StopElapsedTimer();
                    StopUiPump();
                    HasCheckpointInMemory = true;
                    LastStopEpoch = EpochCurrent;
                    AddLogLine($"Finished at epoch {EpochCurrent}");
                    PumpUi(maxDrain: int.MaxValue, force: true);
                    TryAutoExportArtifacts();
                    break;
                case TrainingState.Stopped:
                    Status = "Остановлено.";
                    StopElapsedTimer();
                    StopUiPump();
                    HasCheckpointInMemory = true;
                    LastStopEpoch = EpochCurrent;
                    AddLogLine($"Stopped at epoch {EpochCurrent}");
                    PumpUi(maxDrain: int.MaxValue, force: true);
                    break;
                case TrainingState.Error:
                    StopElapsedTimer();
                    StopUiPump();
                    if (!string.IsNullOrWhiteSpace(message))
                        AddLogLine($"Error: {message}");
                    PumpUi(maxDrain: int.MaxValue, force: true);
                    break;
            }
        });
    }

    private void OnMetric(MetricPoint point)
    {
        _metricQueue.Enqueue(point);
    }

    private void ResetProgress()
    {
        Epochs.Clear();
        _trainLossPoints.Clear();
        _valLossPoints.Clear();
        _accPoints.Clear();
        _logWindow.Clear();
        LogLines.Clear();
        ElapsedText = "00:00:00";
        _lastAccuracy = null;
        _lastStopEpoch = 0;
        _stopReason = "";
        _epochCurrent = 0;
        IsAutoScrollEnabled = true;
        HasCheckpointInMemory = false;
        _allSnapshots.Clear();
        _epochIndex.Clear();
        _lastTrainEpoch = null;
        _lastValEpoch = null;
        _lastAccEpoch = null;
        _chartCappedLogged = false;
        _runPlannedEpochs = 0;
        while (_metricQueue.TryDequeue(out _)) { }
        while (_logQueue.TryDequeue(out _)) { }
    }

    private void StartElapsedTimer()
    {
        _stopwatch.Restart();
        _elapsedTimer?.Stop();
        _elapsedTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
        _elapsedTimer.Tick += (_, _) =>
        {
            var t = _stopwatch.Elapsed;
            ElapsedText = $"{t:hh\\:mm\\:ss}";
        };
        _elapsedTimer.Start();
    }

    private void StopElapsedTimer()
    {
        _elapsedTimer?.Stop();
        _elapsedTimer = null;
    }

    private void StartUiPump()
    {
        _uiPumpTimer?.Stop();
        _uiPumpTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(UiPumpIntervalMs) };
        _uiPumpTimer.Tick += (_, _) => PumpUi();
        _uiPumpTimer.Start();
    }

    private void StopUiPump()
    {
        _uiPumpTimer?.Stop();
        _uiPumpTimer = null;
    }

    private void PumpUi()
    {
        PumpUi(maxDrain: MaxDrainPerTick, force: false);
    }

    private void PumpUi(int maxDrain, bool force)
    {
        int drained = 0;
        bool metricsUpdated = false;
        while (force || drained < maxDrain)
        {
            if (!_logQueue.TryDequeue(out var logLine))
                break;

            drained++;
            AppendLogLine(logLine);
        }

        drained = 0;
        while (force || drained < maxDrain)
        {
            if (!_metricQueue.TryDequeue(out var point))
                break;

            drained++;
            ApplyMetric(point);
            metricsUpdated = true;
        }

        if (metricsUpdated && _allSnapshots.Count > 0 && State is TrainingState.Running or TrainingState.Stopping)
            Status = $"Обучение: эпоха {_allSnapshots[^1].Epoch}";
    }

    private void EnqueueLog(string message)
    {
        if (string.IsNullOrWhiteSpace(message))
            return;

        if (message.Contains("train.start", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("train.completed", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("train.stopped", StringComparison.OrdinalIgnoreCase) ||
            message.Contains("train.error", StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        _logQueue.Enqueue(CreateLogLine(message, epoch: 0));
    }

    public void SaveConfig(string? path = null)
    {
        try
        {
            var config = CaptureConfig();
            var target = string.IsNullOrWhiteSpace(path)
                ? (string.IsNullOrWhiteSpace(ConfigSavePath) ? GetDefaultConfigPath() : ConfigSavePath)
                : path!;
            EnsureDirectory(target);
            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(target, json);
            Status = $"Конфиг сохранён: {target}";
        }
        catch (Exception ex)
        {
            Status = $"Сохранение конфига не удалось: {ex.Message}";
        }
    }

    public void LoadConfig(string path)
    {
        try
        {
            var target = string.IsNullOrWhiteSpace(path)
                ? (string.IsNullOrWhiteSpace(ConfigLoadPath) ? GetDefaultConfigPath() : ConfigLoadPath)
                : path;

            if (string.IsNullOrWhiteSpace(target) || !File.Exists(target))
                throw new FileNotFoundException("Файл конфига не найден.", target);

            var json = File.ReadAllText(target);
            var cfg = JsonSerializer.Deserialize<ExperimentConfig>(json);
            if (cfg == null) throw new InvalidOperationException("Не удалось прочитать конфиг.");
            ApplyConfig(cfg);
            Status = $"Конфиг загружен: {target}";
        }
        catch (Exception ex)
        {
            Status = $"Загрузка конфига не удалась: {ex.Message}";
        }
    }

    public void ExportMetrics(string? path = null, bool exportJson = false)
    {
        try
        {
            var target = string.IsNullOrWhiteSpace(path)
                ? (string.IsNullOrWhiteSpace(MetricsExportPath) ? GetDefaultMetricsPath() : MetricsExportPath)
                : path!;
            EnsureDirectory(target);

            var sb = new StringBuilder();
            sb.AppendLine("epoch,train_loss,val_loss,accuracy,elapsed_ms");
            var exportSnapshots = _allSnapshots.ToList();

            foreach (var e in exportSnapshots)
            {
                sb.Append(e.Epoch).Append(',')
                  .Append(e.TrainLoss.ToString(CultureInfo.InvariantCulture)).Append(',')
                  .Append((e.ValLoss ?? double.NaN).ToString(CultureInfo.InvariantCulture)).Append(',')
                  .Append((e.Accuracy ?? double.NaN).ToString(CultureInfo.InvariantCulture)).Append(',')
                  .Append(e.ElapsedMs.ToString(CultureInfo.InvariantCulture))
                  .AppendLine();
            }

            File.WriteAllText(target, sb.ToString());

            if (exportJson)
            {
                var jsonTarget = Path.ChangeExtension(target, ".json");
                EnsureDirectory(jsonTarget);
                var json = JsonSerializer.Serialize(exportSnapshots, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(jsonTarget, json);
            }

            Status = $"Метрики сохранены: {target}";
        }
        catch (Exception ex)
        {
            Status = $"Экспорт метрик не удался: {ex.Message}";
        }
    }

    private ExperimentConfig CaptureConfig()
    {
        return new ExperimentConfig
        {
            PresetName = SelectedPreset?.Name,
            DatasetPath = DatasetPath,
            InputSize = InputSize,
            OutputSize = OutputSize,
            HiddenSizes = HiddenSizes,
            Activation = Activation,
            Epochs = EpochsCount,
            BatchSize = BatchSize,
            AccumulationSteps = AccumulationSteps,
            EpochDisplayEvery = EpochDisplayEvery,
            UpdateUiEveryNEpochs = UpdateUiEveryNEpochs,
            Shuffle = Shuffle,
            DropLast = DropLast,
            LearningRate = LearningRate,
            GradClipNorm = GradClipNorm,
            Seed = Seed,
            ModelName = ModelName,
            SavePath = SavePath,
            LoadPath = LoadPath
        };
    }

    private void ApplyConfig(ExperimentConfig cfg)
    {
        if (!string.IsNullOrWhiteSpace(cfg.PresetName))
        {
            var preset = Presets.FirstOrDefault(p => p.Name.Equals(cfg.PresetName, StringComparison.OrdinalIgnoreCase));
            if (preset != null)
                SelectedPreset = preset;
        }

        DatasetPath = cfg.DatasetPath ?? DatasetPath;
        InputSize = cfg.InputSize;
        OutputSize = cfg.OutputSize;
        HiddenSizes = cfg.HiddenSizes;
        Activation = cfg.Activation;
        EpochsCount = cfg.Epochs;
        BatchSize = cfg.BatchSize;
        AccumulationSteps = cfg.AccumulationSteps;
        EpochDisplayEvery = cfg.EpochDisplayEvery;
        UpdateUiEveryNEpochs = cfg.UpdateUiEveryNEpochs;
        Shuffle = cfg.Shuffle;
        DropLast = cfg.DropLast;
        LearningRate = cfg.LearningRate;
        GradClipNorm = cfg.GradClipNorm;
        Seed = cfg.Seed;
        ModelName = cfg.ModelName;
        SavePath = cfg.SavePath;
        LoadPath = cfg.LoadPath;
    }

    private string GetDefaultBasePath()
    {
        var name = string.IsNullOrWhiteSpace(ModelName) ? "model" : ModelName;
        var root = ModelPath.ModelsRoot;
        Directory.CreateDirectory(root);
        return Path.Combine(root, name);
    }

    public string GetDefaultConfigPath() => GetDefaultBasePath() + "_config.json";
    public string GetDefaultMetricsPath() => GetDefaultBasePath() + "_metrics.csv";

    private void TryAutoExportArtifacts()
    {
        if (!AutoExportArtifacts) return;
        try
        {
            SaveConfig();
            ExportMetrics();
        }
        catch (Exception ex)
        {
            AddLogLine($"Автосохранение не удалось: {ex.Message}");
        }
    }

    private static void EnsureDirectory(string path)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);
    }

    public void Dispose()
    {
        StopElapsedTimer();
        StopUiPump();
    }

    private async Task<bool> EnsureConnectedAsync()
    {
        if (_connection.IsConnected)
            return true;

        await _connection.ConnectAsync(_connection.LastHost, _connection.LastPort);
        return _connection.IsConnected;
    }

    public void AddLogLine(string message)
    {
        if (string.IsNullOrWhiteSpace(message))
            return;

        _logQueue.Enqueue(CreateLogLine(message, epoch: 0));
        if (_uiPumpTimer == null)
            Dispatcher.UIThread.Post(() => PumpUi(maxDrain: int.MaxValue, force: true));
    }

    private void ApplyMetric(MetricPoint point)
    {
        if (point.Epoch <= 0)
            return;

        int lastEpoch = _epochStart + _runPlannedEpochs;
        int uiEvery = Math.Max(1, UpdateUiEveryNEpochs);
        bool shouldPlot = point.Epoch == 1 ||
                          (lastEpoch > 0 && point.Epoch == lastEpoch) ||
                          point.Epoch % uiEvery == 0;

        var snapshot = new TrainingMetricsSnapshot(
            point.Epoch,
            point.Loss,
            point.ValLoss,
            point.Accuracy,
            DateTimeOffset.Now,
            point.ElapsedMs ?? 0);

        if (!shouldPlot)
        {
            if (point.Epoch > EpochCurrent)
                EpochCurrent = point.Epoch;
            return;
        }

        if (_epochIndex.TryGetValue(point.Epoch, out var existingIndex))
        {
            _allSnapshots[existingIndex] = snapshot;
        }
        else
        {
            if (_allSnapshots.Count == 0 || point.Epoch > _allSnapshots[^1].Epoch)
            {
                _allSnapshots.Add(snapshot);
                _epochIndex[point.Epoch] = _allSnapshots.Count - 1;
            }
            else
            {
                int insertIndex = FindInsertIndex(point.Epoch);
                _allSnapshots.Insert(insertIndex, snapshot);
                for (int i = insertIndex; i < _allSnapshots.Count; i++)
                    _epochIndex[_allSnapshots[i].Epoch] = i;
            }
        }

        if (point.Accuracy.HasValue)
            _lastAccuracy = point.Accuracy;

        if (point.Epoch > EpochCurrent)
            EpochCurrent = point.Epoch;

        AddOrUpdatePoint(_trainLossPoints, point.Epoch, point.Loss, ref _lastTrainEpoch);
        AddOrUpdatePoint(_valLossPoints, point.Epoch, point.ValLoss ?? double.NaN, ref _lastValEpoch);
        if (point.Accuracy.HasValue)
        {
            AddOrUpdatePoint(_accPoints, point.Epoch, point.Accuracy.Value, ref _lastAccEpoch);
            if (_accSeries != null && !_accSeries.IsVisible)
                _accSeries.IsVisible = true;
        }

        TrimPoints(_trainLossPoints);
        TrimPoints(_valLossPoints);
        TrimPoints(_accPoints);

        AppendLogLine(CreateLogLine(FormatMetricLogLine(snapshot), snapshot.Epoch));
    }

    private int FindInsertIndex(int epoch)
    {
        int lo = 0;
        int hi = _allSnapshots.Count - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            int midEpoch = _allSnapshots[mid].Epoch;
            if (midEpoch == epoch)
                return mid;
            if (midEpoch < epoch)
                lo = mid + 1;
            else
                hi = mid - 1;
        }

        return lo;
    }

    private void AddOrUpdatePoint(ObservableCollection<ObservablePoint> points, int epoch, double value, ref int? lastEpoch)
    {
        if (lastEpoch.HasValue && lastEpoch.Value == epoch && points.Count > 0)
        {
            var last = points[^1];
            points[^1] = new ObservablePoint(last.X, value);
            return;
        }

        if (lastEpoch.HasValue && epoch < lastEpoch.Value)
            return;

        points.Add(new ObservablePoint(epoch, value));
        lastEpoch = epoch;
    }

    private void TrimPoints(ObservableCollection<ObservablePoint> points)
    {
        bool trimmed = false;
        while (points.Count > _maxChartPoints)
        {
            points.RemoveAt(0);
            trimmed = true;
        }

        if (trimmed && !_chartCappedLogged)
        {
            _chartCappedLogged = true;
            AddLogLine($"Chart points capped to MaxChartPoints={_maxChartPoints} (keep last N).");
        }
    }

    private void AppendLogLine(LogLineVm logLine)
    {
        _logWindow.Add(logLine);
        LogLines.Add(logLine);

        if (_logWindow.Count > _maxLogLines)
        {
            int remove = _logWindow.Count - _maxLogLines;
            _logWindow.RemoveRange(0, remove);
            for (int i = 0; i < remove && LogLines.Count > 0; i++)
                LogLines.RemoveAt(0);
        }
    }

    private static LogLineVm CreateLogLine(string message, int epoch)
    {
        var level = LogLevel.Info;
        if (message.StartsWith("Error", StringComparison.OrdinalIgnoreCase))
            level = LogLevel.Error;
        else if (message.StartsWith("Warn", StringComparison.OrdinalIgnoreCase))
            level = LogLevel.Warn;

        return new LogLineVm(epoch, message.Trim(), DateTimeOffset.Now, level);
    }

    private static string FormatMetricLogLine(TrainingMetricsSnapshot snapshot)
    {
        var culture = CultureInfo.CurrentCulture;
        string accText = snapshot.Accuracy.HasValue ? snapshot.Accuracy.Value.ToString("0.0000", culture) : "n/a";
        string trainText = snapshot.TrainLoss.ToString("0.0000", culture);
        string valText = snapshot.ValLoss.HasValue ? snapshot.ValLoss.Value.ToString("0.0000", culture) : "n/a";
        return $"Epoch {snapshot.Epoch}: Acc={accText}, TrainLoss={trainText}, ValLoss={valText}";
    }
}
