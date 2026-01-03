using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using ML.Core;
using ML.Core.Examples;
using ML.Core.Layers;
using ML.Core.Serialization;
using ML.Core.Training;
using ML.Gui.Models;
using ML.Gui.Services;
using ML.Gui.Utils;
using ML.Core.Path;

namespace ML.Gui.ViewModels;

public sealed class TrainingViewModel : ViewModelBase
{
    private readonly TrainingHost _host = new();
    private CancellationTokenSource? _sessionCts;
    private CancellationTokenSource? _accuracyCts;

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

    private readonly ObservableCollection<double> _trainLossValues = new();
    private readonly ObservableCollection<double?> _valLossValues = new();
    private Task? _runningTask;
    private const int MaxEpochRows = 500;
    private const int MaxChartPoints = 500;
    private const int UiPumpIntervalMs = 250;
    private const int AccuracyEveryEpochs = 10;
    private string _logText = "";
    private readonly Stopwatch _stopwatch = new();
    private DispatcherTimer? _elapsedTimer;
    private DispatcherTimer? _uiPumpTimer;
    private string _elapsedText = "00:00:00";
    private bool _autoExportArtifacts = true;
    private readonly object _metricsLock = new();
    private readonly object _logLock = new();
    private readonly List<string> _fullLogs = new();
    private int _lastUiLogIndex;
    private bool _pendingUiLogFlush;
    private bool _pendingUiChartUpdate;
    private bool _forceFinalUiUpdate;
    private int _lastUiChartIndex;
    private readonly List<TrainingMetricsSnapshot> _allSnapshots = new();
    private double? _lastAccuracy;
    private Task? _accuracyTask;
    private int _updateUiEveryNEpochs = 10;
    private const int MaxLogLines = 1000;
    private int _lastStopEpoch;
    private string _stopReason = "";
    private bool _hasCheckpoint;
    private int _epochCurrent;
    private int _epochStart;
    private int _runPlannedEpochs;

    public TrainingViewModel()
    {
        StartCommand = new AsyncRelayCommand(StartAsync, () => CanStart || CanResume || CanStartNew);
        StopCommand = new RelayCommand(Stop, () => CanStop);
        SaveModelCommand = new RelayCommand(SaveModel, () => !IsBusy);
        LoadModelCommand = new RelayCommand(LoadModel, () => !IsBusy);
        NewStartCommand = new AsyncRelayCommand(StartFreshAsync, () => CanStartNew);
        ApplyPresetCommand = new RelayCommand(() => { if (SelectedPreset != null) ApplyPreset(SelectedPreset); });

        Series = new ISeries[]
        {
            new LineSeries<double>
            {
                Name = "Train loss",
                Values = _trainLossValues
            },
            new LineSeries<double?>
            {
                Name = "Val loss",
                Values = _valLossValues
            }
        };

        // default preset
        SelectedPreset = Presets.FirstOrDefault();
    }

    public ObservableCollection<EpochViewModel> Epochs { get; } = new();
    public ObservableCollection<string> Logs { get; } = new();
    public ObservableCollection<double> TrainLossValues => _trainLossValues;
    public ObservableCollection<double?> ValLossValues => _valLossValues;
    public string LogText
    {
        get => _logText;
        private set => SetField(ref _logText, value);
    }
    public bool AutoExportArtifacts
    {
        get => _autoExportArtifacts;
        set => SetField(ref _autoExportArtifacts, value);
    }
    public IReadOnlyList<ExamplePreset> Presets { get; } = ExampleRegistry.Presets;
    public ExamplePreset? SelectedPreset
    {
        get => _selectedPreset;
        set
        {
            if (SetField(ref _selectedPreset, value) && value != null)
                ApplyPreset(value);
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
    public bool CanStart => State is TrainingState.Idle or TrainingState.Finished or TrainingState.Error;
    public bool CanStartNew => CanStart || State is TrainingState.Stopped;
    public bool CanResume => State is TrainingState.Stopped && HasCheckpointInMemory;
    public bool CanStop => State is TrainingState.Running or TrainingState.Stopping;

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
        set => SetField(ref _savePath, value);
    }

    public string LoadPath
    {
        get => _loadPath;
        set => SetField(ref _loadPath, value);
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
        set => SetField(ref _epochDisplayEvery, value <= 0 ? 1 : value);
    }

    public int UpdateUiEveryNEpochs
    {
        get => _updateUiEveryNEpochs;
        set => SetField(ref _updateUiEveryNEpochs, value <= 0 ? 1 : value);
    }

    public AsyncRelayCommand StartCommand { get; }
    public RelayCommand StopCommand { get; }
    public RelayCommand SaveModelCommand { get; }
    public RelayCommand LoadModelCommand { get; }
    public AsyncRelayCommand NewStartCommand { get; }
    public RelayCommand ApplyPresetCommand { get; }

    public ISeries[] Series { get; }

    /// <summary>
    /// Configure training pipeline from the hosting app (model/optimizer/loss/data).
    /// </summary>
    public void Configure(
        Func<Network> networkFactory,
        Func<Network, Trainer> trainerFactory,
        Func<IEnumerable<(double[] x, int y)>> trainProvider,
        Func<IEnumerable<(double[] x, int y)>?>? valProvider = null)
    {
        _host.Configure(networkFactory, trainerFactory, trainProvider, valProvider);
    }

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

        _host.SetDataProviders(
            () => preset.BuildTrain(preset.RequiresDatasetPath ? DatasetPath : null),
            () => preset.BuildVal(preset.RequiresDatasetPath ? DatasetPath : null));
    }

    public Network BuildNetworkFromState()
    {
        var net = new Network();
        var hidden = ParseHiddenSizes();

        int last = InputSize;
        int seed = Seed ?? 123;

        foreach (var h in hidden)
        {
            net.Add(new LinearLayer(last, h, seed++));
            net.Add(new ActivationLayer(h, Activation));
            last = h;
        }

        net.Add(new LinearLayer(last, OutputSize, seed++));
        net.Add(new SoftmaxLayer(OutputSize));

        return net;
    }

    public TrainOptions BuildOptions(int? overrideEpochs = null)
    {
        return new TrainOptions
        {
            Epochs = overrideEpochs ?? EpochsCount,
            BatchSize = BatchSize,
            Shuffle = Shuffle,
            DropLast = DropLast,
            GradClipNorm = GradClipNorm,
            GradientAccumulationSteps = AccumulationSteps,
            Seed = Seed
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

    private Task StartAsync() => StartInternalAsync(resume: CanResume);

    private Task StartFreshAsync()
    {
        ResetProgress();
        HasCheckpointInMemory = false;
        EpochStart = 0;
        return StartInternalAsync(resume: false);
    }

    private Task StartInternalAsync(bool resume)
    {
        if ((!CanStart && !resume && !CanStartNew) || _runningTask is { IsCompleted: false })
        {
            Status = "Уже идёт обучение.";
            return Task.CompletedTask;
        }

        if (!_host.IsConfigured)
        {
            State = TrainingState.Error;
            Status = "Нет конфигурации тренировки: вызовите Configure(...) из кода.";
            return Task.CompletedTask;
        }

        if (EpochsCount <= 0 || BatchSize <= 0 || AccumulationSteps <= 0 || InputSize <= 0 || OutputSize <= 0)
        {
            State = TrainingState.Error;
            Status = "Проверьте размеры и Epochs/BatchSize/AccumSteps (>0).";
            return Task.CompletedTask;
        }

        if (RequiresDatasetPath && string.IsNullOrWhiteSpace(DatasetPath))
        {
            State = TrainingState.Error;
            Status = "Укажите путь к датасету (CSV/JSON).";
            return Task.CompletedTask;
        }

        if (!resume)
            ResetProgress();

        Status = resume ? "Возобновление..." : "Запуск...";
        State = TrainingState.Running;
        StartElapsedTimer();
        StartUiPump();

        _sessionCts = new CancellationTokenSource();
        if (resume)
        {
            _host.PrepareResume();
            EpochStart = LastStopEpoch;
        }
        else
        {
            EpochStart = 0;
        }

        _runPlannedEpochs = resume && LastStopEpoch > 0 ? Math.Max(1, EpochsCount - LastStopEpoch) : EpochsCount;
        if (_runPlannedEpochs < UpdateUiEveryNEpochs)
            _updateUiEveryNEpochs = 1;
        else
            _updateUiEveryNEpochs = UpdateUiEveryNEpochs;

        try
        {
            var effectiveEpochs = _runPlannedEpochs;
            var options = BuildOptions(effectiveEpochs);
            _runningTask = _host.StartAsync(options, OnEpoch, _sessionCts.Token, useCheckpoint: resume);
            Status = "Обучение...";

            _ = _runningTask.ContinueWith(t =>
            {
                Dispatcher.UIThread.Post(() =>
                {
                    if (t.IsCanceled)
                    {
                        Status = $"Остановлено на эпохе {LastStopEpoch}";
                        StopReason = "Остановлено пользователем";
                        State = TrainingState.Stopped;
                        HasCheckpointInMemory = _host.HasCheckpoint;
                        _forceFinalUiUpdate = true;
                    }
                    else if (t.IsFaulted)
                    {
                        Status = $"Ошибка: {t.Exception?.GetBaseException().Message}";
                        StopReason = Status;
                        State = TrainingState.Error;
                        if (t.Exception != null)
                            AddLogLine($"Ошибка обучения: {t.Exception.GetBaseException().Message}");
                        _forceFinalUiUpdate = true;
                    }
                    else
                    {
                        Status = "Обучение завершено";
                        StopReason = "Завершено";
                        State = TrainingState.Finished;
                        LastStopEpoch = EpochCurrent;
                        HasCheckpointInMemory = _host.HasCheckpoint;
                        TryAutoExportArtifacts();
                        _forceFinalUiUpdate = true;
                    }

                    if (_forceFinalUiUpdate)
                        PumpUi();

                    StopElapsedTimer();
                    StopUiPump();
                    CancelAccuracyComputation();
                    _sessionCts?.Dispose();
                    _sessionCts = null;
                    _runningTask = null;
                    StartCommand.RaiseCanExecuteChanged();
                });
            });
        }
        catch (Exception ex)
        {
            StopElapsedTimer();
            StopUiPump();
            CancelAccuracyComputation();
            _sessionCts?.Dispose();
            _sessionCts = null;
            Dispatcher.UIThread.Post(() =>
            {
                Status = $"Запуск не удался: {ex.GetBaseException().Message}";
                State = TrainingState.Error;
                AddLogLine($"Ошибка запуска: {ex.GetBaseException().Message}");
            });
        }

        return Task.CompletedTask;
    }

    private void Stop()
    {
        if (!CanStop) return;
        Status = "Остановка...";
        State = TrainingState.Stopping;
        StopReason = "Остановлено пользователем";
        _host.Stop();
        _sessionCts?.Cancel();
        CancelAccuracyComputation();
    }

    private void SaveModel()
    {
        try
        {
            _host.SaveModel(ModelName, string.IsNullOrWhiteSpace(SavePath) ? null : SavePath);
            Status = $"Сохранено: {(string.IsNullOrWhiteSpace(SavePath) ? ModelName : SavePath)}";
        }
        catch (Exception ex)
        {
            Status = $"Сохранение не удалось: {ex.Message}";
        }
    }

    private void LoadModel()
    {
        try
        {
            _host.LoadModel(ModelName, string.IsNullOrWhiteSpace(LoadPath) ? null : LoadPath);
            ApplyLoadedNetwork();
            Status = $"Загружена модель: {(string.IsNullOrWhiteSpace(LoadPath) ? ModelName : LoadPath)}";
        }
        catch (Exception ex)
        {
            Status = $"Загрузка не удалась: {ex.Message}";
        }
    }

    private void ApplyLoadedNetwork()
    {
        var net = _host.CurrentNetwork;
        if (net == null) return;

        var linears = net.Layers.OfType<LinearLayer>().ToList();
        if (linears.Count > 0)
        {
            InputSize = linears.First().InputSize;
            OutputSize = linears.Last().OutputSize;
            var hidden = linears.Skip(1).Take(linears.Count - 2).Select(l => l.OutputSize).ToArray();
            HiddenSizes = hidden.Length > 0 ? string.Join(",", hidden) : "";
        }

        var firstActivation = net.Layers.OfType<ActivationLayer>().FirstOrDefault();
        if (firstActivation != null)
            Activation = firstActivation.Type;
    }

    private void OnEpoch(TrainEpochResult r)
    {
        EpochCurrent = EpochStart + r.Epoch;
        LastStopEpoch = EpochCurrent;
        var elapsedMs = _stopwatch.ElapsedMilliseconds;
        var snapshot = new TrainingMetricsSnapshot(
            EpochCurrent,
            r.TrainLoss,
            r.ValLoss,
            _lastAccuracy,
            DateTimeOffset.UtcNow,
            elapsedMs);

        lock (_metricsLock)
        {
            _allSnapshots.Add(snapshot);
        }

        bool shouldDisplay = r.Epoch == 1 ||
                             _updateUiEveryNEpochs <= 1 ||
                             (r.Epoch % _updateUiEveryNEpochs == 0) ||
                             r.Epoch == _runPlannedEpochs;
        if (shouldDisplay)
        {
            _pendingUiChartUpdate = true;
        }

        QueueLogLine(r, shouldDisplay);
        TryScheduleAccuracy(r.Epoch);
    }

    private void ResetProgress()
    {
        Epochs.Clear();
        _trainLossValues.Clear();
        _valLossValues.Clear();
        Logs.Clear();
        lock (_logLock)
        {
        }
        LogText = string.Empty;
        ElapsedText = "00:00:00";
        _lastAccuracy = null;
        _fullLogs.Clear();
        _lastStopEpoch = 0;
        _stopReason = "";
        _epochCurrent = 0;
        HasCheckpointInMemory = false;
        _lastUiLogIndex = 0;
        _pendingUiLogFlush = false;
        _pendingUiChartUpdate = false;
        _forceFinalUiUpdate = false;
        _allSnapshots.Clear();
        _lastUiChartIndex = 0;
        _runPlannedEpochs = 0;
    }

    private void AddLogLine(string line)
    {
        if (Logs.Count >= MaxLogLines)
            Logs.RemoveAt(0);
        Logs.Add(line);
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
            List<TrainingMetricsSnapshot> exportSnapshots;
            lock (_metricsLock)
            {
                exportSnapshots = _allSnapshots.ToList();
            }

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
                var jsonTarget = System.IO.Path.ChangeExtension(target, ".json");
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
        return System.IO.Path.Combine(root, name);
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

    private void StartElapsedTimer()
    {
        _stopwatch.Restart();
        _elapsedTimer?.Stop();
        _elapsedTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(300) };
        _elapsedTimer.Tick += (_, _) => UpdateElapsed();
        _elapsedTimer.Start();
        UpdateElapsed();
    }

    private void StopElapsedTimer()
    {
        _elapsedTimer?.Stop();
        _elapsedTimer = null;
        _stopwatch.Stop();
        UpdateElapsed();
    }

    private void UpdateElapsed()
    {
        ElapsedText = _stopwatch.Elapsed.ToString(@"hh\:mm\:ss");
    }

    private void StartUiPump()
    {
        if (_uiPumpTimer != null)
            _uiPumpTimer.Tick -= UiPumpTick;
        _uiPumpTimer?.Stop();
        _uiPumpTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(UiPumpIntervalMs) };
        _uiPumpTimer.Tick += UiPumpTick;
        _uiPumpTimer.Start();
    }

    private void StopUiPump()
    {
        if (_uiPumpTimer != null)
        {
            _uiPumpTimer.Stop();
            _uiPumpTimer.Tick -= UiPumpTick;
            _uiPumpTimer = null;
        }
    }

    private void UiPumpTick(object? sender, EventArgs e) => PumpUi();

    private void PumpUi()
    {
        TrainingMetricsSnapshot? latest = null;
        List<TrainingMetricsSnapshot>? tail = null;

        if (_pendingUiChartUpdate || _forceFinalUiUpdate)
        {
            lock (_metricsLock)
            {
                if (_allSnapshots.Count > 0)
                {
                    int start = Math.Max(0, _allSnapshots.Count - MaxChartPoints);
                    tail = _allSnapshots.Skip(start).ToList();
                    latest = _allSnapshots[^1];
                    _lastUiChartIndex = _allSnapshots.Count;
                }
            }
        }

        if (tail != null && latest.HasValue)
        {
            _pendingUiChartUpdate = false;
            _trainLossValues.Clear();
            _valLossValues.Clear();
            Epochs.Clear();

            foreach (var s in tail)
            {
                _trainLossValues.Add(s.TrainLoss);
                _valLossValues.Add(s.ValLoss);
                Epochs.Add(new EpochViewModel
                {
                    Epoch = s.Epoch,
                    TrainLoss = s.TrainLoss,
                    ValLoss = s.ValLoss,
                    Accuracy = s.Accuracy,
                    ElapsedMs = s.ElapsedMs
                });
            }

            var snapshot = latest.Value;
            if (State is TrainingState.Running or TrainingState.Stopping)
                Status = $"Обучение: эпоха {snapshot.Epoch}";
        }

        FlushLogs(force: _forceFinalUiUpdate);
        if (_forceFinalUiUpdate) _forceFinalUiUpdate = false;
    }

    private void QueueLogLine(TrainEpochResult r, bool flushUi)
    {
        var accText = _lastAccuracy.HasValue ? _lastAccuracy.Value.ToString("F4") : "n/a";
        var line = $"Epoch {EpochStart + r.Epoch}: Acc={accText}, TrainLoss={r.TrainLoss:F4}" + (r.ValLoss.HasValue ? $", ValLoss={r.ValLoss:F4}" : "");
        lock (_logLock)
        {
            _fullLogs.Add(line);
            if (flushUi) _pendingUiLogFlush = true;
        }
    }

    private void FlushLogs(bool force)
    {
        List<string>? toAdd = null;
        lock (_logLock)
        {
            if ((_pendingUiLogFlush || force) && _fullLogs.Count > _lastUiLogIndex)
            {
                toAdd = _fullLogs.Skip(_lastUiLogIndex).ToList();
                _lastUiLogIndex = _fullLogs.Count;
                _pendingUiLogFlush = false;
            }
        }

        if (toAdd == null || toAdd.Count == 0)
            return;

        foreach (var line in toAdd)
        {
            AddLogLine(line);
        }

        LogText = string.Join(Environment.NewLine, Logs);
    }

    private void TryScheduleAccuracy(int epoch)
    {
        if (epoch % AccuracyEveryEpochs != 0)
            return;

        if (_accuracyTask is { IsCompleted: false })
            return;

        _accuracyCts?.Dispose();
        _accuracyCts = new CancellationTokenSource();
        var ct = _accuracyCts.Token;
        _accuracyTask = Task.Run(() =>
        {
            try
            {
                return _host.ComputeAccuracy(ct);
            }
            catch (OperationCanceledException)
            {
                return null;
            }
        }, ct).ContinueWith(t =>
        {
            if (t.IsCompletedSuccessfully && !ct.IsCancellationRequested)
            {
                _lastAccuracy = t.Result;
            }
        }, TaskScheduler.Default);
    }

    private void CancelAccuracyComputation()
    {
        _accuracyCts?.Cancel();
        _accuracyCts?.Dispose();
        _accuracyCts = null;
        _accuracyTask = null;
        _lastAccuracy = null;
    }

    private static int ComputeUiUpdateInterval(int epochsPlanned)
    {
        if (epochsPlanned <= 200) return 1;
        if (epochsPlanned <= 2000) return 10;
        return 50;
    }
}
