using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
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

namespace ML.Gui.ViewModels;

public sealed class TrainingViewModel : ViewModelBase
{
    private readonly TrainingHost _host = new();
    private CancellationTokenSource? _sessionCts;

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
    private int _epochDisplayEvery = 1;
    private ExamplePreset? _selectedPreset;

    private readonly ObservableCollection<double> _trainLossValues = new();
    private readonly ObservableCollection<double?> _valLossValues = new();
    private Task? _runningTask;
    private const int MaxEpochRows = 500;
    private string _logText = "";
    private readonly Stopwatch _stopwatch = new();
    private DispatcherTimer? _elapsedTimer;
    private string _elapsedText = "00:00:00";

    public TrainingViewModel()
    {
        StartCommand = new AsyncRelayCommand(StartAsync, () => CanStart);
        StopCommand = new RelayCommand(Stop, () => CanStop);
        SaveModelCommand = new RelayCommand(SaveModel, () => !IsBusy);
        LoadModelCommand = new RelayCommand(LoadModel, () => !IsBusy);
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
                OnPropertyChanged(nameof(CanStart));
                OnPropertyChanged(nameof(CanStop));
                StartCommand.RaiseCanExecuteChanged();
                StopCommand.RaiseCanExecuteChanged();
                SaveModelCommand.RaiseCanExecuteChanged();
                LoadModelCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public bool IsBusy => State is TrainingState.Starting or TrainingState.Running or TrainingState.Stopping;
    public bool CanStart => State is TrainingState.Idle or TrainingState.Completed or TrainingState.Failed;
    public bool CanStop => State == TrainingState.Running;

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

    public int EpochDisplayEvery
    {
        get => _epochDisplayEvery;
        set => SetField(ref _epochDisplayEvery, value <= 0 ? 1 : value);
    }

    public AsyncRelayCommand StartCommand { get; }
    public RelayCommand StopCommand { get; }
    public RelayCommand SaveModelCommand { get; }
    public RelayCommand LoadModelCommand { get; }
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

    public TrainOptions BuildOptions()
    {
        return new TrainOptions
        {
            Epochs = EpochsCount,
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

    private Task StartAsync()
    {
        if (!CanStart || _runningTask is { IsCompleted: false })
        {
            Status = "Уже идёт обучение.";
            return Task.CompletedTask;
        }

        if (!_host.IsConfigured)
        {
            State = TrainingState.Failed;
            Status = "Нет конфигурации тренировки: вызовите Configure(...) из кода.";
            return Task.CompletedTask;
        }

        if (EpochsCount <= 0 || BatchSize <= 0 || AccumulationSteps <= 0 || InputSize <= 0 || OutputSize <= 0)
        {
            State = TrainingState.Failed;
            Status = "Проверьте размеры и Epochs/BatchSize/AccumSteps (>0).";
            return Task.CompletedTask;
        }

        if (RequiresDatasetPath && string.IsNullOrWhiteSpace(DatasetPath))
        {
            State = TrainingState.Failed;
            Status = "Укажите путь к датасету (CSV/JSON).";
            return Task.CompletedTask;
        }

        ResetProgress();
        Status = "Запуск...";
        State = TrainingState.Starting;
        StartElapsedTimer();

        _sessionCts = new CancellationTokenSource();

        try
        {
            var options = BuildOptions();
            _runningTask = _host.StartAsync(options, OnEpoch, _sessionCts.Token);
            State = TrainingState.Running;

            _ = _runningTask.ContinueWith(t =>
            {
                Dispatcher.UIThread.Post(() =>
                {
                    if (t.IsCanceled)
                    {
                        Status = "Остановлено.";
                        State = TrainingState.Completed;
                    }
                    else if (t.IsFaulted)
                    {
                        Status = $"Ошибка: {t.Exception?.GetBaseException().Message}";
                        State = TrainingState.Failed;
                        if (t.Exception != null)
                            AddLogLine($"Ошибка обучения: {t.Exception.GetBaseException().Message}");
                    }
                    else
                    {
                        Status = "Обучение завершено.";
                        State = TrainingState.Completed;
                    }

                    StopElapsedTimer();
                    _sessionCts?.Dispose();
                    _sessionCts = null;
                    _runningTask = null;
                });
            });
        }
        catch (Exception ex)
        {
            StopElapsedTimer();
            _sessionCts?.Dispose();
            _sessionCts = null;
            Dispatcher.UIThread.Post(() =>
            {
                Status = $"Запуск не удался: {ex.GetBaseException().Message}";
                State = TrainingState.Failed;
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
        _host.Stop();
        _sessionCts?.Cancel();
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
        bool shouldDisplay = r.Epoch == 1 || EpochDisplayEvery <= 1 || (r.Epoch % EpochDisplayEvery == 0);
        if (!shouldDisplay) return;

        var acc = _host.ComputeAccuracy();

        Dispatcher.UIThread.Post(() =>
        {
            // ограничиваем длину таблицы/серий
            if (Epochs.Count >= MaxEpochRows)
                Epochs.RemoveAt(0);
            if (_trainLossValues.Count >= MaxEpochRows)
                _trainLossValues.RemoveAt(0);
            if (_valLossValues.Count >= MaxEpochRows)
                _valLossValues.RemoveAt(0);

            Epochs.Add(new EpochViewModel
            {
                Epoch = r.Epoch,
                TrainLoss = r.TrainLoss,
                ValLoss = r.ValLoss
            });

            _trainLossValues.Add(r.TrainLoss);
            _valLossValues.Add(r.ValLoss);

            Status = $"Эпоха {r.Epoch}: loss={r.TrainLoss:F4}" + (r.ValLoss.HasValue ? $" | val={r.ValLoss:F4}" : "");

            var accText = acc.HasValue ? acc.Value.ToString("F4") : "n/a";
            var line = $"Epoch {r.Epoch}: Accuracy={accText}, TrainLoss={r.TrainLoss:F4}" + (r.ValLoss.HasValue ? $", ValLoss={r.ValLoss:F4}" : "");
            AddLogLine(line);
        });
    }

    private void ResetProgress()
    {
        Epochs.Clear();
        _trainLossValues.Clear();
        _valLossValues.Clear();
        Logs.Clear();
        LogText = string.Empty;
        ElapsedText = "00:00:00";
    }

    private void AddLogLine(string line)
    {
        if (Logs.Count >= MaxEpochRows)
            Logs.RemoveAt(0);
        Logs.Add(line);
        LogText = string.Join(Environment.NewLine, Logs);
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
}
