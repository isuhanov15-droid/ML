using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using ML.Core;
using ML.Core.Layers;
using ML.Core.Losses;
using ML.Core.Optimizers;
using ML.Core.Serialization;
using ML.Core.Training;
using ML.Core.Training.Callbacks;
using ML.Gui.Models;
using ML.Gui.Services;
using ML.Gui.Utils;
using SkiaSharp;

namespace ML.Gui.ViewModels;

public sealed class TrainingViewModel : ViewModelBase
{
    private readonly TrainingHost _host = new();
    private CancellationTokenSource? _sessionCts;

    private bool _isRunning;
    private string _status = "Готов";
    private string _datasetPath = "Demo XOR встроенный датасет";
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

    private readonly ObservableCollection<double> _trainLossValues = new();
    private readonly ObservableCollection<double?> _valLossValues = new();

    public TrainingViewModel()
    {
        StartCommand = new AsyncRelayCommand(StartAsync, () => !IsRunning);
        StopCommand = new RelayCommand(Stop, () => IsRunning);
        SaveModelCommand = new RelayCommand(SaveModel, () => !IsRunning);
        LoadModelCommand = new RelayCommand(LoadModel, () => !IsRunning);

        Series = new ISeries[]
        {
            new LineSeries<double>
            {
                Name = "Train loss",
                Values = _trainLossValues,
                Fill = null,
                GeometrySize = 6,
                Stroke = new SolidColorPaint(SKColors.DeepSkyBlue, 3)
            },
            new LineSeries<double?>
            {
                Name = "Val loss",
                Values = _valLossValues,
                Fill = null,
                GeometrySize = 6,
                Stroke = new SolidColorPaint(SKColors.Orange, 3)
            }
        };

        XAxes = new[] { new Axis { Name = "Epoch", MinLimit = 1 } };
        YAxes = new[] { new Axis { Name = "Loss" } };
    }

    public ObservableCollection<EpochViewModel> Epochs { get; } = new();

    public bool IsRunning
    {
        get => _isRunning;
        private set
        {
            if (SetField(ref _isRunning, value))
            {
                StartCommand.RaiseCanExecuteChanged();
                StopCommand.RaiseCanExecuteChanged();
                SaveModelCommand.RaiseCanExecuteChanged();
                LoadModelCommand.RaiseCanExecuteChanged();
            }
        }
    }

    public string Status
    {
        get => _status;
        private set => SetField(ref _status, value);
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

    public AsyncRelayCommand StartCommand { get; }
    public RelayCommand StopCommand { get; }
    public RelayCommand SaveModelCommand { get; }
    public RelayCommand LoadModelCommand { get; }

    public ISeries[] Series { get; }
    public Axis[] XAxes { get; }
    public Axis[] YAxes { get; }

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

    private async Task StartAsync()
    {
        if (!_host.IsConfigured)
        {
            Status = "Нет конфигурации тренировки: вызовите Configure(...) из кода.";
            return;
        }

        if (EpochsCount <= 0 || BatchSize <= 0 || AccumulationSteps <= 0 || InputSize <= 0 || OutputSize <= 0)
        {
            Status = "Проверьте размеры и Epochs/BatchSize/AccumSteps (>0).";
            return;
        }

        ResetProgress();
        Status = "Запуск...";
        IsRunning = true;

        _sessionCts = new CancellationTokenSource();

        try
        {
            var options = BuildOptions();
            await _host.StartAsync(options, OnEpoch, _sessionCts.Token);
            Status = "Обучение завершено.";
        }
        catch (OperationCanceledException)
        {
            Status = "Остановлено.";
        }
        catch (Exception ex)
        {
            Status = $"Ошибка: {ex.Message}";
        }
        finally
        {
            IsRunning = false;
            _sessionCts?.Dispose();
            _sessionCts = null;
        }
    }

    private void Stop()
    {
        if (!IsRunning) return;
        Status = "Остановка...";
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
            Status = $"Загружена модель: {(string.IsNullOrWhiteSpace(LoadPath) ? ModelName : LoadPath)}";
        }
        catch (Exception ex)
        {
            Status = $"Загрузка не удалась: {ex.Message}";
        }
    }

    private void OnEpoch(TrainEpochResult r)
    {
        Dispatcher.UIThread.Post(() =>
        {
            Epochs.Add(new EpochViewModel
            {
                Epoch = r.Epoch,
                TrainLoss = r.TrainLoss,
                ValLoss = r.ValLoss
            });

            _trainLossValues.Add(r.TrainLoss);
            _valLossValues.Add(r.ValLoss);

            Status = $"Эпоха {r.Epoch}: loss={r.TrainLoss:F4}" + (r.ValLoss.HasValue ? $" | val={r.ValLoss:F4}" : "");
        });
    }

    private void ResetProgress()
    {
        Epochs.Clear();
        _trainLossValues.Clear();
        _valLossValues.Clear();
    }
}
