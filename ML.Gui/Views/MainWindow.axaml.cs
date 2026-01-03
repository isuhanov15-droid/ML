using System.Linq;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using LiveChartsCore.Drawing;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Drawing;
using ML.Core.Losses;
using ML.Core.Optimizers;
using ML.Core.Training;
using ML.Gui.ViewModels;

namespace ML.Gui.Views;

public partial class MainWindow : Window
{
    private bool _configured;

    public MainWindow()
    {
        InitializeComponent();
        DataContextChanged += OnDataContextChanged;
    }

    private void InitializeComponent()
    {
        AvaloniaXamlLoader.Load(this);
    }

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (_configured) return;
        if (DataContext is not MainViewModel vm) return;

        var preset = vm.Training.SelectedPreset ?? vm.Training.Presets.First();
        vm.Training.Configure(
            networkFactory: () => vm.Training.BuildNetworkFromState(),
            trainerFactory: net => new Trainer(net, new AdamOptimizer(vm.Training.LearningRate), new CrossEntropyLoss()),
            trainProvider: () => preset.BuildTrain(vm.Training.DatasetPath),
            valProvider: () => preset.BuildVal(vm.Training.DatasetPath)
        );

        _configured = true;
        Dispatcher.UIThread.Post(ApplyChartStyles);
    }

    private async void OnBrowseSave(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Сохранить модель",
            SuggestedFileName = string.IsNullOrWhiteSpace(vm.Training.ModelName) ? "model.json" : vm.Training.ModelName,
            FileTypeChoices = new[]
            {
                new FilePickerFileType("Model json") { Patterns = new[] { "*.json" } }
            }
        });

        var path = file?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(path))
            vm.Training.SetSavePath(path);
    }

    private async void OnBrowseDataset(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Выберите датасет (CSV/JSON)",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("CSV/JSON") { Patterns = new[] { "*.csv", "*.json", "*.jsonl" } }
            }
        });

        var path = files?.FirstOrDefault()?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(path))
            vm.Training.DatasetPath = path;
    }

    private async void OnBrowseLoad(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Загрузить модель",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Model json") { Patterns = new[] { "*.json" } }
            }
        });

        var path = files?.FirstOrDefault()?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(path))
            vm.Training.SetLoadPath(path);
    }

    private async void OnSaveAs(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Сохранить модель",
            SuggestedFileName = string.IsNullOrWhiteSpace(vm.Training.ModelName) ? "model.json" : vm.Training.ModelName,
            FileTypeChoices = new[]
            {
                new FilePickerFileType("Model json") { Patterns = new[] { "*.json" } }
            }
        });

        var path = file?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(path))
        {
            vm.Training.SetSavePath(path);
            vm.Training.SaveModelCommand.Execute(null);
        }
    }

    private async void OnLoadFile(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Загрузить модель",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Model json") { Patterns = new[] { "*.json" } }
            }
        });

        var path = files?.FirstOrDefault()?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(path))
        {
            vm.Training.SetLoadPath(path);
            vm.Training.LoadModelCommand.Execute(null);
        }
    }

    private async void OnBrowseInferenceModel(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Загрузить модель для инференса",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Model json") { Patterns = new[] { "*.json" } }
            }
        });

        var path = files?.FirstOrDefault()?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(path))
            vm.Inference.ModelPath = path;
    }

    private async void OnBrowseSaveConfig(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Сохранить конфиг",
            SuggestedFileName = string.IsNullOrWhiteSpace(vm.Training.ModelName) ? "config.json" : $"{vm.Training.ModelName}_config.json",
            FileTypeChoices = new[] { new FilePickerFileType("Config json") { Patterns = new[] { "*.json" } } }
        });

        var path = file?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(path))
            vm.Training.ConfigSavePath = path;
    }

    private void OnSaveConfig(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;
        vm.Training.SaveConfig();
    }

    private async void OnBrowseLoadConfig(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Загрузить конфиг",
            AllowMultiple = false,
            FileTypeFilter = new[] { new FilePickerFileType("Config json") { Patterns = new[] { "*.json" } } }
        });

        var path = files?.FirstOrDefault()?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(path))
            vm.Training.ConfigLoadPath = path;
    }

    private void OnLoadConfig(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;
        vm.Training.LoadConfig(vm.Training.ConfigLoadPath);
    }

    private async void OnBrowseExportMetrics(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Экспорт метрик",
            SuggestedFileName = string.IsNullOrWhiteSpace(vm.Training.ModelName) ? "metrics.csv" : $"{vm.Training.ModelName}_metrics.csv",
            FileTypeChoices = new[] { new FilePickerFileType("CSV") { Patterns = new[] { "*.csv" } } }
        });

        var path = file?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(path))
            vm.Training.MetricsExportPath = path;
    }

    private void OnExportMetrics(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;
        vm.Training.ExportMetrics();
    }

    private void ApplyChartStyles()
    {
        if (LossChart?.Series == null)
            return;

        var trainStroke = this.FindResource("ChartTrainStroke") as IPaint<SkiaSharpDrawingContext>;
        var valStroke = this.FindResource("ChartValStroke") as IPaint<SkiaSharpDrawingContext>;

        foreach (var series in LossChart.Series)
        {
            switch (series)
            {
                case LineSeries<double> train:
                    train.GeometrySize = 4;
                    train.Fill = null;
                    if (trainStroke != null) train.Stroke = trainStroke;
                    break;
                case LineSeries<double?> val:
                    val.GeometrySize = 4;
                    val.Fill = null;
                    if (valStroke != null) val.Stroke = valStroke;
                    break;
            }
        }
    }
}
