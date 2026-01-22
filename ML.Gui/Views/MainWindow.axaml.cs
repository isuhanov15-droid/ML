using System.ComponentModel;
using System.Globalization;
using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using Avalonia.Threading;
using Avalonia.VisualTree;
using LiveChartsCore.Drawing;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Drawing;
using LiveChartsCore.Defaults;
using ML.Gui.ViewModels;

namespace ML.Gui.Views;

public partial class MainWindow : Window
{
    private bool _configured;
    private ScrollViewer? _logScrollViewer;
    private ListBox? _logsList;

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

        vm.Training.PropertyChanged += OnTrainingPropertyChanged;
        vm.Training.LogLines.CollectionChanged += (_, _) =>
            Dispatcher.UIThread.Post(AutoScrollLogsIfNeeded, DispatcherPriority.Background);
        UpdateEpochAxisLabels(vm);
        ConfigureLogAutoScroll();

        _configured = true;
        Dispatcher.UIThread.Post(ApplyChartStyles);
    }

    private void OnTrainingPropertyChanged(object? sender, PropertyChangedEventArgs e)
    {
        if (DataContext is not MainViewModel vm) return;

        if (e.PropertyName == nameof(TrainingViewModel.UpdateUiEveryNEpochs) ||
            e.PropertyName == nameof(TrainingViewModel.EpochStart))
        {
            UpdateEpochAxisLabels(vm);
        }
        if (e.PropertyName == nameof(TrainingViewModel.IsAutoScrollEnabled))
            Dispatcher.UIThread.Post(AutoScrollLogsIfNeeded, DispatcherPriority.Background);
    }

    private void UpdateEpochAxisLabels(MainViewModel vm)
    {
        var axes = LossChart?.XAxes;
        if (axes == null || !axes.Any())
            return;

        var axis = axes.First();
        axis.MinLimit = 0;
        axis.Labeler = value => ((int)Math.Round(value)).ToString(CultureInfo.InvariantCulture);
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

    private void OnScrollLogsDown(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (_logScrollViewer == null || DataContext is not MainViewModel vm)
            return;

        vm.Training.IsAutoScrollEnabled = true;
        ScrollLogsToEnd();
    }

    private void ApplyChartStyles()
    {
        if (LossChart?.Series == null)
            return;

        var trainStroke = this.FindResource("ChartTrainStroke") as IPaint<SkiaSharpDrawingContext>;
        var valStroke = this.FindResource("ChartValStroke") as IPaint<SkiaSharpDrawingContext>;
        var accStroke = this.FindResource("ChartAccStroke") as IPaint<SkiaSharpDrawingContext>;

        foreach (var series in LossChart.Series.OfType<LineSeries<ObservablePoint>>())
        {
            series.GeometrySize = 0;
            series.GeometryStroke = null;
            series.GeometryFill = null;
            series.Fill = null;

            if (string.Equals(series.Name, "Accuracy", StringComparison.OrdinalIgnoreCase))
            {
                if (accStroke != null) series.Stroke = accStroke;
                series.ScalesYAt = 1;
            }
            else if (string.Equals(series.Name, "Train loss", StringComparison.OrdinalIgnoreCase))
            {
                if (trainStroke != null) series.Stroke = trainStroke;
            }
            else
            {
                if (valStroke != null) series.Stroke = valStroke;
            }
        }
    }

    private void ConfigureLogAutoScroll()
    {
        _logsList = this.FindControl<ListBox>("LogsList");
        if (_logsList == null)
            return;

        _logsList.AttachedToVisualTree += (_, _) => HookLogScrollViewer();
        _logsList.DetachedFromVisualTree += (_, _) => UnhookLogScrollViewer();
        HookLogScrollViewer();
        Dispatcher.UIThread.Post(AutoScrollLogsIfNeeded, DispatcherPriority.Background);
    }

    private void HookLogScrollViewer()
    {
        if (_logsList == null)
            return;

        var viewer = _logsList.GetVisualDescendants().OfType<ScrollViewer>().FirstOrDefault();
        if (viewer == null || viewer == _logScrollViewer)
            return;

        if (_logScrollViewer != null)
            _logScrollViewer.ScrollChanged -= OnLogScrollChanged;

        _logScrollViewer = viewer;
        _logScrollViewer.ScrollChanged += OnLogScrollChanged;
    }

    private void UnhookLogScrollViewer()
    {
        if (_logScrollViewer == null)
            return;

        _logScrollViewer.ScrollChanged -= OnLogScrollChanged;
        _logScrollViewer = null;
    }

    private void OnLogScrollChanged(object? sender, ScrollChangedEventArgs e)
    {
        if (_logScrollViewer == null || DataContext is not MainViewModel vm)
            return;

        if (_logScrollViewer.Extent.Height <= _logScrollViewer.Viewport.Height + 1)
        {
            vm.Training.IsAutoScrollEnabled = true;
            return;
        }

        var maxOffset = _logScrollViewer.Extent.Height - _logScrollViewer.Viewport.Height;
        vm.Training.IsAutoScrollEnabled = _logScrollViewer.Offset.Y >= maxOffset - 1;
    }

    private void AutoScrollLogsIfNeeded()
    {
        if (_logScrollViewer == null || DataContext is not MainViewModel vm)
            return;

        if (!vm.Training.IsAutoScrollEnabled)
            return;

        ScrollLogsToEnd();
    }

    private void ScrollLogsToEnd()
    {
        if (_logScrollViewer == null || _logsList == null || DataContext is not MainViewModel vm)
            return;

        Dispatcher.UIThread.Post(() =>
        {
            var lines = vm.Training.LogLines;
            if (lines.Count > 0)
                _logsList.ScrollIntoView(lines[^1]);
            _logScrollViewer.ScrollToEnd();
        }, DispatcherPriority.Background);
    }
}
