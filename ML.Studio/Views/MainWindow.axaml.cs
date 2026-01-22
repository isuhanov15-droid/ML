using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Avalonia.VisualTree;
using ML.Studio.Models;
using ML.Studio.Services;
using ML.Studio.ViewModels;

namespace ML.Studio.Views;

public sealed partial class MainWindow : Window
{
    private MainViewModel ViewModel => (MainViewModel)DataContext!;
    private ScrollViewer? _logsScrollViewer;
    private bool _logsAutoScroll = true;
    private ScrollViewer? _metricsScrollViewer;
    private bool _metricsAutoScroll = true;

    public MainWindow()
    {
        InitializeComponent();
        DataContext = new MainViewModel();
        Opened += OnOpened;
        Closed += OnClosed;
    }

    private async void OnOpened(object? sender, EventArgs e)
    {
        await ViewModel.InitializeAsync();
        AttachLogScroller();
        AttachMetricsScroller();
        ViewModel.Logger.Appended += OnLogAppended;
        ViewModel.Metrics.CollectionChanged += OnMetricsChanged;
    }

    private void OnClosed(object? sender, EventArgs e)
    {
        ViewModel.Logger.Appended -= OnLogAppended;
        ViewModel.Metrics.CollectionChanged -= OnMetricsChanged;
        if (_logsScrollViewer != null)
            _logsScrollViewer.ScrollChanged -= OnLogsScrollChanged;
        if (_metricsScrollViewer != null)
            _metricsScrollViewer.ScrollChanged -= OnMetricsScrollChanged;
    }

    private async void OnConnect(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.ConnectAsync();
    }

    private async void OnDisconnect(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.DisconnectAsync();
    }

    private async void OnNewProject(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var dialog = new InputDialog("Имя нового проекта", "Новый проект")
        {
            WindowStartupLocation = WindowStartupLocation.CenterOwner
        };
        var ok = await dialog.ShowDialog<bool>(this);
        if (!ok || string.IsNullOrWhiteSpace(dialog.Value))
            return;

        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Выберите папку проекта",
            AllowMultiple = false
        });
        var folder = folders.FirstOrDefault();
        if (folder == null)
            return;

        await ViewModel.CreateProjectAsync(folder.Path.LocalPath, dialog.Value, "mlp");
    }

    private async void OnOpenProject(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Открыть проект",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("ML Project") { Patterns = new[] { "*.mlproj" } }
            }
        });

        var file = files.FirstOrDefault();
        if (file == null)
            return;

        await ViewModel.OpenProjectAsync(file.Path.LocalPath);
    }

    private async void OnNewFile(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var dialog = new InputDialog("Имя файла", "new-file.json")
        {
            WindowStartupLocation = WindowStartupLocation.CenterOwner
        };
        var ok = await dialog.ShowDialog<bool>(this);
        if (ok && !string.IsNullOrWhiteSpace(dialog.Value))
            await ViewModel.CreateFileAsync(dialog.Value);
    }

    private async void OnRefreshProject(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.RefreshProjectFilesAsync();
    }

    private async void OnOpenFile(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.OpenSelectedFileAsync();
    }

    private async void OnSaveFile(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.SaveCurrentFileAsync();
    }

    private async void OnSaveFileAs(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var dialog = new InputDialog("Сохранить как", "copy.json")
        {
            WindowStartupLocation = WindowStartupLocation.CenterOwner
        };
        var ok = await dialog.ShowDialog<bool>(this);
        if (ok && !string.IsNullOrWhiteSpace(dialog.Value))
            await ViewModel.SaveCurrentFileAsAsync(dialog.Value);
    }

    private async void OnRenameFile(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var dialog = new InputDialog("Новое имя файла", "renamed.json")
        {
            WindowStartupLocation = WindowStartupLocation.CenterOwner
        };
        var ok = await dialog.ShowDialog<bool>(this);
        if (ok && !string.IsNullOrWhiteSpace(dialog.Value))
            await ViewModel.RenameSelectedFileAsync(dialog.Value);
    }

    private async void OnDeleteFile(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var tab = ViewModel.GetSelectedFileTab();
        if (tab?.IsDirty == true)
        {
            var dialog = new InputDialog("Сохранить как", tab.Title)
            {
                WindowStartupLocation = WindowStartupLocation.CenterOwner
            };
            var ok = await dialog.ShowDialog<bool>(this);
            if (!ok)
                return;
            if (string.IsNullOrWhiteSpace(dialog.Value))
                return;
            await ViewModel.SaveTabAsAsync(tab, dialog.Value);
        }

        var result = await ViewModel.DeleteSelectedFileAsync();
        if (result is { Deleted: true, Required: true } && !string.IsNullOrWhiteSpace(result.RelativePath))
        {
            var recreateDialog = new RecreateDialog($"Файл {result.RelativePath} обязателен. Пересоздать?");
            var recreate = await recreateDialog.ShowDialog<bool>(this);
            if (recreate)
                await ViewModel.RecreateRequiredFileAsync(result.RelativePath!);
        }
    }

    private async void OnSaveAll(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.SaveAllAsync();
    }

    private async void OnProjectTreeDoubleTapped(object? sender, TappedEventArgs e)
    {
        await ViewModel.OpenSelectedFileAsync();
    }

    private async void OnCloseTab(object? sender, RoutedEventArgs e)
    {
        if (sender is not Button { Tag: EditorTab tab })
            return;

        if (tab.IsDirty)
        {
            var dialog = new ConfirmDialog($"Сохранить изменения в {tab.Title}?");
            var result = await dialog.ShowDialog<ConfirmDialogResult>(this);
            if (result == ConfirmDialogResult.Cancel)
                return;
            if (result == ConfirmDialogResult.Save)
                await ViewModel.SaveTabAsync(tab);
        }

        ViewModel.CloseTab(tab);
    }

    private void OnRevealFile(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        ViewModel.RevealSelectedFile();
    }

    private async void OnSaveParams(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.SaveParamsAsync();
    }

    private async void OnKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.KeyModifiers.HasFlag(KeyModifiers.Control) && e.Key == Key.S && e.KeyModifiers.HasFlag(KeyModifiers.Shift))
        {
            await ViewModel.SaveAllAsync();
            e.Handled = true;
            return;
        }

        if (e.KeyModifiers.HasFlag(KeyModifiers.Control) && e.Key == Key.S)
        {
            await ViewModel.SaveCurrentFileAsync();
            e.Handled = true;
        }
    }

    private async void OnLoadParams(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.ReloadParamsAsync();
    }

    private void OnResetParams(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        ViewModel.ResetParams();
    }

    private async void OnStart(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.StartTrainingAsync();
    }

    private async void OnStop(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.StopTrainingAsync();
    }

    private async void OnLoadModel(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.LoadModelAsync();
    }

    private async void OnInferSingle(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.InferSingleAsync();
    }

    private async void OnEvalDataset(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.EvalDatasetAsync();
    }

    private async void OnBuildHeatmap(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.BuildHeatmapAsync();
    }

    private void OnClearHeatmap(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        ViewModel.ClearHeatmap();
    }

    private void OnCancelHeatmap(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        ViewModel.CancelHeatmap();
    }

    private void OnStructureFit(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        StructureGraphView?.FitToView();
    }

    private void OnStructureReset(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        StructureGraphView?.ResetView();
    }

    private async void OnCompareRuns(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        await ViewModel.CompareRunsAsync();
    }

    private void OnClearCompare(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        ViewModel.ClearCompare();
    }

    private void OnThemeVsDark(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        ThemeService.Apply("vs-dark");
    }

    private void OnThemeGraphite(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        ThemeService.Apply("graphite");
    }

    private void OnThemeMidnight(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        ThemeService.Apply("midnight");
    }

    private async void OnCopyRunId(object? sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(ViewModel.CurrentRunId))
            return;

        var clipboard = TopLevel.GetTopLevel(this)?.Clipboard;
        if (clipboard != null)
            await clipboard.SetTextAsync(ViewModel.CurrentRunId);
    }

    private async void OnExportMetricsCsv(object? sender, RoutedEventArgs e)
    {
        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Экспорт метрик CSV",
            DefaultExtension = "csv",
            SuggestedFileName = "metrics.csv",
            FileTypeChoices = new[]
            {
                new FilePickerFileType("CSV") { Patterns = new[] { "*.csv" } }
            }
        });

        if (file == null)
            return;

        await ViewModel.ExportMetricsCsvAsync(file.Path.LocalPath);
    }

    private void OnAutoScale(object? sender, RoutedEventArgs e)
    {
        MetricChart.InvalidateVisual();
    }

    private void AttachLogScroller()
    {
        _logsScrollViewer = LogsList.GetVisualDescendants().OfType<ScrollViewer>().FirstOrDefault();
        if (_logsScrollViewer != null)
            _logsScrollViewer.ScrollChanged += OnLogsScrollChanged;
    }

    private void AttachMetricsScroller()
    {
        _metricsScrollViewer = MetricsList.GetVisualDescendants().OfType<ScrollViewer>().FirstOrDefault();
        if (_metricsScrollViewer != null)
            _metricsScrollViewer.ScrollChanged += OnMetricsScrollChanged;
        if (ViewModel.FollowMetrics && _metricsScrollViewer != null)
            _metricsScrollViewer.Offset = new Vector(_metricsScrollViewer.Offset.X, _metricsScrollViewer.Extent.Height);
    }

    private void OnLogsScrollChanged(object? sender, ScrollChangedEventArgs e)
    {
        if (_logsScrollViewer == null)
            return;

        var maxOffset = Math.Max(0, _logsScrollViewer.Extent.Height - _logsScrollViewer.Viewport.Height);
        _logsAutoScroll = _logsScrollViewer.Offset.Y >= maxOffset - 2;
    }

    private void OnLogAppended()
    {
        if (!_logsAutoScroll || _logsScrollViewer == null)
            return;

        _logsScrollViewer.Offset = new Vector(_logsScrollViewer.Offset.X, _logsScrollViewer.Extent.Height);
    }

    private void OnMetricsScrollChanged(object? sender, ScrollChangedEventArgs e)
    {
        if (_metricsScrollViewer == null)
            return;

        var maxOffset = Math.Max(0, _metricsScrollViewer.Extent.Height - _metricsScrollViewer.Viewport.Height);
        _metricsAutoScroll = _metricsScrollViewer.Offset.Y >= maxOffset - 2;
        if (!_metricsAutoScroll && ViewModel.FollowMetrics)
            ViewModel.FollowMetrics = false;
    }

    private void OnMetricsChanged(object? sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
    {
        if (_metricsScrollViewer == null)
            return;

        if (!ViewModel.FollowMetrics)
            return;

        _metricsScrollViewer.Offset = new Vector(_metricsScrollViewer.Offset.X, _metricsScrollViewer.Extent.Height);
    }
}
