using System.Linq;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Platform.Storage;
using ML.Core.Losses;
using ML.Core.Optimizers;
using ML.Core.Training;
using ML.Gui.Services;
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

        vm.Training.Configure(
            networkFactory: () => vm.Training.BuildNetworkFromState(),
            trainerFactory: net => new Trainer(net, new AdamOptimizer(vm.Training.LearningRate), new CrossEntropyLoss()),
            trainProvider: () => DemoPipelines.XorData,
            valProvider: () => DemoPipelines.XorData // для демонстрации — те же данные
        );

        _configured = true;
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
}
