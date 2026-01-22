using System.Linq;
using Avalonia.Controls;
using Avalonia.Platform.Storage;
using Avalonia.Threading;

namespace ML.Gui.Services;

public interface IFileDialogService
{
    Task<string?> PickOpenFileAsync(string? title = null, IReadOnlyList<FilePickerFileType>? filters = null);
    Task<string?> PickSaveFileAsync(string? title = null, string? suggestedFileName = null, IReadOnlyList<FilePickerFileType>? filters = null);
}

public sealed class FileDialogService : IFileDialogService
{
    private readonly Window _window;

    public FileDialogService(Window window)
    {
        _window = window ?? throw new ArgumentNullException(nameof(window));
    }

    public async Task<string?> PickOpenFileAsync(string? title = null, IReadOnlyList<FilePickerFileType>? filters = null)
    {
        var options = new FilePickerOpenOptions
        {
            Title = title ?? "Open file",
            AllowMultiple = false,
            FileTypeFilter = filters?.ToList()
        };

        var provider = GetStorageProvider();
        var result = await Dispatcher.UIThread.InvokeAsync(() => provider.OpenFilePickerAsync(options));
        var file = result?.FirstOrDefault();
        return file?.TryGetLocalPath() ?? file?.Path.LocalPath;
    }

    public async Task<string?> PickSaveFileAsync(string? title = null, string? suggestedFileName = null, IReadOnlyList<FilePickerFileType>? filters = null)
    {
        var options = new FilePickerSaveOptions
        {
            Title = title ?? "Save file",
            SuggestedFileName = suggestedFileName,
            FileTypeChoices = filters?.ToList()
        };

        var provider = GetStorageProvider();
        var result = await Dispatcher.UIThread.InvokeAsync(() => provider.SaveFilePickerAsync(options));
        return result?.TryGetLocalPath() ?? result?.Path.LocalPath;
    }

    private IStorageProvider GetStorageProvider()
    {
        var topLevel = TopLevel.GetTopLevel(_window) ?? _window;
        return topLevel.StorageProvider;
    }
}
