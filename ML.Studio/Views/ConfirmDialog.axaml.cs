using Avalonia.Controls;

namespace ML.Studio.Views;

public enum ConfirmDialogResult
{
    Save,
    Discard,
    Cancel
}

public sealed partial class ConfirmDialog : Window
{
    public string Message { get; }

    public ConfirmDialog()
    {
        InitializeComponent();
        Message = "";
        DataContext = this;
    }

    public ConfirmDialog(string message)
    {
        InitializeComponent();
        Message = message;
        DataContext = this;
    }

    private void OnSave(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        Close(ConfirmDialogResult.Save);
    }

    private void OnDiscard(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        Close(ConfirmDialogResult.Discard);
    }

    private void OnCancel(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        Close(ConfirmDialogResult.Cancel);
    }
}
