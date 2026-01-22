using Avalonia.Controls;

namespace ML.Studio.Views;

public sealed partial class RecreateDialog : Window
{
    public string Message { get; }

    public RecreateDialog()
    {
        InitializeComponent();
        Message = "";
        DataContext = this;
    }

    public RecreateDialog(string message)
    {
        InitializeComponent();
        Message = message;
        DataContext = this;
    }

    private void OnRecreate(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        Close(true);
    }

    private void OnClose(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        Close(false);
    }
}
