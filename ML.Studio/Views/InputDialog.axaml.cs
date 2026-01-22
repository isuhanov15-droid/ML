using Avalonia.Controls;

namespace ML.Studio.Views;

public sealed partial class InputDialog : Window
{
    public string Prompt { get; }
    public string Value { get; private set; }

    public InputDialog() : this(string.Empty, string.Empty)
    {
    }

    public InputDialog(string prompt, string initialValue)
    {
        InitializeComponent();
        Prompt = prompt;
        Value = initialValue;
        DataContext = this;
    }

    private void OnOk(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is InputDialog dialog)
        {
            Value = dialog.Value;
        }
        Close(true);
    }

    private void OnCancel(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        Close(false);
    }
}
