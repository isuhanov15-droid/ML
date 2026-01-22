using System.Collections.ObjectModel;
using Avalonia.Threading;

namespace ML.Studio.Services;

public sealed class LoggingService
{
    private readonly int _capacity;

    public ObservableCollection<string> Entries { get; } = new();
    public event Action? Appended;

    public LoggingService(int capacity = 2000)
    {
        _capacity = capacity;
    }

    public void Append(string level, string message)
    {
        var line = $"[{DateTime.Now:HH:mm:ss}] {level}: {message}";
        Dispatcher.UIThread.Post(() =>
        {
            Entries.Add(line);
            while (Entries.Count > _capacity)
                Entries.RemoveAt(0);
            Appended?.Invoke();
        });
    }
}
