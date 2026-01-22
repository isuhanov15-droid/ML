namespace ML.Gui.ViewModels;

public enum LogLevel
{
    Info,
    Warn,
    Error
}

public sealed record LogLineVm(int Epoch, string Text, DateTimeOffset Timestamp, LogLevel Level);
