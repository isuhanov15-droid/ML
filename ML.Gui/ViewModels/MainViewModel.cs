namespace ML.Gui.ViewModels;

public sealed class MainViewModel : ViewModelBase
{
    public TrainingViewModel Training { get; } = new();
}
