namespace ML.Gui.Models;

public sealed class EpochViewModel
{
    public int Epoch { get; init; }
    public double TrainLoss { get; init; }
    public double? ValLoss { get; init; }
    public double? Accuracy { get; init; }
    public long ElapsedMs { get; init; }
}
