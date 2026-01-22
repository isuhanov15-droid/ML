namespace ML.Core.Training;

public sealed class TrainEpochResult
{
    public int Epoch { get; }
    public double TrainLoss { get; }
    public double? ValLoss { get; }

    public bool StopRequested { get; private set; }

    public TrainEpochResult(int epoch, double trainLoss, double? valLoss)
    {
        Epoch = epoch;
        TrainLoss = trainLoss;
        ValLoss = valLoss;
    }

    public void RequestStop() => StopRequested = true;
}
