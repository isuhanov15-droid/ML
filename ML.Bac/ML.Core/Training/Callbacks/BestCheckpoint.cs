namespace ML.Core.Training.Callbacks;

public sealed class BestCheckpoint : ITrainCallback
{
    private readonly Func<TrainEpochResult, double?> _metric;
    private readonly Action _save;
    private readonly double _minDelta;

    private double _best = double.PositiveInfinity;

    public BestCheckpoint(Func<TrainEpochResult, double?> metric, Action save, double minDelta = 1e-4)
    {
        _metric = metric ?? throw new ArgumentNullException(nameof(metric));
        _save = save ?? throw new ArgumentNullException(nameof(save));
        _minDelta = minDelta < 0 ? throw new ArgumentOutOfRangeException(nameof(minDelta)) : minDelta;
    }

    public void OnEpochEnd(TrainEpochResult r)
    {
        var value = _metric(r);
        if (value is null) return;

        double v = value.Value;
        if (v < _best - _minDelta)
        {
            _best = v;
            _save();
        }
    }
}
