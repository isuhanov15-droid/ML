namespace ML.Core.Training.Callbacks;

public sealed class EarlyStoppingByLoss : ITrainCallback
{
    private readonly Func<TrainEpochResult, double?> _metric; // null => can't evaluate
    private readonly int _patience;
    private readonly double _minDelta;
    private readonly Action? _onStop;
    private readonly Action? _onBest;

    private double _best = double.PositiveInfinity;
    private int _badEpochs = 0;

    public EarlyStoppingByLoss(
        Func<TrainEpochResult, double?> metric,
        int patience = 10,
        double minDelta = 1e-4,
        Action? onBest = null,
        Action? onStop = null)
    {
        _metric = metric ?? throw new ArgumentNullException(nameof(metric));
        _patience = patience <= 0 ? throw new ArgumentOutOfRangeException(nameof(patience)) : patience;
        _minDelta = minDelta < 0 ? throw new ArgumentOutOfRangeException(nameof(minDelta)) : minDelta;
        _onStop = onStop;
        _onBest = onBest;
    }

    public void OnEpochEnd(TrainEpochResult r)
    {
        var value = _metric(r);
        if (value is null) return;

        double v = value.Value;

        // improve if lower by at least minDelta
        if (v < _best - _minDelta)
        {
            _best = v;
            _badEpochs = 0;
            _onBest?.Invoke();
            return;
        }

        _badEpochs++;

        if (_badEpochs >= _patience)
        {
            _onStop?.Invoke();
            r.RequestStop();
        }
    }
}
