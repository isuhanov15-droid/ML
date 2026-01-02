using System;

namespace ML.Core.Training.Callbacks;

/// <summary>
/// Checks an external flag after each epoch and requests stop if needed.
/// </summary>
public sealed class ManualStopCallback : ITrainCallback
{
    private readonly Func<bool> _shouldStop;

    public ManualStopCallback(Func<bool> shouldStop)
    {
        _shouldStop = shouldStop ?? throw new ArgumentNullException(nameof(shouldStop));
    }

    public void OnEpochEnd(TrainEpochResult r)
    {
        if (_shouldStop())
            r.RequestStop();
    }
}
