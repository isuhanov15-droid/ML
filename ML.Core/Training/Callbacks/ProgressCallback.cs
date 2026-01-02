using System;

namespace ML.Core.Training.Callbacks;

/// <summary>
/// Pushes epoch results to an external observer (UI/logger/etc).
/// </summary>
public sealed class ProgressCallback : ITrainCallback
{
    private readonly Action<TrainEpochResult> _onEpoch;

    public ProgressCallback(Action<TrainEpochResult> onEpoch)
    {
        _onEpoch = onEpoch ?? throw new ArgumentNullException(nameof(onEpoch));
    }

    public void OnEpochEnd(TrainEpochResult r) => _onEpoch(r);
}
