using System;
using System.Collections.Generic;

namespace ML.Gui.Models;

/// <summary>
/// Thread-safe buffer for training metrics to decouple training thread from UI updates.
/// </summary>
public sealed class TrainingMetricsBuffer
{
    private readonly object _sync = new();
    private readonly Queue<TrainingMetricsSnapshot> _window;
    private readonly int _maxWindow;
    private TrainingMetricsSnapshot? _latest;
    private bool _hasNew;

    public TrainingMetricsBuffer(int maxWindow = 2000)
    {
        _maxWindow = maxWindow <= 0 ? 2000 : maxWindow;
        _window = new Queue<TrainingMetricsSnapshot>(_maxWindow);
    }

    public void Clear()
    {
        lock (_sync)
        {
            _window.Clear();
            _latest = null;
            _hasNew = false;
        }
    }

    public void Push(TrainingMetricsSnapshot snapshot)
    {
        lock (_sync)
        {
            _latest = snapshot;
            _hasNew = true;
            _window.Enqueue(snapshot);
            while (_window.Count > _maxWindow)
                _window.Dequeue();
        }
    }

    public bool TryConsumeLatest(out TrainingMetricsSnapshot snapshot)
    {
        lock (_sync)
        {
            if (_hasNew && _latest.HasValue)
            {
                snapshot = _latest.Value;
                _hasNew = false;
                return true;
            }
        }

        snapshot = default;
        return false;
    }

    public TrainingMetricsSnapshot[] SnapshotWindow()
    {
        lock (_sync)
        {
            return _window.ToArray();
        }
    }
}

public readonly record struct TrainingMetricsSnapshot(
    int Epoch,
    double TrainLoss,
    double? ValLoss,
    double? Accuracy,
    DateTimeOffset Timestamp,
    long ElapsedMs);
