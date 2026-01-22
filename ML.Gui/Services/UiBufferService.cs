using System.Collections.Concurrent;
using Avalonia.Threading;
using ML.Gui.Models;

namespace ML.Gui.Services;

public sealed class UiBufferService : IDisposable
{
    private readonly ConcurrentQueue<MetricPoint> _metricQueue = new();
    private readonly ConcurrentQueue<string> _logQueue = new();
    private readonly DispatcherTimer _timer;
    private readonly EventHandler _tickHandler;
    private readonly Action<IReadOnlyList<MetricPoint>> _metricsDrain;
    private readonly Action<IReadOnlyList<string>> _logsDrain;

    public int MaxMetrics { get; set; } = 5000;
    public int MaxLogs { get; set; } = 2000;
    public int MaxMetricsPerTick { get; set; } = 300;
    public int MaxLogsPerTick { get; set; } = 60;

    public UiBufferService(
        Action<IReadOnlyList<MetricPoint>> metricsDrain,
        Action<IReadOnlyList<string>> logsDrain,
        TimeSpan? interval = null)
    {
        _metricsDrain = metricsDrain ?? throw new ArgumentNullException(nameof(metricsDrain));
        _logsDrain = logsDrain ?? throw new ArgumentNullException(nameof(logsDrain));
        _timer = new DispatcherTimer { Interval = interval ?? TimeSpan.FromMilliseconds(150) };
        _tickHandler = (_, _) => Flush();
        _timer.Tick += _tickHandler;
        _timer.Start();
    }

    public void EnqueueMetric(MetricPoint point)
    {
        _metricQueue.Enqueue(point);
        TrimQueue(_metricQueue, MaxMetrics);
    }

    public void EnqueueLog(string line)
    {
        _logQueue.Enqueue(line);
        TrimQueue(_logQueue, MaxLogs);
    }

    public void Flush()
    {
        var metrics = Dequeue(_metricQueue, MaxMetricsPerTick);
        if (metrics.Count > 0)
            _metricsDrain(metrics);

        var logs = Dequeue(_logQueue, MaxLogsPerTick);
        if (logs.Count > 0)
            _logsDrain(logs);
    }

    private static List<T> Dequeue<T>(ConcurrentQueue<T> queue, int max)
    {
        var items = new List<T>(max);
        while (items.Count < max && queue.TryDequeue(out var item))
            items.Add(item);
        return items;
    }

    private static void TrimQueue<T>(ConcurrentQueue<T> queue, int max)
    {
        if (max <= 0) return;
        while (queue.Count > max && queue.TryDequeue(out _))
        {
        }
    }

    public void Dispose()
    {
        _timer.Stop();
        _timer.Tick -= _tickHandler;
    }
}
