using System.Text.Json;
using ML.Shared.Protocol;

namespace ML.Host.Storage;

public sealed class RunStore
{
    private readonly string _runsPath;
    private readonly string _metricsDir;
    private readonly object _lock = new();

    public RunStore(string baseDirectory)
    {
        var dataDir = Path.Combine(baseDirectory, "data");
        _runsPath = Path.Combine(dataDir, "runs.json");
        _metricsDir = Path.Combine(dataDir, "run-metrics");
    }

    public IReadOnlyList<RunDto> List(string projectId, string? experimentId)
    {
        lock (_lock)
        {
            var runs = JsonFileStore.ReadList<RunDto>(_runsPath)
                .Where(r => r.projectId == projectId);
            if (!string.IsNullOrWhiteSpace(experimentId))
                runs = runs.Where(r => r.experimentId == experimentId);
            return runs.ToArray();
        }
    }

    public RunDto Create(string runId, string projectId, string experimentId)
    {
        var now = DateTime.UtcNow;
        var run = new RunDto(
            runId: runId,
            projectId: projectId,
            experimentId: experimentId,
            state: "running",
            startedUtc: now,
            endedUtc: null,
            lastEpoch: 0,
            lastLoss: null,
            note: null);

        lock (_lock)
        {
            var runs = JsonFileStore.ReadList<RunDto>(_runsPath);
            runs.Add(run);
            JsonFileStore.WriteList(_runsPath, runs);
        }

        return run;
    }

    public void UpdateState(string runId, string state, string? note = null)
    {
        lock (_lock)
        {
            var runs = JsonFileStore.ReadList<RunDto>(_runsPath);
            var idx = runs.FindIndex(r => r.runId == runId);
            if (idx < 0)
                return;

            var run = runs[idx];
            var endedUtc = run.endedUtc;
            if (state is "stopped" or "finished" or "failed")
                endedUtc = DateTime.UtcNow;

            runs[idx] = run with
            {
                state = state,
                endedUtc = endedUtc,
                note = note ?? run.note
            };
            JsonFileStore.WriteList(_runsPath, runs);
        }
    }

    public void UpdateLast(string runId, int epoch, double loss)
    {
        lock (_lock)
        {
            var runs = JsonFileStore.ReadList<RunDto>(_runsPath);
            var idx = runs.FindIndex(r => r.runId == runId);
            if (idx < 0)
                return;

            var run = runs[idx];
            runs[idx] = run with
            {
                lastEpoch = epoch,
                lastLoss = loss
            };
            JsonFileStore.WriteList(_runsPath, runs);
        }
    }

    public void AppendMetric(RunMetricsPointDto point)
    {
        lock (_lock)
        {
            Directory.CreateDirectory(_metricsDir);
            var path = Path.Combine(_metricsDir, $"{point.runId}.json");
            var metrics = JsonFileStore.ReadList<RunMetricsPointDto>(path);
            metrics.Add(point);
            JsonFileStore.WriteList(path, metrics);
        }
    }

    public IReadOnlyList<RunMetricsPointDto> GetMetrics(string runId)
    {
        lock (_lock)
        {
            var path = Path.Combine(_metricsDir, $"{runId}.json");
            return JsonFileStore.ReadList<RunMetricsPointDto>(path).ToArray();
        }
    }
}
