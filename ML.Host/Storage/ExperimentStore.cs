using System.Text.Json;
using ML.Shared.Protocol;

namespace ML.Host.Storage;

public sealed class ExperimentStore
{
    private readonly string _path;
    private readonly object _lock = new();

    public ExperimentStore(string baseDirectory)
    {
        _path = Path.Combine(baseDirectory, "data", "experiments.json");
    }

    public IReadOnlyList<ExperimentDto> List(string projectId)
    {
        lock (_lock)
            return JsonFileStore.ReadList<ExperimentDto>(_path)
                .Where(e => e.projectId == projectId)
                .ToArray();
    }

    public ExperimentDto? Get(string experimentId)
    {
        lock (_lock)
            return JsonFileStore.ReadList<ExperimentDto>(_path)
                .FirstOrDefault(e => e.experimentId == experimentId);
    }

    public ExperimentDto Create(string projectId, string name, string? description, JsonElement? trainConfig, JsonElement? computeSpec)
    {
        var now = DateTime.UtcNow;
        var experiment = new ExperimentDto(
            experimentId: Guid.NewGuid().ToString("N"),
            projectId: projectId,
            name: name,
            description: description,
            createdUtc: now,
            updatedUtc: now,
            trainConfig: trainConfig?.Clone(),
            computeSpec: computeSpec?.Clone());

        lock (_lock)
        {
            var items = JsonFileStore.ReadList<ExperimentDto>(_path);
            items.Add(experiment);
            JsonFileStore.WriteList(_path, items);
        }

        return experiment;
    }
}
