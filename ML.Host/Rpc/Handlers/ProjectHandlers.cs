using System.Text.Json;
using ML.Host.Storage;
using ML.Shared.Protocol;

namespace ML.Host.Rpc.Handlers;

public sealed class ProjectHandlers
{
    private readonly ProjectStore _projects;
    private readonly ExperimentStore _experiments;
    private readonly RunStore _runs;

    public ProjectHandlers(ProjectStore projects, ExperimentStore experiments, RunStore runs)
    {
        _projects = projects;
        _experiments = experiments;
        _runs = runs;
    }

    public Task<object?> ListProjectsAsync(RpcRequest req)
    {
        return Task.FromResult<object?>(_projects.List());
    }

    public Task<object?> CreateProjectAsync(RpcRequest req)
    {
        string name = GetRequiredString(req, "name");
        return Task.FromResult<object?>(_projects.Create(name));
    }

    public Task<object?> UpdateProjectAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        string name = GetRequiredString(req, "name");
        return Task.FromResult<object?>(_projects.Update(projectId, name));
    }

    public Task<object?> DeleteProjectAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        _projects.Delete(projectId);
        return Task.FromResult<object?>(new { ok = true });
    }

    public Task<object?> ListProjectFilesAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        return Task.FromResult<object?>(_projects.ListFiles(projectId));
    }

    public Task<object?> ReadProjectFileAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        string path = GetRequiredString(req, "path");
        string content = _projects.ReadFile(projectId, path);
        return Task.FromResult<object?>(new { path, content });
    }

    public Task<object?> WriteProjectFileAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        string path = GetRequiredString(req, "path");
        string content = GetRequiredString(req, "content");
        _projects.WriteFile(projectId, path, content);
        return Task.FromResult<object?>(new { ok = true });
    }

    public Task<object?> CreateProjectFileAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        string path = GetRequiredString(req, "path");
        string? content = GetOptionalString(req, "content");
        _projects.CreateFile(projectId, path, content);
        return Task.FromResult<object?>(new { ok = true });
    }

    public Task<object?> RenameProjectFileAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        string path = GetRequiredString(req, "path");
        string newPath = GetRequiredString(req, "newPath");
        _projects.RenameFile(projectId, path, newPath);
        return Task.FromResult<object?>(new { ok = true });
    }

    public Task<object?> DeleteProjectFileAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        string path = GetRequiredString(req, "path");
        _projects.DeleteFile(projectId, path);
        return Task.FromResult<object?>(new { ok = true });
    }

    public Task<object?> ListExperimentsAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        return Task.FromResult<object?>(_experiments.List(projectId));
    }

    public Task<object?> CreateExperimentAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        string name = GetRequiredString(req, "name");
        string? description = GetOptionalString(req, "description");
        JsonElement? trainConfig = GetOptionalElement(req, "trainConfig");
        JsonElement? computeSpec = GetOptionalElement(req, "computeSpec");
        return Task.FromResult<object?>(_experiments.Create(projectId, name, description, trainConfig, computeSpec));
    }

    public Task<object?> ListRunsAsync(RpcRequest req)
    {
        string projectId = GetRequiredString(req, "projectId");
        string? experimentId = GetOptionalString(req, "experimentId");
        return Task.FromResult<object?>(_runs.List(projectId, experimentId));
    }

    public Task<object?> GetRunMetricsAsync(RpcRequest req)
    {
        string runId = GetRequiredString(req, "runId");
        return Task.FromResult<object?>(_runs.GetMetrics(runId));
    }

    private static string GetRequiredString(RpcRequest req, string name)
    {
        var value = GetOptionalString(req, name);
        if (string.IsNullOrWhiteSpace(value))
            throw new InvalidOperationException($"{name} is required");
        return value;
    }

    private static string? GetOptionalString(RpcRequest req, string name)
    {
        if (req.@params is JsonElement el && el.ValueKind == JsonValueKind.Object)
        {
            if (el.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.String)
                return prop.GetString();
        }

        return null;
    }

    private static JsonElement? GetOptionalElement(RpcRequest req, string name)
    {
        if (req.@params is JsonElement el && el.ValueKind == JsonValueKind.Object)
        {
            if (el.TryGetProperty(name, out var prop))
                return prop;
        }

        return null;
    }
}
