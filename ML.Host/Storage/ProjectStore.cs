using ML.Shared.Protocol;

namespace ML.Host.Storage;

public sealed class ProjectStore
{
    private readonly string _path;
    private readonly string _projectsDir;
    private readonly object _lock = new();

    public ProjectStore(string baseDirectory)
    {
        var dataDir = Path.Combine(baseDirectory, "data");
        _path = Path.Combine(dataDir, "projects.json");
        _projectsDir = Path.Combine(dataDir, "projects");
    }

    public IReadOnlyList<ProjectDto> List()
    {
        lock (_lock)
            return JsonFileStore.ReadList<ProjectDto>(_path).ToArray();
    }

    public ProjectDto Create(string name)
    {
        var now = DateTime.UtcNow;
        var project = new ProjectDto(
            projectId: Guid.NewGuid().ToString("N"),
            name: name,
            createdUtc: now,
            updatedUtc: now);

        lock (_lock)
        {
            var items = JsonFileStore.ReadList<ProjectDto>(_path);
            items.Add(project);
            JsonFileStore.WriteList(_path, items);
            EnsureProjectFiles(project.projectId, project.name);
        }

        return project;
    }

    public ProjectDto Update(string projectId, string name)
    {
        lock (_lock)
        {
            var items = JsonFileStore.ReadList<ProjectDto>(_path);
            var idx = items.FindIndex(p => p.projectId == projectId);
            if (idx < 0)
                throw new InvalidOperationException("Project not found.");

            var existing = items[idx];
            var updated = existing with
            {
                name = name,
                updatedUtc = DateTime.UtcNow
            };
            items[idx] = updated;
            JsonFileStore.WriteList(_path, items);
            return updated;
        }
    }

    public void Delete(string projectId)
    {
        lock (_lock)
        {
            var items = JsonFileStore.ReadList<ProjectDto>(_path);
            items.RemoveAll(p => p.projectId == projectId);
            JsonFileStore.WriteList(_path, items);
        }

        var dir = GetProjectDir(projectId);
        if (Directory.Exists(dir))
        {
            try { Directory.Delete(dir, true); } catch { }
        }
    }

    public IReadOnlyList<ProjectFileDto> ListFiles(string projectId)
    {
        var dir = GetProjectDir(projectId);
        if (!Directory.Exists(dir))
            return Array.Empty<ProjectFileDto>();

        var files = Directory.EnumerateFiles(dir, "*.*", SearchOption.AllDirectories)
            .Where(IsAllowedExtension)
            .Select(path =>
            {
                var info = new FileInfo(path);
                return new ProjectFileDto(
                    path: Path.GetRelativePath(dir, path),
                    sizeBytes: info.Length,
                    updatedUtc: info.LastWriteTimeUtc);
            })
            .OrderBy(f => f.path)
            .ToArray();

        return files;
    }

    public string ReadFile(string projectId, string relativePath)
    {
        var path = GetSafePath(projectId, relativePath);
        return File.ReadAllText(path);
    }

    public void WriteFile(string projectId, string relativePath, string content)
    {
        var path = GetSafePath(projectId, relativePath);
        JsonFileStore.WriteAtomic(path, content);
    }

    public void CreateFile(string projectId, string relativePath, string? content)
    {
        var path = GetSafePath(projectId, relativePath);
        if (!File.Exists(path))
            JsonFileStore.WriteAtomic(path, content ?? string.Empty);
    }

    public void RenameFile(string projectId, string relativePath, string newRelativePath)
    {
        var path = GetSafePath(projectId, relativePath);
        var newPath = GetSafePath(projectId, newRelativePath);
        File.Move(path, newPath, true);
    }

    public void DeleteFile(string projectId, string relativePath)
    {
        var path = GetSafePath(projectId, relativePath);
        if (File.Exists(path))
            File.Delete(path);
    }

    public bool TryGetProjectModelPath(string projectId, out string? modelPath)
    {
        var path = Path.Combine(GetProjectDir(projectId), "model.json");
        if (File.Exists(path))
        {
            modelPath = path;
            return true;
        }

        modelPath = null;
        return false;
    }

    private string GetProjectDir(string projectId)
    {
        return Path.Combine(_projectsDir, projectId);
    }

    private string GetSafePath(string projectId, string relativePath)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
            throw new InvalidOperationException("Path is required.");

        var baseDir = GetProjectDir(projectId);
        Directory.CreateDirectory(baseDir);
        var fullPath = Path.GetFullPath(Path.Combine(baseDir, relativePath));
        var baseFull = Path.GetFullPath(baseDir);

        if (!fullPath.StartsWith(baseFull, StringComparison.Ordinal))
            throw new InvalidOperationException("Invalid path.");

        if (!IsAllowedExtension(fullPath))
            throw new InvalidOperationException("File type not allowed.");

        var dir = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        return fullPath;
    }

    private static bool IsAllowedExtension(string path)
    {
        var ext = Path.GetExtension(path).ToLowerInvariant();
        return ext is ".json" or ".csv" or ".txt";
    }

    private void EnsureProjectFiles(string projectId, string name)
    {
        var dir = GetProjectDir(projectId);
        Directory.CreateDirectory(dir);

        var modelPath = Path.Combine(dir, "model.json");
        var dataPath = Path.Combine(dir, "dataset.csv");
        var trainPath = Path.Combine(dir, "train.json");

        if (!File.Exists(modelPath))
            JsonFileStore.WriteAtomic(modelPath, "{\n  \"name\": \"" + name + "\",\n  \"type\": \"xor\"\n}\n");
        if (!File.Exists(dataPath))
            JsonFileStore.WriteAtomic(dataPath, "x1,x2,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n");
        if (!File.Exists(trainPath))
            JsonFileStore.WriteAtomic(trainPath, "{\n  \"epochs\": 100,\n  \"batchSize\": 8,\n  \"learningRate\": 0.05\n}\n");
    }
}
