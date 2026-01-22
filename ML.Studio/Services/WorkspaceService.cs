using System.Text.Json;
using System.Text.Json.Serialization;
using ML.Studio.Models;

namespace ML.Studio.Services;

public sealed class WorkspaceService
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true
    };

    public async Task<StudioProject> CreateProjectAsync(string folderPath, string name, string projectType)
    {
        if (string.IsNullOrWhiteSpace(folderPath))
            throw new InvalidOperationException("Папка проекта не задана.");

        Directory.CreateDirectory(folderPath);

        var now = DateTime.UtcNow;
        var files = new List<ProjectFileItem>();

        var networkPath = Path.Combine(folderPath, "network.json");
        var paramsPath = Path.Combine(folderPath, "params.json");
        var datasetPath = Path.Combine(folderPath, "dataset.csv");

        if (!File.Exists(networkPath))
            await File.WriteAllTextAsync(networkPath, DefaultNetworkJson());
        if (!File.Exists(paramsPath))
            await File.WriteAllTextAsync(paramsPath, DefaultParamsJson());
        if (!File.Exists(datasetPath))
            await File.WriteAllTextAsync(datasetPath, DefaultDatasetCsv());

        files.Add(ToItem(folderPath, "network.json", "Network"));
        files.Add(ToItem(folderPath, "params.json", "Params"));
        files.Add(ToItem(folderPath, "dataset.csv", "Dataset"));

        var project = new StudioProject(
            Name: name,
            ProjectType: projectType,
            RootPath: folderPath,
            Files: files.ToArray(),
            CreatedUtc: now,
            UpdatedUtc: now);

        await SaveProjectAsync(project);
        return project;
    }

    public async Task<StudioProject> OpenProjectAsync(string mlprojPath)
    {
        var json = await File.ReadAllTextAsync(mlprojPath);
        var dto = JsonSerializer.Deserialize<ProjectFileDto>(json, JsonOptions);
        if (dto == null)
            throw new InvalidOperationException("Файл проекта поврежден.");

        var root = Path.GetDirectoryName(mlprojPath) ?? Directory.GetCurrentDirectory();
        var files = dto.Files
            .Select(f => ToItem(root, f.Path, f.Kind))
            .ToArray();

        return new StudioProject(
            Name: dto.Name,
            ProjectType: dto.ProjectType,
            RootPath: root,
            Files: files,
            CreatedUtc: dto.CreatedUtc,
            UpdatedUtc: dto.UpdatedUtc);
    }

    public async Task SaveProjectAsync(StudioProject project)
    {
        var dto = new ProjectFileDto
        {
            Name = project.Name,
            ProjectType = project.ProjectType,
            Files = project.Files.Select(f => new ProjectFileEntry
            {
                Path = f.RelativePath,
                Kind = f.Kind
            }).ToList(),
            CreatedUtc = project.CreatedUtc,
            UpdatedUtc = DateTime.UtcNow
        };

        var path = Path.Combine(project.RootPath, $"{project.Name}.mlproj");
        var json = JsonSerializer.Serialize(dto, JsonOptions);
        await File.WriteAllTextAsync(path, json);
    }

    public string GetProjectFilePath(StudioProject project)
    {
        return Path.Combine(project.RootPath, $"{project.Name}.mlproj");
    }

    public static ProjectFileItem ToItem(string root, string relativePath, string kind)
    {
        var full = Path.Combine(root, relativePath);
        return new ProjectFileItem(
            RelativePath: relativePath,
            Kind: kind,
            FullPath: full,
            DisplayName: Path.GetFileName(relativePath));
    }

    private static string DefaultNetworkJson()
    {
        return "{\n  \"inputSize\": 2,\n  \"outputSize\": 2,\n  \"hidden\": [4, 4],\n  \"activation\": \"ReLu\",\n  \"seed\": 123\n}\n";
    }

    private static string DefaultParamsJson()
    {
        return "{\n  \"epochs\": 50,\n  \"learningRate\": 0.05,\n  \"batchSize\": 8\n}\n";
    }

    private static string DefaultDatasetCsv()
    {
        return "x1,x2,y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n";
    }

    public static string GetDefaultContentFor(string relativePath)
    {
        var name = Path.GetFileName(relativePath).ToLowerInvariant();
        return name switch
        {
            "network.json" => DefaultNetworkJson(),
            "params.json" => DefaultParamsJson(),
            "dataset.csv" => DefaultDatasetCsv(),
            _ => ""
        };
    }

    public static string GetDefaultKindFor(string relativePath)
    {
        var name = Path.GetFileName(relativePath).ToLowerInvariant();
        return name switch
        {
            "network.json" => "Network",
            "params.json" => "Params",
            "dataset.csv" => "Dataset",
            "model.json" => "Model",
            _ => "Other"
        };
    }

    private sealed class ProjectFileDto
    {
        public string Name { get; set; } = "";
        public string ProjectType { get; set; } = "";
        public List<ProjectFileEntry> Files { get; set; } = new();
        public DateTime CreatedUtc { get; set; }
        public DateTime UpdatedUtc { get; set; }
    }

    private sealed class ProjectFileEntry
    {
        public string Path { get; set; } = "";
        public string Kind { get; set; } = "";
    }
}
