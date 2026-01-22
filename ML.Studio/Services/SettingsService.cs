using System.Text.Json;

namespace ML.Studio.Services;

public sealed class SettingsService
{
    private readonly string _path;

    public SettingsService()
    {
        var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "ML.Studio");
        Directory.CreateDirectory(dir);
        _path = Path.Combine(dir, "settings.json");
    }

    public StudioSettings Load()
    {
        if (!File.Exists(_path))
            return new StudioSettings();

        try
        {
            var json = File.ReadAllText(_path);
            return JsonSerializer.Deserialize<StudioSettings>(json) ?? new StudioSettings();
        }
        catch
        {
            return new StudioSettings();
        }
    }

    public void Save(StudioSettings settings)
    {
        var json = JsonSerializer.Serialize(settings, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(_path, json);
    }
}

public sealed class StudioSettings
{
    public string? LastProjectPath { get; set; }
}
