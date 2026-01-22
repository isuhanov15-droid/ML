using System.Text.Json;
using System.Text.Json.Serialization;

namespace ML.Gui.Services;

public sealed class SettingsService
{
    private readonly string _path;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    public SettingsService()
    {
        _path = ResolvePath();
    }

    public SettingsData Load()
    {
        try
        {
            if (File.Exists(_path))
            {
                var json = File.ReadAllText(_path);
                var data = JsonSerializer.Deserialize<SettingsData>(json, JsonOptions);
                if (data != null)
                    return data;
            }
        }
        catch
        {
        }

        return new SettingsData();
    }

    public void Save(SettingsData data)
    {
        try
        {
            var dir = Path.GetDirectoryName(_path);
            if (!string.IsNullOrWhiteSpace(dir))
                Directory.CreateDirectory(dir);

            var json = JsonSerializer.Serialize(data, JsonOptions);
            File.WriteAllText(_path, json);
        }
        catch
        {
        }
    }

    private static string ResolvePath()
    {
        var appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        return Path.Combine(appData, "ML.Gui", "settings.json");
    }
}

public sealed class SettingsData
{
    public string Host { get; set; } = "127.0.0.1";
    public int Port { get; set; } = 5001;
    public string? LastSavePath { get; set; }
    public string? LastLoadPath { get; set; }
    public string? LastPreset { get; set; }
    public double? WindowWidth { get; set; }
    public double? WindowHeight { get; set; }
}
