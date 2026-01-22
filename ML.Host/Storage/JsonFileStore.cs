using System.Text.Json;
using ML.Shared.Protocol;

namespace ML.Host.Storage;

internal static class JsonFileStore
{
    public static List<T> ReadList<T>(string path)
    {
        if (!File.Exists(path))
            return new List<T>();

        try
        {
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<List<T>>(json, JsonOptions.Default) ?? new List<T>();
        }
        catch
        {
            return new List<T>();
        }
    }

    public static void WriteList<T>(string path, List<T> items)
    {
        var json = JsonSerializer.Serialize(items, JsonOptions.Default);
        WriteAtomic(path, json);
    }

    public static void WriteAtomic(string path, string content)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        var tempPath = path + ".tmp";
        File.WriteAllText(tempPath, content);
        File.Move(tempPath, path, true);
    }
}
