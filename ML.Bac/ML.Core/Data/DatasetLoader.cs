using System.Globalization;
using System.Text.Json;

namespace ML.Core.Data;

public static class DatasetLoader
{
    public static IEnumerable<(double[] x, int y)> LoadClassification(string path, bool hasHeader = true, char separator = ',')
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            throw new FileNotFoundException($"Файл датасета не найден: {path}");

        var ext = System.IO.Path.GetExtension(path).ToLowerInvariant();
        return ext is ".json" or ".jsonl"
            ? LoadJson(path)
            : LoadCsv(path, hasHeader, separator);
    }

    private static IEnumerable<(double[] x, int y)> LoadCsv(string path, bool hasHeader, char separator)
    {
        using var reader = new System.IO.StreamReader(path);
        string? line;
        bool skippedHeader = false;
        while ((line = reader.ReadLine()) != null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (hasHeader && !skippedHeader)
            {
                skippedHeader = true;
                continue;
            }

            var parts = line.Split(separator, StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length < 2) continue;

            if (!int.TryParse(parts[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var label))
                continue;

            var features = new double[parts.Length - 1];
            bool ok = true;
            for (int i = 0; i < features.Length; i++)
            {
                if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                {
                    ok = false;
                    break;
                }
                features[i] = v;
            }

            if (ok)
                yield return (features, label);
        }
    }

    private static IEnumerable<(double[] x, int y)> LoadJson(string path)
    {
        var json = System.IO.File.ReadAllText(path);
        var items = JsonSerializer.Deserialize<List<Sample>>(json, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        });
        if (items == null) yield break;

        foreach (var it in items)
        {
            if (it?.X == null) continue;
            yield return (it.X, it.Y);
        }
    }

    private sealed record Sample(double[] X, int Y);
}
