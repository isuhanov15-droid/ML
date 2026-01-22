using System.Text.Json;
using ML.Studio.Models;

namespace ML.Studio.Services;

public static class NetworkGraphBuilder
{
    public static GraphModel Build(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return GraphModel.Empty;

        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            var nodes = new List<GraphNode>();
            var edges = new List<GraphEdge>();

            int inputSize = TryGetInt(root, "inputSize");
            int outputSize = TryGetInt(root, "outputSize");
            var hidden = TryGetIntArray(root, "hidden");
            string activation = TryGetString(root, "activation") ?? "Unknown";

            var sizes = new List<int>();
            if (inputSize > 0)
                sizes.Add(inputSize);
            sizes.AddRange(hidden);
            if (outputSize > 0)
                sizes.Add(outputSize);

            const double nodeWidth = 180;
            const double nodeHeight = 60;
            const double xStep = 220;

            for (int i = 0; i < sizes.Count; i++)
            {
                int from = i == 0 ? sizes[0] : sizes[i - 1];
                int to = sizes[i];
                string title = i == 0
                    ? $"Input {to}"
                    : i == sizes.Count - 1
                        ? $"Output {from} -> {to}"
                        : $"Dense {from} -> {to}";
                string subtitle = $"activation={activation}";
                var meta = new Dictionary<string, string>
                {
                    ["type"] = i == 0 ? "Input" : (i == sizes.Count - 1 ? "Output" : "Dense"),
                    ["units"] = to.ToString(),
                    ["activation"] = activation
                };

                double x = i * xStep;
                double y = 0;
                nodes.Add(new GraphNode(
                    Id: i,
                    Title: title,
                    Subtitle: subtitle,
                    Meta: meta,
                    X: x,
                    Y: y,
                    Width: nodeWidth,
                    Height: nodeHeight));

                if (i > 0)
                    edges.Add(new GraphEdge(i - 1, i));
            }

            return new GraphModel(nodes, edges);
        }
        catch
        {
            return GraphModel.Empty;
        }
    }

    private static int TryGetInt(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var prop))
            return 0;
        if (prop.ValueKind == JsonValueKind.Number && prop.TryGetInt32(out var value))
            return value;
        return 0;
    }

    private static List<int> TryGetIntArray(JsonElement root, string name)
    {
        var list = new List<int>();
        if (!root.TryGetProperty(name, out var prop) || prop.ValueKind != JsonValueKind.Array)
            return list;

        foreach (var el in prop.EnumerateArray())
        {
            if (el.ValueKind == JsonValueKind.Number && el.TryGetInt32(out var v))
                list.Add(v);
        }

        return list;
    }

    private static string? TryGetString(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var prop))
            return null;
        if (prop.ValueKind == JsonValueKind.String)
            return prop.GetString();
        return null;
    }
}
