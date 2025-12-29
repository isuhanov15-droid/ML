using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using ML.Core.Layers;

namespace ML.Core.Serialization;

public static class ModelStore
{
    // В репо: ML/Models
    // Если запускаешь из bin/Debug, надо подняться вверх.
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public static void Save(string modelName, Network network)
    {
        if (string.IsNullOrWhiteSpace(modelName)) throw new ArgumentException("modelName is empty");
        if (network == null) throw new ArgumentNullException(nameof(network));

        var dto = new NetworkDto();

        foreach (var layer in network.Layers)
        {
            switch (layer)
            {
                case LinearLayer lin:
                    dto.Layers.Add(new LayerDto
                    {
                        Type = "Linear",
                        InputSize = lin.InputSize,
                        OutputSize = lin.OutputSize,
                        Weights = lin.Weights.ToArray(),
                        Bias = lin.Bias.ToArray()
                    });
                    break;

                case ActivationLayer act:
                    dto.Layers.Add(new LayerDto
                    {
                        Type = "Activation",
                        Size = act.Size,
                        Activation = act.Type.ToString()
                    });
                    break;

                case SoftmaxLayer sm:
                    dto.Layers.Add(new LayerDto
                    {
                        Type = "Softmax",
                        Size = sm.Size
                    });
                    break;

                default:
                    throw new InvalidOperationException($"Unsupported layer for serialization: {layer.GetType().Name}");
            }
        }

        string dir = ResolveModelsDir();
        Directory.CreateDirectory(dir);

        string path = System.IO.Path.Combine(dir, $"{modelName}.json");
        File.WriteAllText(path, JsonSerializer.Serialize(dto, JsonOptions));
    }

    public static Network Load(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName)) throw new ArgumentException("modelName is empty");

        string dir = ResolveModelsDir();
        string path = System.IO.Path.Combine(dir, $"{modelName}.json");

        if (!File.Exists(path))
            throw new FileNotFoundException($"Model file not found: {path}");

        var dto = JsonSerializer.Deserialize<NetworkDto>(File.ReadAllText(path))
                  ?? throw new InvalidOperationException("Invalid model json.");

        var net = new Network();

        int layerIndex = 0;
        foreach (var ld in dto.Layers)
        {
            layerIndex++;

            switch (ld.Type)
            {
                case "Linear":
                    if (ld.InputSize <= 0 || ld.OutputSize <= 0)
                        throw new InvalidOperationException($"Linear sizes invalid at layer #{layerIndex}");

                    var lin = new LinearLayer(ld.InputSize, ld.OutputSize, seed: 123);
                    if (ld.Weights == null || ld.Bias == null)
                        throw new InvalidOperationException($"Linear weights/bias missing at layer #{layerIndex}");

                    if (ld.Weights.Length != lin.Weights.Length)
                        throw new InvalidOperationException($"Weights length mismatch at layer #{layerIndex}");

                    if (ld.Bias.Length != lin.Bias.Length)
                        throw new InvalidOperationException($"Bias length mismatch at layer #{layerIndex}");

                    Array.Copy(ld.Weights, lin.Weights, lin.Weights.Length);
                    Array.Copy(ld.Bias, lin.Bias, lin.Bias.Length);

                    // grads оставить нулями
                    net.Add(lin);
                    break;

                case "Activation":
                    if (ld.Size <= 0) throw new InvalidOperationException($"Activation size invalid at layer #{layerIndex}");
                    if (string.IsNullOrWhiteSpace(ld.Activation))
                        throw new InvalidOperationException($"Activation type missing at layer #{layerIndex}");

                    var type = Enum.Parse<ActivationType>(ld.Activation, ignoreCase: true);
                    net.Add(new ActivationLayer(ld.Size, type));
                    break;

                case "Softmax":
                    if (ld.Size <= 0) throw new InvalidOperationException($"Softmax size invalid at layer #{layerIndex}");
                    net.Add(new SoftmaxLayer(ld.Size));
                    break;

                default:
                    throw new InvalidOperationException($"Unknown layer type '{ld.Type}' at layer #{layerIndex}");
            }
        }

        return net;
    }

    private static string ResolveModelsDir()
    {
        // current = .../bin/Debug/netX
        // поднимаемся вверх, пока не найдём папку ML
        var dir = AppContext.BaseDirectory;

        for (int i = 0; i < 8; i++)
        {
            var candidate = System.IO.Path.Combine(dir, "ML", "Models");
            if (Directory.Exists(System.IO.Path.Combine(dir, "ML")) || Directory.Exists(candidate))
                return candidate;

            var parent = Directory.GetParent(dir);
            if (parent == null) break;
            dir = parent.FullName;
        }

        // fallback: относительный путь от текущей директории
        return System.IO.Path.Combine(Directory.GetCurrentDirectory(), "ML", "Models");
    }

    // DTO
    private sealed class NetworkDto
    {
        public List<LayerDto> Layers { get; set; } = new();
    }

    private sealed class LayerDto
    {
        public string Type { get; set; } = "";

        // Linear
        public int InputSize { get; set; }
        public int OutputSize { get; set; }
        public double[]? Weights { get; set; }
        public double[]? Bias { get; set; }

        // Activation/Softmax
        public int Size { get; set; }
        public string? Activation { get; set; }
    }
}
