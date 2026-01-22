using ML.Core;
using ML.Core.Layers;
using ML.Core.Serialization;
using ML.Shared.Protocol;

namespace ML.Host.Services;

public sealed class InferenceService
{
    public sealed record LoadedModel(
        string ModelId,
        Network Network,
        string? ProjectId,
        string? RunId,
        string? ModelPath,
        string TaskType,
        int InputSize,
        int OutputSize,
        string[]? Labels);

    private readonly object _lock = new();
    private readonly Dictionary<string, LoadedModel> _models = new();
    private readonly Dictionary<string, string> _projectLastModel = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, string> _runLastModel = new(StringComparer.OrdinalIgnoreCase);
    private string? _defaultModelId;

    public void Load(string? modelName, string? filePath)
    {
        var path = !string.IsNullOrWhiteSpace(filePath)
            ? filePath
            : null;

        Network network = path != null
            ? ModelStore.LoadFromFile(path)
            : ModelStore.Load(modelName ?? throw new InvalidOperationException("modelName required"));

        var (inputSize, outputSize) = GetSizes(network);
        var taskType = GetTaskType(outputSize);
        var modelId = Guid.NewGuid().ToString("N");

        lock (_lock)
        {
            _models[modelId] = new LoadedModel(
                modelId,
                network,
                ProjectId: null,
                RunId: null,
                ModelPath: path,
                TaskType: taskType,
                InputSize: inputSize,
                OutputSize: outputSize,
                Labels: null);
            _defaultModelId = modelId;
        }
    }

    public double[] Predict(double[] input)
    {
        string? modelId;
        lock (_lock)
            modelId = _defaultModelId;

        if (string.IsNullOrWhiteSpace(modelId))
            throw new InvalidOperationException("Model not loaded.");

        return Predict(modelId, input);
    }

    public ModelLoadResponse LoadFromPath(string modelPath, string? projectId, string? runId)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new InvalidOperationException("modelPath required.");

        var network = ModelStore.LoadFromFile(modelPath);
        var (inputSize, outputSize) = GetSizes(network);
        var taskType = GetTaskType(outputSize);
        var modelId = Guid.NewGuid().ToString("N");

        lock (_lock)
        {
            _models[modelId] = new LoadedModel(
                modelId,
                network,
                ProjectId: projectId,
                RunId: runId,
                ModelPath: modelPath,
                TaskType: taskType,
                InputSize: inputSize,
                OutputSize: outputSize,
                Labels: null);

            if (!string.IsNullOrWhiteSpace(projectId))
                _projectLastModel[projectId] = modelPath;
            if (!string.IsNullOrWhiteSpace(runId))
                _runLastModel[runId] = modelPath;
        }

        return new ModelLoadResponse(
            ModelId: modelId,
            TaskType: taskType,
            InputSize: inputSize,
            OutputSize: outputSize,
            Labels: null);
    }

    public LoadedModel GetModel(string modelId)
    {
        lock (_lock)
        {
            if (!_models.TryGetValue(modelId, out var model))
                throw new InvalidOperationException("modelId not found.");
            return model;
        }
    }

    public double[] Predict(string modelId, double[] input)
    {
        var model = GetModel(modelId);
        return Predict(model, input);
    }

    public double[] Predict(LoadedModel model, double[] input)
    {
        lock (_lock)
        {
            return model.Network.Forward(input, training: false);
        }
    }

    public bool TryGetLastProjectModelPath(string projectId, out string? modelPath)
    {
        lock (_lock)
            return _projectLastModel.TryGetValue(projectId, out modelPath);
    }

    public bool TryGetLastRunModelPath(string runId, out string? modelPath)
    {
        lock (_lock)
            return _runLastModel.TryGetValue(runId, out modelPath);
    }

    public void RegisterModelPath(string? projectId, string? runId, string? modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            return;

        lock (_lock)
        {
            if (!string.IsNullOrWhiteSpace(projectId))
                _projectLastModel[projectId] = modelPath;
            if (!string.IsNullOrWhiteSpace(runId))
                _runLastModel[runId] = modelPath;
        }
    }

    private static (int inputSize, int outputSize) GetSizes(Network network)
    {
        int inputSize = 0;
        int outputSize = 0;
        foreach (var layer in network.Layers)
        {
            switch (layer)
            {
                case LinearLayer linear:
                    if (inputSize == 0)
                        inputSize = linear.InputSize;
                    outputSize = linear.OutputSize;
                    break;
                case NeuronLayer neuron:
                    if (inputSize == 0)
                        inputSize = neuron.InputSize;
                    outputSize = neuron.OutputSize;
                    break;
            }
        }

        return (inputSize, outputSize);
    }

    private static string GetTaskType(int outputSize)
    {
        if (outputSize <= 0)
            return "unknown";
        if (outputSize == 1)
            return "regression";
        return "classification";
    }
}
