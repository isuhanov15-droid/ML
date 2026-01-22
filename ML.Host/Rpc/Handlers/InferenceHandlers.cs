using System.Globalization;
using System.Text.Json;
using ML.Host.Services;
using ML.Host.Storage;
using ML.Host.Rpc;
using ML.Shared.Protocol;

namespace ML.Host.Rpc.Handlers;

public sealed class InferenceHandlers
{
    private readonly InferenceService _inference;
    private readonly ProjectStore _projects;
    private readonly EventBus _events;

    public InferenceHandlers(InferenceService inference, ProjectStore projects, EventBus events)
    {
        _inference = inference;
        _projects = projects;
        _events = events;
    }

    public Task<object?> LoadAsync(RpcRequest req)
    {
        try
        {
            var request = Deserialize<ModelLoadRequest>(req);
            string? modelPath = ResolveModelPath(request);
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new InvalidOperationException("Model path not resolved.");
            if (!File.Exists(modelPath))
                throw new InvalidOperationException($"Model file not found: {modelPath}");

            var response = _inference.LoadFromPath(modelPath, request.ProjectId, request.RunId);
            LogInfo($"model.loaded: {modelPath}");
            return Task.FromResult<object?>(response);
        }
        catch (Exception ex)
        {
            LogError($"model.load.error: {ex.Message}");
            throw;
        }
    }

    public Task<object?> InferSingleAsync(RpcRequest req)
    {
        try
        {
            var request = Deserialize<InferSingleRequest>(req);
            var output = _inference.Predict(request.ModelId, request.Input);
            if (output.Length > 1)
            {
                var probabilities = Softmax(output);
                int predicted = ArgMax(probabilities);
                return Task.FromResult<object?>(new InferSingleResponse(output, predicted, probabilities));
            }

            return Task.FromResult<object?>(new InferSingleResponse(output, null, null));
        }
        catch (Exception ex)
        {
            LogError($"infer.single.error: {ex.Message}");
            throw;
        }
    }

    public Task<object?> InferBatchAsync(RpcRequest req)
    {
        try
        {
            var request = Deserialize<InferBatchRequest>(req);
            if (request.Inputs.Length > 50000)
                throw new InvalidOperationException("inputs limit exceeded (max 50000).");

            var model = _inference.GetModel(request.ModelId);
            int total = request.Inputs.Length;
            var outputs = new double[total][];
            double[][]? probabilities = model.OutputSize > 1 ? new double[total][] : null;
            int[]? predicted = model.OutputSize > 1 ? new int[total] : null;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            const int chunkSize = 1024;
            for (int offset = 0; offset < total; offset += chunkSize)
            {
                int count = Math.Min(chunkSize, total - offset);
                for (int i = 0; i < count; i++)
                {
                    int idx = offset + i;
                    var input = request.Inputs[idx];
                    var output = _inference.Predict(model, input);
                    outputs[idx] = output;

                    if (model.OutputSize > 1)
                    {
                        var probs = Softmax(output);
                        probabilities![idx] = probs;
                        predicted![idx] = ArgMax(probs);
                    }
                }
            }
            sw.Stop();

            LogInfo($"infer.batch: n={total} ms={sw.ElapsedMilliseconds}");
            var response = new InferBatchResponse(outputs, probabilities, predicted);
            return Task.FromResult<object?>(response);
        }
        catch (Exception ex)
        {
            LogError($"infer.batch.error: {ex.Message}");
            throw;
        }
    }

    public Task<object?> EvalDatasetAsync(RpcRequest req)
    {
        try
        {
            var request = Deserialize<EvalDatasetRequest>(req);
            if (string.IsNullOrWhiteSpace(request.DatasetPath))
                throw new InvalidOperationException("datasetPath required.");
            if (!File.Exists(request.DatasetPath))
                throw new InvalidOperationException($"Dataset not found: {request.DatasetPath}");

            var rows = ReadCsv(request.DatasetPath, request.HasHeader);
            int correct = 0;
            int total = 0;
            var samples = new List<EvalSampleDto>();

            foreach (var row in rows)
            {
                if (!TryBuildInput(row, request.InputCols, out var input))
                    continue;
                if (!TryGetLabel(row, request.LabelCol, out var label))
                    continue;

                var output = _inference.Predict(request.ModelId, input);
                int predicted = output.Length > 1 ? ArgMax(output) : (int)Math.Round(output[0]);
                if (predicted == label)
                    correct++;
                total++;

                if (samples.Count < 10)
                {
                    samples.Add(new EvalSampleDto(
                        Input: input,
                        Expected: new[] { (double)label },
                        Predicted: output));
                }
            }

            double? accuracy = total == 0 ? null : (double)correct / total;
            var response = new EvalDatasetResponse(accuracy, Loss: null, samples.ToArray());
            return Task.FromResult<object?>(response);
        }
        catch (Exception ex)
        {
            LogError($"eval.dataset.error: {ex.Message}");
            throw;
        }
    }

    private static T Deserialize<T>(RpcRequest req)
    {
        if (req.@params is not JsonElement el)
            throw new InvalidOperationException("Invalid payload.");

        var value = JsonSerializer.Deserialize<T>(el.GetRawText(), ML.Shared.Protocol.JsonOptions.Default);
        if (value == null)
            throw new InvalidOperationException("Invalid payload.");
        return value;
    }

    private string? ResolveModelPath(ModelLoadRequest request)
    {
        if (!string.IsNullOrWhiteSpace(request.ModelPath))
            return request.ModelPath;

        if (!string.IsNullOrWhiteSpace(request.RunId)
            && _inference.TryGetLastRunModelPath(request.RunId, out var runPath)
            && !string.IsNullOrWhiteSpace(runPath))
            return runPath;

        if (!string.IsNullOrWhiteSpace(request.ProjectId)
            && _inference.TryGetLastProjectModelPath(request.ProjectId, out var projectPath)
            && !string.IsNullOrWhiteSpace(projectPath))
            return projectPath;

        if (!string.IsNullOrWhiteSpace(request.ProjectId)
            && _projects.TryGetProjectModelPath(request.ProjectId, out var defaultPath)
            && !string.IsNullOrWhiteSpace(defaultPath))
            return defaultPath;

        return null;
    }

    private static double[] Softmax(double[] values)
    {
        double max = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > max) max = values[i];

        var exp = new double[values.Length];
        double sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            exp[i] = Math.Exp(values[i] - max);
            sum += exp[i];
        }

        if (sum == 0)
            return exp;

        for (int i = 0; i < exp.Length; i++)
            exp[i] /= sum;

        return exp;
    }

    private static int ArgMax(double[] values)
    {
        int idx = 0;
        double max = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > max)
            {
                max = values[i];
                idx = i;
            }
        }
        return idx;
    }

    private static IEnumerable<string[]> ReadCsv(string path, bool hasHeader)
    {
        bool first = true;
        foreach (var line in File.ReadLines(path))
        {
            if (first && hasHeader)
            {
                first = false;
                continue;
            }
            first = false;

            var parts = line.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length == 0)
                continue;
            yield return parts;
        }
    }

    private static bool TryBuildInput(string[] row, int[] inputCols, out double[] input)
    {
        input = Array.Empty<double>();
        if (inputCols.Length == 0)
            return false;

        var values = new double[inputCols.Length];
        for (int i = 0; i < inputCols.Length; i++)
        {
            int col = inputCols[i];
            if (col < 0 || col >= row.Length)
                return false;
            if (!double.TryParse(row[col], NumberStyles.Float, CultureInfo.InvariantCulture, out var v)
                && !double.TryParse(row[col], NumberStyles.Float, CultureInfo.CurrentCulture, out v))
                return false;
            values[i] = v;
        }

        input = values;
        return true;
    }

    private static bool TryGetLabel(string[] row, int labelCol, out int label)
    {
        label = 0;
        if (labelCol < 0 || labelCol >= row.Length)
            return false;
        if (!int.TryParse(row[labelCol], NumberStyles.Integer, CultureInfo.InvariantCulture, out label)
            && !int.TryParse(row[labelCol], NumberStyles.Integer, CultureInfo.CurrentCulture, out label))
            return false;
        return true;
    }

    private void LogInfo(string message)
    {
        var evt = new RpcEvent(
            v: Protocol.ProtocolVersion,
            type: "log.append",
            data: new { level = "info", message, utc = DateTimeOffset.UtcNow });
        _ = _events.BroadcastAsync(evt);
    }

    private void LogError(string message)
    {
        var evt = new RpcEvent(
            v: Protocol.ProtocolVersion,
            type: "log.append",
            data: new { level = "error", message, utc = DateTimeOffset.UtcNow });
        _ = _events.BroadcastAsync(evt);
    }
}
