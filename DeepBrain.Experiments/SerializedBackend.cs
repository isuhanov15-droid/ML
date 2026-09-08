using System.Linq;
using System.Text.Json;
using DeepBrain.Shared.MlBridge;

using DeepBrain.Host.BrainLife;
using DeepBrain.Host.BrainLife.Ml;

namespace DeepBrain.Experiments;

public sealed class SerializedBackend : IBrainMlBackend
{
    public double[] LastQ { get; set; } = Array.Empty<double>();
    public long TrainCalls => _client.TrainCalls;
    public double MaxAbsQ => _client.MaxAbsQ;
    public double MaxLoss => _client.MaxLoss;
    private readonly Action<string> _log;
    private ServiceBridge _client;
    private readonly Queue<double> _lossWindow = new(100);
    private DateTime _lastErrorLog = DateTime.MinValue;
    private double _avgQ;
    private double _lastLoss;
    private long _trainSteps;
    private int _bufferSize;
    private string? _checkpointError;
    private string? _trainError;

    public SerializedBackend(MlConfig config, Action<string> log)
    {
        _log = log;
        _client = new ServiceBridge(config.Remote, log);
    }

    public bool IsAvailable => true;
    public string Kind => "remote";
    public bool IsConnected => _client.IsConnected;
    public string? LastError => _checkpointError ?? _trainError ?? _client.LastError;
    public double LastRttMs => _client.LastRttMs;
    public int InputDim => StateVectorizer.InputDim;
    public int ActionCount => ActionCatalog.Count;
    public int BufferSize => _bufferSize;
    public int BufferCapacity { get; private set; }
    public double LastLoss => _lastLoss;
    public double AvgLoss100 => AverageLoss();
    public long TrainSteps => _trainSteps;
    public double AvgQ => _avgQ;

    public async Task<MlInferResult> InferAsync(float[] stateVec, float[] actionMask, MlConfig config, CancellationToken ct)
    {
        BufferCapacity = Math.Max(1024, config.BufferSize);
        var req = new MlInferRequest(
            State: stateVec,
            ActionMask: config.ActionMasking ? ToBoolMask(actionMask) : null,
            InputDim: StateVectorizer.InputDim,
            ActionCount: ActionCatalog.Count
        ) { Seed = config.Seed, LearningRate = config.LearningRate };

        var resp = await _client.CallAsync<MlInferRequest, MlInferResponse>("ml.infer", req, ct);
        if (resp == null)
        {
            RateLimitedLog($"Предупреждение: ошибка ml.infer: {_client.LastError}");
            return new MlInferResult(false, Array.Empty<double>(), null, 0, 0);
        }

        LastQ = resp.QValues.ToArray();
        _avgQ = resp.AvgQ;
        return new MlInferResult(true, resp.QValues, resp.Probabilities, resp.Entropy, resp.AvgQ);
    }

    public async Task<MlTrainResult> TrainAsync(Transition transition, MlConfig config, long tick, CancellationToken ct)
    {
        BufferCapacity = Math.Max(1024, config.BufferSize);
        var req = new MlTrainRequest(
            Tick: tick,
            Transition: new MlTransitionDto(
                transition.State,
                transition.Action,
                transition.Reward,
                transition.NextState,
                transition.Done,
                ToBoolMask(transition.ActionMask)
            ),
            Config: new MlTrainConfigDto(
                BufferSize: BufferCapacity,
                BatchSize: config.BatchSize,
                TrainStepsPerBatch: config.TrainStepsPerBatch,
                TrainEveryTicks: config.TrainEveryTicks,
                TargetUpdateTicks: config.TargetUpdateTicks,
                Gamma: config.Gamma,
                GradClip: config.GradClip,
                Seed: config.Seed
            ),
            InputDim: StateVectorizer.InputDim,
            ActionCount: ActionCatalog.Count
        );

        var resp = await _client.CallAsync<MlTrainRequest, MlTrainResponse>("ml.train", req, ct);
        if (resp == null)
        {
            RateLimitedLog($"Предупреждение: ошибка ml.train: {_client.LastError}");
            return new MlTrainResult(false, double.NaN, 0, _trainSteps, true);
        }

        _bufferSize = Math.Max(0, resp.BufferSize);
        _trainSteps = resp.TrainSteps;
        _lastLoss = resp.Loss;
        var isNaN = double.IsNaN(resp.Loss) || double.IsInfinity(resp.Loss);

        if (!resp.Ok)
        {
            _trainError = string.IsNullOrWhiteSpace(resp.Reason) ? "ml.train returned ok=false" : resp.Reason;
            RateLimitedLog($"Предупреждение: ошибка ml.train: {_trainError}");
            return new MlTrainResult(false, resp.Loss, resp.GradNorm, resp.TrainSteps, isNaN);
        }

        _trainError = null;
        if (resp.Trained && !isNaN)
            PushLoss(resp.Loss);
        return new MlTrainResult(resp.Trained, resp.Loss, resp.GradNorm, resp.TrainSteps, isNaN);
    }

    public bool TryLoad(string path, out int episodeId)
    {
        episodeId = 0;
        var req = new MlCheckpointRequest(path);
        var resp = _client.CallAsync<MlCheckpointRequest, MlCheckpointResponse>("ml.checkpoint.load", req, CancellationToken.None).GetAwaiter().GetResult();
        if (resp is null)
        {
            _checkpointError = _client.LastError ?? "checkpoint load: no response";
            _log($"Предупреждение: ML checkpoint не загружен: {_checkpointError}");
            return false;
        }

        if (!resp.Ok)
        {
            _checkpointError = string.IsNullOrWhiteSpace(resp.Meta) ? "load failed" : resp.Meta;
            _log($"Предупреждение: ML checkpoint не загружен: {_checkpointError}");
            return false;
        }

        _checkpointError = null;
        if (!string.IsNullOrWhiteSpace(resp.Meta))
        {
            try
            {
                var meta = JsonSerializer.Deserialize<RemoteCheckpointMeta>(
                    resp.Meta,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                if (meta is not null)
                    _trainSteps = Math.Max(_trainSteps, meta.TrainSteps);
            }
            catch (JsonException ex)
            {
                _log($"Предупреждение: ответ ML checkpoint не содержит корректные метаданные: {ex.Message}");
            }
        }

        return true;
    }

    public void TrySave(string path, int episodeId)
    {
        var req = new MlCheckpointRequest(path);
        var resp = _client.CallAsync<MlCheckpointRequest, MlCheckpointResponse>("ml.checkpoint.save", req, CancellationToken.None).GetAwaiter().GetResult();
        if (resp is null || !resp.Ok)
        {
            _checkpointError = resp?.Meta ?? _client.LastError ?? "checkpoint save: no response";
            _log($"Предупреждение: ML checkpoint не сохранён: {_checkpointError}");
            throw new IOException(_checkpointError);
        }

        _checkpointError = null;
    }

    public void Reset(MlConfig config)
    {
        _client.UpdateConfig(config.Remote);
        var response = _client.CallAsync<MlResetRequest, MlCheckpointResponse>("ml.reset",
            new MlResetRequest(config.Seed, config.LearningRate), CancellationToken.None).GetAwaiter().GetResult();
        if (response?.Ok != true) throw new IOException("ML reset failed: " + response?.Meta);
        _lossWindow.Clear();
        _lastLoss = 0;
        _avgQ = 0;
        _trainSteps = 0;
        _bufferSize = 0;
        _checkpointError = null;
        _trainError = null;
        BufferCapacity = Math.Max(1024, config.BufferSize);
        _client.UpdateConfig(config.Remote);
    }

    public void ResetCounters()
    {
        _lossWindow.Clear();
    }

    public void UpdateConfig(MlConfig config)
    {
        BufferCapacity = Math.Max(1024, config.BufferSize);
        _client.UpdateConfig(config.Remote);
    }

    public bool TryConnect() => _client.TryConnect();
    public void Disconnect() => _client.Disconnect();
    public ValueTask DisposeAsync() => _client.DisposeAsync();

    private void PushLoss(double loss)
    {
        if (_lossWindow.Count >= 100) _lossWindow.Dequeue();
        _lossWindow.Enqueue(loss);
    }

    private double AverageLoss()
    {
        if (_lossWindow.Count == 0) return 0;
        return _lossWindow.Average();
    }

    private sealed record RemoteCheckpointMeta(long TrainSteps, string? WeightsPath);

    private void RateLimitedLog(string message)
    {
        var now = DateTime.UtcNow;
        if ((now - _lastErrorLog).TotalSeconds < 5)
            return;
        _lastErrorLog = now;
        _log(message);
    }

    private static bool[]? ToBoolMask(float[]? mask)
    {
        if (mask == null) return null;
        var arr = new bool[mask.Length];
        for (var i = 0; i < mask.Length; i++)
            arr[i] = mask[i] > 0.5f;
        return arr;
    }
}

// Uses the real DeepBrain JSON DTOs and real ML.Host service, without sockets.
// Service metadata is separate from the advisor wrapper even in one process.
internal sealed class ServiceBridge
{
    private readonly ML.Host.Services.BrainMlService _service = new();
    private readonly JsonSerializerOptions _json = new(JsonSerializerDefaults.Web) { PropertyNameCaseInsensitive = true };
    public long TrainCalls { get; private set; }
    public double MaxAbsQ { get; private set; }
    public double MaxLoss { get; private set; }
    public bool IsConnected => true;
    public string? LastError => null;
    public double LastRttMs => 0;
    public ServiceBridge(MlRemoteConfig config, Action<string> log) { }
    public void UpdateConfig(MlRemoteConfig config) { }
    public bool TryConnect() => true;
    public void Disconnect() { }
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    public Task<TResponse?> CallAsync<TRequest,TResponse>(string method, TRequest req, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var wire = JsonSerializer.Serialize(req, _json);
        object response;
        switch (method)
        {
            case "ml.reset":
                response = _service.Reset(JsonSerializer.Deserialize<ML.Shared.Protocol.MlResetRequest>(wire, _json)!);
                break;
            case "ml.infer":
                var infer = _service.Infer(JsonSerializer.Deserialize<ML.Shared.Protocol.MlInferRequest>(wire, _json)!);
                if (!infer.Ok || infer.QValues.Any(v => !double.IsFinite(v))) throw new InvalidOperationException("ml.infer failed");
                MaxAbsQ = Math.Max(MaxAbsQ, infer.QValues.Select(Math.Abs).DefaultIfEmpty().Max());
                response = infer;
                break;
            case "ml.train":
                TrainCalls++;
                var train = _service.Train(JsonSerializer.Deserialize<ML.Shared.Protocol.MlTrainRequest>(wire, _json)!);
                if (!train.Ok || !double.IsFinite(train.Loss)) throw new InvalidOperationException("ml.train failed: " + train.Reason);
                MaxLoss = Math.Max(MaxLoss, train.Loss);
                response = train;
                break;
            case "ml.checkpoint.save":
            case "ml.checkpoint.load":
                var checkpoint = JsonSerializer.Deserialize<ML.Shared.Protocol.MlCheckpointRequest>(wire, _json)!;
                checkpoint = checkpoint with { Path = checkpoint.Path + ".service" };
                var saved = method.EndsWith("save") ? _service.Save(checkpoint) : _service.Load(checkpoint);
                if (!saved.Ok) throw new InvalidOperationException(method + ": " + saved.Meta);
                response = saved;
                break;
            default: throw new NotSupportedException(method);
        }
        return Task.FromResult(JsonSerializer.Deserialize<TResponse>(JsonSerializer.Serialize(response, _json), _json));
    }
}
