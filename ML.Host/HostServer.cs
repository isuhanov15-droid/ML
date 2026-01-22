using System.Diagnostics;
using System.Net;
using System.Text.Json;
using System.Linq;
using ML.Core;
using ML.Host.Net;
using ML.Host.Services;
using ML.Shared.Protocol.Training;
using ProtocolJson = ML.Shared.Protocol.ProtocolJson;
using ProtocolTypes = ML.Shared.Protocol.ProtocolTypes;

namespace ML.Host;

public sealed class HostServer : IAsyncDisposable
{
    private readonly TcpHost _tcp;
    private readonly TrainingService _training;
    private readonly InferenceService _inference;
    private readonly ModelStoreService _modelStore;

    private static readonly JsonSerializerOptions JsonOptions = ProtocolJson.Options;

    public HostServer(IPAddress address, int port)
    {
        _tcp = new TcpHost(address, port);
        _training = new TrainingService();
        _inference = new InferenceService();
        _modelStore = new ModelStoreService();

        _tcp.ClientConnected += OnClientConnected;
        _tcp.ClientDisconnected += _ => { };

        _training.Log += msg =>
        {
            BroadcastAsync(ProtocolTypes.LogAppend, new { level = "info", message = msg }).Forget();
            BroadcastAsync("train.log", new { level = "info", message = msg }).Forget();
        };
        _training.StateChanged += OnTrainingStateChanged;
        _training.Metrics += metrics =>
        {
            var payload = new
            {
                runId = metrics.RunId,
                epoch = metrics.Epoch,
                loss = metrics.Loss,
                valLoss = metrics.ValLoss,
                acc = metrics.Acc,
                lr = metrics.LearningRate,
                elapsedMs = metrics.ElapsedMs
            };
            BroadcastAsync(ProtocolTypes.MetricsUpdate, payload).Forget();
            BroadcastAsync("train.metrics", payload).Forget();
        };
        _training.RunIdChanged += runId => BroadcastAsync("train.started", new { runId }).Forget();
    }

    public void Start() => _tcp.Start();

    private void OnClientConnected(ClientConnection conn)
    {
        conn.MessageReceived += OnMessage;
        BroadcastCoreHello(conn).Forget();
    }

    private void OnMessage(ClientConnection conn, ML.Host.Net.Envelope env)
    {
        var type = env.Type.ToLowerInvariant();
        switch (type)
        {
            case "ping":
                conn.SendAsync("pong", new { serverTimeUtc = DateTimeOffset.UtcNow }, env.Rid).Forget();
                break;

            case ProtocolTypes.HostInfo:
                HandleHostInfo(conn, env);
                break;

            case ProtocolTypes.HostCapabilities:
                HandleHostCapabilities(conn, env);
                break;

            case ProtocolTypes.TrainStart:
                HandleTrainStart(conn, env);
                break;

            case ProtocolTypes.TrainStop:
                HandleTrainStop(conn, env);
                break;

            case "model.save":
                HandleModelSave(conn, env);
                break;

            case "model.load":
                HandleModelLoad(conn, env);
                break;

            case "infer.load":
                HandleInferLoad(conn, env);
                break;

            case "infer.predict":
                HandleInferPredict(conn, env);
                break;

            default:
                conn.SendAsync("error", new { message = $"Unknown message type: {env.Type}" }, env.Rid).Forget();
                break;
        }
    }

    private void HandleTrainStart(ClientConnection conn, ML.Host.Net.Envelope env)
    {
        try
        {
            var config = DeserializeConfig(env.Data);
            _training.Start(config);
            _modelStore.SetCurrentNetwork(_training.CurrentNetwork);
            conn.SendAsync(ProtocolTypes.TrainStartResult, new { ok = true, runId = _training.CurrentRunId }, env.Rid).Forget();
        }
        catch (Exception ex)
        {
            conn.SendAsync(ProtocolTypes.TrainStartResult, new { ok = false, message = ex.Message }, env.Rid).Forget();
            conn.SendAsync("error", new { message = ex.Message }, env.Rid).Forget();
            BroadcastAsync("train.state", new { state = "error", message = ex.Message }).Forget();
            BroadcastAsync(ProtocolTypes.TrainStateChanged, new { state = "error", message = ex.Message, runId = _training.CurrentRunId }).Forget();
        }
    }

    private void HandleTrainStop(ClientConnection conn, ML.Host.Net.Envelope env)
    {
        _training.Stop();
        conn.SendAsync(ProtocolTypes.TrainStopResult, new { ok = true }, env.Rid).Forget();
    }

    private void HandleModelSave(ClientConnection conn, ML.Host.Net.Envelope env)
    {
        try
        {
            if (!TryGetString(env.Data, "modelName", out var modelName))
                modelName = null;
            if (!TryGetString(env.Data, "filePath", out var filePath))
                filePath = null;

            _modelStore.SetCurrentNetwork(_training.CurrentNetwork);
            _modelStore.Save(modelName, filePath);
            conn.SendAsync("model.save.result", new { ok = true }, env.Rid).Forget();
        }
        catch (Exception ex)
        {
            conn.SendAsync("model.save.result", new { ok = false, message = ex.Message }, env.Rid).Forget();
        }
    }

    private void HandleModelLoad(ClientConnection conn, ML.Host.Net.Envelope env)
    {
        try
        {
            if (!TryGetString(env.Data, "modelName", out var modelName))
                modelName = null;
            if (!TryGetString(env.Data, "filePath", out var filePath))
                filePath = null;

            _modelStore.Load(modelName, filePath);
            conn.SendAsync("model.load.result", new { ok = true }, env.Rid).Forget();
        }
        catch (Exception ex)
        {
            conn.SendAsync("model.load.result", new { ok = false, message = ex.Message }, env.Rid).Forget();
        }
    }

    private void HandleInferLoad(ClientConnection conn, ML.Host.Net.Envelope env)
    {
        try
        {
            if (!TryGetString(env.Data, "modelName", out var modelName))
                modelName = null;
            if (!TryGetString(env.Data, "filePath", out var filePath))
                filePath = null;

            _inference.Load(modelName, filePath);
            conn.SendAsync("infer.load.result", new { ok = true }, env.Rid).Forget();
        }
        catch (Exception ex)
        {
            conn.SendAsync("infer.load.result", new { ok = false, message = ex.Message }, env.Rid).Forget();
        }
    }

    private void HandleInferPredict(ClientConnection conn, ML.Host.Net.Envelope env)
    {
        try
        {
            if (!env.Data.TryGetProperty("input", out var inputEl) || inputEl.ValueKind != JsonValueKind.Array)
                throw new InvalidOperationException("input is required");

            var input = inputEl.EnumerateArray()
                .Where(x => x.ValueKind == JsonValueKind.Number)
                .Select(x => x.GetDouble())
                .ToArray();

            var sw = Stopwatch.StartNew();
            var output = _inference.Predict(input);
            sw.Stop();

            conn.SendAsync("infer.predict.result", new
            {
                ok = true,
                output,
                latencyMs = sw.ElapsedMilliseconds
            }, env.Rid).Forget();
        }
        catch (Exception ex)
        {
            conn.SendAsync("infer.predict.result", new { ok = false, message = ex.Message }, env.Rid).Forget();
        }
    }

    private void OnTrainingStateChanged(TrainingState state, string? message)
    {
        string stateStr = state.ToString().ToLowerInvariant();
        var payload = new { state = stateStr, message, runId = _training.CurrentRunId };
        BroadcastAsync(ProtocolTypes.TrainStateChanged, payload).Forget();
        BroadcastAsync("train.state", new { state = stateStr, message }).Forget();

        if (state == TrainingState.Stopped)
            BroadcastAsync("train.stopped", new { runId = _training.CurrentRunId }).Forget();

        if (state == TrainingState.Finished)
            BroadcastAsync("train.completed", new { runId = _training.CurrentRunId }).Forget();

        if (state == TrainingState.Error)
            BroadcastAsync("error", new { message = message ?? "train error" }).Forget();
    }

    private void HandleHostInfo(ClientConnection conn, ML.Host.Net.Envelope env)
    {
        var hostVersion = typeof(HostServer).Assembly.GetName().Version?.ToString() ?? "dev";
        var coreVersion = typeof(ML.Core.Network).Assembly.GetName().Version?.ToString() ?? "dev";
        var payload = new
        {
            ok = true,
            name = "ML.Host",
            version = hostVersion,
            coreVersion,
            protocolVersion = ProtocolTypes.Version,
            serverTimeUtc = DateTimeOffset.UtcNow
        };
        conn.SendAsync(ProtocolTypes.HostInfoResult, payload, env.Rid).Forget();
    }

    private void HandleHostCapabilities(ClientConnection conn, ML.Host.Net.Envelope env)
    {
        var payload = new
        {
            ok = true,
            protocolVersion = ProtocolTypes.Version,
            capabilities = new[] { "train", "infer", "model" }
        };
        conn.SendAsync(ProtocolTypes.HostCapabilitiesResult, payload, env.Rid).Forget();
    }
    private async Task BroadcastCoreHello(ClientConnection conn)
    {
        var version = typeof(ML.Core.Network).Assembly.GetName().Version?.ToString() ?? "dev";
        await conn.SendAsync("core.hello", new { version, capabilities = new[] { "train", "infer", "model" } });
    }

    private async Task BroadcastAsync(string type, object data)
    {
        foreach (var conn in _tcp.Clients)
        {
            try
            {
                await conn.SendAsync(type, data);
            }
            catch
            {
            }
        }
    }

    private static bool TryGetString(JsonElement data, string name, out string? value)
    {
        value = null;
        if (data.ValueKind != JsonValueKind.Object) return false;
        if (!data.TryGetProperty(name, out var el)) return false;
        value = el.GetString();
        return true;
    }

    private static TrainStartConfig DeserializeConfig(JsonElement data)
    {
        if (data.ValueKind == JsonValueKind.Object && data.TryGetProperty("config", out var cfgEl))
            return Deserialize<TrainStartConfig>(cfgEl);

        return Deserialize<TrainStartConfig>(data);
    }

    private static T Deserialize<T>(JsonElement element)
    {
        var value = JsonSerializer.Deserialize<T>(element.GetRawText(), JsonOptions);
        if (value == null)
            throw new InvalidOperationException("Invalid payload");
        return value;
    }

    public ValueTask DisposeAsync() => _tcp.DisposeAsync();
}

internal static class TaskExtensions
{
    public static void Forget(this Task task)
    {
        // Intentionally ignored.
    }
}
