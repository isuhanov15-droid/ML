using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using ML.Core;
using ML.Core.Layers;
using ML.Core.Optimizers;
using ML.Core.Serialization;
using ML.Shared.Protocol;

namespace ML.Host.Services;

public sealed partial class BrainMlService
{
    private static readonly JsonSerializerOptions CheckpointJson = new(JsonSerializerDefaults.Web);
    private sealed record ReplaySnapshot(int SchemaVersion, int InputDim, int ActionCount,
        int Capacity, int Count, int NextIndex, Transition[] Entries);
    private sealed record PersistentCheckpoint(long TrainSteps, string WeightsPath, int InputDim = 0,
        int ActionCount = 0, int Seed = 1337, double LearningRate = 0.0005, int SchemaVersion = 0,
        string? WeightsSha256 = null, string? TargetWeightsPath = null, string? TargetSha256 = null,
        string? ReplayPath = null, string? ReplaySha256 = null, int BufferCount = 0,
        int BufferCapacity = 1024, double LastLoss = 0, string ResumeMode = "weights-only");

    public MlCheckpointResponse Save(MlCheckpointRequest req)
    {
        lock (_lock)
        {
            var created = new List<string>();
            var committed = false;
            try
            {
                if (req.ExpectedInstanceId is not null && req.ExpectedInstanceId != _instanceId)
                    throw new InvalidOperationException("ML.Host instance changed");
                if (string.IsNullOrWhiteSpace(req.Path) || _net is null || _targetNet is null)
                    throw new InvalidOperationException("checkpoint path missing or model not initialized");
                var path = Path.GetFullPath(req.Path);
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                PersistentCheckpoint? previous = null;
                try { if (File.Exists(path)) previous = JsonSerializer.Deserialize<PersistentCheckpoint>(File.ReadAllText(path), CheckpointJson); }
                catch { /* Keep unreadable previous files for recovery. */ }
                var generation = Guid.NewGuid().ToString("N");
                var weights = path + ".weights-" + generation + ".net";
                var target = path + ".target-" + generation + ".net";
                var replay = path + ".replay-" + generation + ".json.gz";
                created.AddRange(new[] { weights, target, replay });
                ModelStore.SaveToFile(weights, _net); FlushFile(weights);
                ModelStore.SaveToFile(target, _targetNet); FlushFile(target);
                using (var file = new FileStream(replay, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    using (var zip = new GZipStream(file, CompressionLevel.Fastest, leaveOpen: true))
                        JsonSerializer.Serialize(zip, _buffer.Snapshot(_inputDim, _actionCount), CheckpointJson);
                    file.Flush(true);
                }
                var meta = new PersistentCheckpoint(_trainSteps, weights, _inputDim, _actionCount,
                    _seed, _lr, 2, Hash(weights), target, Hash(target), replay, Hash(replay),
                    _buffer.Count, _buffer.Capacity, _lastLoss, "weights-target-and-replay");
                var json = JsonSerializer.Serialize(meta, CheckpointJson);
                var temp = path + ".tmp-" + generation; created.Add(temp);
                File.WriteAllText(temp, json); FlushFile(temp);
                File.Move(temp, path, overwrite: true); committed = true;
                foreach (var (old, prefix) in new[] { (previous?.WeightsPath, ".weights-"),
                    (previous?.TargetWeightsPath, ".target-"), (previous?.ReplayPath, ".replay-") })
                    if (old is not null && old.StartsWith(path + prefix, StringComparison.Ordinal))
                        try { File.Delete(old); } catch { /* An old immutable generation is harmless. */ }
                _lastError = null;
                return new MlCheckpointResponse(true, json);
            }
            catch (Exception ex)
            {
                if (!committed) foreach (var file in created) try { File.Delete(file); } catch { }
                _lastError = ex.Message;
                return new MlCheckpointResponse(false, ex.Message);
            }
        }
    }

    public MlCheckpointResponse Load(MlCheckpointRequest req)
    {
        lock (_lock)
        {
            try
            {
                var json = File.ReadAllText(req.Path);
                var meta = JsonSerializer.Deserialize<PersistentCheckpoint>(json, CheckpointJson)
                    ?? throw new InvalidDataException("checkpoint metadata missing");
                if (meta.SchemaVersion is not (0 or 2) || meta.TrainSteps < 0 ||
                    !double.IsFinite(meta.LearningRate) || meta.LearningRate <= 0 || !double.IsFinite(meta.LastLoss))
                    throw new InvalidDataException("invalid checkpoint metadata");
                if (meta.SchemaVersion == 0 && (meta.ReplayPath is not null || meta.TargetWeightsPath is not null ||
                    meta.ResumeMode != "weights-only"))
                    throw new InvalidDataException("replay checkpoint is missing its schema version");
                if (meta.SchemaVersion == 2) VerifyHash(meta.WeightsPath, meta.WeightsSha256);
                var net = ModelStore.LoadFromFile(meta.WeightsPath);
                var linear = net.Layers.OfType<LinearLayer>().ToArray();
                var inputDim = linear.First().InputSize; var actionCount = linear.Last().OutputSize;
                if ((meta.InputDim != 0 && meta.InputDim != inputDim) || (meta.ActionCount != 0 && meta.ActionCount != actionCount))
                    throw new InvalidDataException("checkpoint model dimensions mismatch");
                var target = ModelStore.LoadFromFile(meta.WeightsPath);
                var buffer = new ExperienceBuffer(1024);
                if (meta.SchemaVersion == 2)
                {
                    VerifyHash(meta.TargetWeightsPath, meta.TargetSha256);
                    VerifyHash(meta.ReplayPath, meta.ReplaySha256);
                    target = ModelStore.LoadFromFile(meta.TargetWeightsPath!);
                    using var file = File.OpenRead(meta.ReplayPath!);
                    using var zip = new GZipStream(file, CompressionMode.Decompress);
                    var replay = JsonSerializer.Deserialize<ReplaySnapshot>(zip, CheckpointJson)
                        ?? throw new InvalidDataException("empty replay snapshot");
                    ValidateReplay(replay, inputDim, actionCount);
                    if (replay.Count != meta.BufferCount || replay.Capacity != meta.BufferCapacity)
                        throw new InvalidDataException("replay metadata mismatch");
                    buffer = ExperienceBuffer.Restore(replay);
                }
                var parameters = net.Parameters().ToArray(); var targetParameters = target.Parameters().ToArray();
                if (!parameters.Select(p => p.Value.Length).SequenceEqual(targetParameters.Select(p => p.Value.Length)) ||
                    parameters.Concat(targetParameters).Any(p => p.Value.Any(v => !double.IsFinite(v))))
                    throw new InvalidDataException("invalid checkpoint network parameters");
                // Publish the new state only after every component validates. Never fall back
                // to an empty replay when a version 2 checkpoint is damaged or incomplete.
                _net = net; _targetNet = target; _buffer = buffer;
                _inputDim = inputDim; _actionCount = actionCount; _seed = meta.Seed; _lr = meta.LearningRate;
                _optimizer = new AdamOptimizer(_lr); _rng = new Random(_seed);
                _trainSteps = meta.TrainSteps; _lastLoss = meta.LastLoss; _lastError = null;
                return new MlCheckpointResponse(true, json);
            }
            catch (Exception ex)
            {
                _lastError = ex.Message;
                return new MlCheckpointResponse(false, ex.Message);
            }
        }
    }

    private static void ValidateReplay(ReplaySnapshot replay, int inputDim, int actionCount)
    {
        if (replay.SchemaVersion != 1 || replay.InputDim != inputDim || replay.ActionCount != actionCount ||
            replay.Capacity < 1 || replay.Capacity > 1_000_000 || replay.Count < 0 || replay.Count > replay.Capacity ||
            replay.NextIndex < 0 || replay.NextIndex >= replay.Capacity ||
            (replay.Count < replay.Capacity && replay.NextIndex != replay.Count) ||
            replay.Entries is null || replay.Entries.Length != replay.Count)
            throw new InvalidDataException("invalid replay header");
        foreach (var t in replay.Entries)
            if (t.State is null || t.NextState is null || t.State.Length != inputDim || t.NextState.Length != inputDim ||
                t.Action < 0 || t.Action >= actionCount || !float.IsFinite(t.Reward) ||
                t.State.Any(v => !float.IsFinite(v)) || t.NextState.Any(v => !float.IsFinite(v)) ||
                (t.NextActionMask is not null && (t.NextActionMask.Length != actionCount || t.NextActionMask.Any(v => v is not (0f or 1f)))))
                throw new InvalidDataException("invalid replay transition");
    }

    private static string Hash(string path) { using var file = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(file)); }
    private static void VerifyHash(string? path, string? expected)
    {
        if (string.IsNullOrWhiteSpace(path) || string.IsNullOrWhiteSpace(expected) || !Hash(path).Equals(expected, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("checkpoint component missing or checksum mismatch");
    }
    private static void FlushFile(string path) { using var file = new FileStream(path, FileMode.Open, FileAccess.ReadWrite); file.Flush(true); }
}
