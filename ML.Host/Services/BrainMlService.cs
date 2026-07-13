using System.Linq;
using ML.Core;
using ML.Core.Layers;
using ML.Core.Optimizers;
using ML.Shared.Protocol;

namespace ML.Host.Services;

public sealed class BrainMlService
{
    private readonly object _lock = new();
    private Network? _net;
    private Network? _targetNet;
    private AdamOptimizer? _optimizer;
    private ExperienceBuffer _buffer = new(1024);
    private Random _rng = new(1337);

    private int _inputDim;
    private int _actionCount;
    private int _seed;
    private double _lr;
    private long _trainSteps;
    private double _lastLoss;
    private string? _lastError;

    public string? LastError => _lastError;

    public MlInferResponse Infer(MlInferRequest req)
    {
        try
        {
            lock (_lock)
            {
                var state = req.Observation ?? req.State;
                if (state == null || state.Length == 0)
                    return new MlInferResponse(false, Array.Empty<double>(), 0, "state missing", null, 0, 0);

                var inputDim = req.InputDim ?? state.Length;
                var actionCount = req.ActionCount ?? _actionCount;
                if (actionCount <= 0)
                    actionCount = _actionCount == 0 ? 8 : _actionCount;

                EnsureModel(inputDim, actionCount, seed: _seed == 0 ? 1337 : _seed, lr: _lr == 0 ? 0.0005 : _lr);
                var q = PredictQ(state);
                var probs = Softmax(q);
                var entropy = Entropy(probs);
                var avgQ = q.Length == 0 ? 0 : q.Average();
                var mask = req.ActionMaskF ?? ToFloatMask(req.ActionMask);
                var actionIdx = ArgMaxWithMask(q, mask);
                return new MlInferResponse(true, q, actionIdx, null, probs.Select(v => (float)v).ToArray(), entropy, avgQ);
            }
        }
        catch (Exception ex)
        {
            _lastError = ex.Message;
            return new MlInferResponse(false, Array.Empty<double>(), 0, ex.Message, null, 0, 0);
        }
    }

    public MlTrainResponse Train(MlTrainRequest req)
    {
        try
        {
            lock (_lock)
            {
                var cfg = req.Config ?? new MlTrainConfigDto(
                    BufferSize: 1024,
                    BatchSize: 256,
                    TrainStepsPerBatch: 1,
                    TrainEveryTicks: 10,
                    TargetUpdateTicks: 200,
                    Gamma: req.Gamma ?? 0.98,
                    GradClip: 1.0,
                    Seed: req.Config?.Seed ?? 1337
                );
                _seed = cfg.Seed;
                _lr = _lr == 0 ? 0.0005 : _lr;
                var state = req.S ?? req.Transition?.State;
                var nextState = req.S2 ?? req.Transition?.NextState;
                var action = req.A ?? req.Transition?.ActionIndex;
                var reward = req.R ?? req.Transition?.Reward;
                var done = req.Done ?? req.Transition?.Done;
                var mask2 = req.ActionMask2 ?? req.Transition?.ActionMask2;
                if (state == null || nextState == null || action == null || reward == null || done == null)
                    return BuildTrainResponse(false, false, 0, 0, "transition missing");

                var inputDim = req.InputDim ?? state.Length;
                var actionCount = req.ActionCount ?? _actionCount;
                if (actionCount <= 0)
                    actionCount = _actionCount == 0 ? 8 : _actionCount;

                EnsureModel(inputDim, actionCount, _seed, _lr);
                EnsureBuffer(cfg.BufferSize);

                var transition = new Transition(
                    state,
                    action.Value,
                    reward.Value,
                    nextState,
                    done.Value,
                    ToFloatMask(mask2)
                );
                _buffer.Add(transition);

                if (cfg.TrainEveryTicks <= 0 || req.Tick % cfg.TrainEveryTicks != 0)
                    return BuildTrainResponse(true, false, _lastLoss, 0, null);
                if (_buffer.Count < cfg.BatchSize || cfg.BatchSize <= 0)
                    return BuildTrainResponse(true, false, _lastLoss, 0, null);

                var steps = Math.Max(1, cfg.TrainStepsPerBatch);
                var lossSum = 0.0;
                var gradNorm = 0.0;
                var didTrain = false;

                for (var i = 0; i < steps; i++)
                {
                    var batch = _buffer.SampleBatch(cfg.BatchSize, _rng);
                    if (batch.Count == 0) break;
                    var loss = TrainBatch(batch, cfg.Gamma, cfg.GradClip, out gradNorm);
                    if (double.IsNaN(loss) || double.IsInfinity(loss))
                        return BuildTrainResponse(false, false, double.NaN, gradNorm, "nan");
                    lossSum += loss;
                    _trainSteps++;
                    didTrain = true;
                    if (cfg.TargetUpdateTicks > 0 && _trainSteps % cfg.TargetUpdateTicks == 0)
                        CopyWeights(_net!, _targetNet!);
                }

                if (didTrain)
                    _lastLoss = lossSum / steps;

                return BuildTrainResponse(true, didTrain, _lastLoss, gradNorm, null);
            }
        }
        catch (Exception ex)
        {
            _lastError = ex.Message;
            return BuildTrainResponse(false, false, double.NaN, 0, ex.Message);
        }
    }

    private MlTrainResponse BuildTrainResponse(
        bool ok,
        bool trained,
        double loss,
        double gradNorm,
        string? reason)
    {
        return new MlTrainResponse(
            Ok: ok,
            Loss: loss,
            AvgQ: 0,
            TrainSteps: _trainSteps,
            Epsilon: 0,
            InvalidActions: 0,
            Reason: reason,
            Trained: trained,
            GradNorm: gradNorm,
            BufferSize: _buffer.Count
        );
    }

    public MlCheckpointResponse Save(MlCheckpointRequest req)
    {
        try
        {
            lock (_lock)
            {
                if (string.IsNullOrWhiteSpace(req.Path))
                    return new MlCheckpointResponse(false, "path empty");
                var dir = Path.GetDirectoryName(req.Path);
                if (!string.IsNullOrWhiteSpace(dir))
                    Directory.CreateDirectory(dir);
                var weightsPath = req.Path + ".net";
                if (_net != null)
                    ML.Core.Serialization.ModelStore.SaveToFile(weightsPath, _net);
                var meta = new { trainSteps = _trainSteps, weightsPath };
                var json = System.Text.Json.JsonSerializer.Serialize(meta);
                File.WriteAllText(req.Path, json);
                return new MlCheckpointResponse(true, json);
            }
        }
        catch (Exception ex)
        {
            _lastError = ex.Message;
            return new MlCheckpointResponse(false, ex.Message);
        }
    }

    public MlCheckpointResponse Load(MlCheckpointRequest req)
    {
        try
        {
            lock (_lock)
            {
                if (string.IsNullOrWhiteSpace(req.Path) || !File.Exists(req.Path))
                    return new MlCheckpointResponse(false, "path missing");
                try
                {
                    var json = File.ReadAllText(req.Path);
                    var meta = System.Text.Json.JsonSerializer.Deserialize<CheckpointMeta>(
                        json,
                        new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                    if (meta is not null && !string.IsNullOrWhiteSpace(meta.WeightsPath) && File.Exists(meta.WeightsPath))
                    {
                        _net = ML.Core.Serialization.ModelStore.LoadFromFile(meta.WeightsPath);
                        _targetNet = ML.Core.Serialization.ModelStore.LoadFromFile(meta.WeightsPath);
                        _optimizer = new AdamOptimizer(_lr == 0 ? 0.0005 : _lr);
                        _trainSteps = meta.TrainSteps;
                        return new MlCheckpointResponse(true, json);
                    }
                }
                catch (Exception ex)
                {
                    _lastError = ex.Message;
                    return new MlCheckpointResponse(false, ex.Message);
                }

                return new MlCheckpointResponse(false, "load failed");
            }
        }
        catch (Exception ex)
        {
            _lastError = ex.Message;
            return new MlCheckpointResponse(false, ex.Message);
        }
    }

    private void EnsureModel(int inputDim, int actionCount, int seed, double lr)
    {
        // Checkpoints created before v1.1 did not store dimensions.  Bind the
        // restored network to the dimensions of its first request instead of
        // rebuilding it and silently discarding the restored weights.
        if (_net != null && _inputDim <= 0 && _actionCount <= 0)
        {
            _inputDim = inputDim;
            _actionCount = actionCount;
            _seed = seed;
            _lr = lr;
            return;
        }

        if (_net != null && _inputDim == inputDim && _actionCount == actionCount)
            return;
        _inputDim = inputDim;
        _actionCount = actionCount;
        _seed = seed;
        _lr = lr;
        _net = BuildNet(inputDim, actionCount, seed);
        _targetNet = BuildNet(inputDim, actionCount, seed + 101);
        _optimizer = new AdamOptimizer(lr);
        CopyWeights(_net, _targetNet);
    }

    private void EnsureBuffer(int capacity)
    {
        var cap = Math.Max(1024, capacity);
        if (_buffer.Capacity != cap)
            _buffer = new ExperienceBuffer(cap);
    }

    private double[] PredictQ(float[] state)
    {
        if (_net == null) return Array.Empty<double>();
        var input = ToDouble(state);
        var q = _net.Forward(input, training: false);
        if (q.Any(v => double.IsNaN(v) || double.IsInfinity(v)))
            return new double[_actionCount];
        return q;
    }

    private double TrainBatch(IReadOnlyList<Transition> batch, double gamma, double gradClip, out double gradNorm)
    {
        gradNorm = 0;
        if (_net == null || _optimizer == null || _targetNet == null) return 0;
        _optimizer.ZeroGrad(_net.Parameters());
        double lossSum = 0;

        foreach (var t in batch)
        {
            var q = _net.Forward(ToDouble(t.State), training: true);
            var nextQ = _targetNet.Forward(ToDouble(t.NextState), training: false);
            if (q.Any(v => double.IsNaN(v) || double.IsInfinity(v)) || nextQ.Any(v => double.IsNaN(v) || double.IsInfinity(v)))
                return double.NaN;
            var maxNext = MaxMasked(nextQ, t.NextActionMask);
            var target = t.Reward + (t.Done ? 0.0 : gamma * maxNext);
            var diff = q[t.Action] - target;
            lossSum += diff * diff;
            var grad = new double[_actionCount];
            grad[t.Action] = 2.0 * diff;
            _net.Backward(grad);
        }

        var scale = 1.0 / batch.Count;
        ScaleGrads(_net, scale);
        gradNorm = GradNorm(_net);
        ClipGrads(_net, gradClip);

        if (double.IsNaN(lossSum) || double.IsInfinity(lossSum))
        {
            _optimizer.ZeroGrad(_net.Parameters());
            return double.NaN;
        }

        _optimizer.Step(_net.Parameters());
        return lossSum / batch.Count;
    }

    private static double[] ToDouble(float[] input)
    {
        var arr = new double[input.Length];
        for (var i = 0; i < input.Length; i++)
            arr[i] = input[i];
        return arr;
    }

    private static double[] Softmax(double[] logits)
    {
        if (logits.Length == 0) return Array.Empty<double>();
        var max = logits.Max();
        var exps = new double[logits.Length];
        double sum = 0;
        for (var i = 0; i < logits.Length; i++)
        {
            var e = Math.Exp(logits[i] - max);
            exps[i] = e;
            sum += e;
        }
        if (sum <= 0) return exps.Select(_ => 1.0 / logits.Length).ToArray();
        for (var i = 0; i < exps.Length; i++)
            exps[i] /= sum;
        return exps;
    }

    private static double Entropy(double[] probs)
    {
        if (probs.Length == 0) return 0;
        double sum = 0;
        for (var i = 0; i < probs.Length; i++)
        {
            var p = Math.Clamp(probs[i], 1e-6, 1.0);
            sum -= p * Math.Log(p);
        }
        return sum / Math.Log(probs.Length);
    }

    private static int ArgMaxWithMask(double[] q, float[]? mask)
    {
        if (mask == null || mask.Length != q.Length)
            return ArgMax(q);
        var max = double.NegativeInfinity;
        var idx = 0;
        for (var i = 0; i < q.Length; i++)
        {
            if (mask[i] <= 0f) continue;
            if (q[i] > max)
            {
                max = q[i];
                idx = i;
            }
        }
        return idx;
    }

    private static int ArgMax(double[] values)
    {
        var idx = 0;
        var max = values.Length > 0 ? values[0] : 0;
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] > max)
            {
                max = values[i];
                idx = i;
            }
        }
        return idx;
    }

    private static double MaxMasked(double[] q, float[]? mask)
    {
        if (mask == null || mask.Length != q.Length)
            return q.Length == 0 ? 0 : q.Max();
        var max = double.NegativeInfinity;
        for (var i = 0; i < q.Length; i++)
        {
            if (mask[i] <= 0f) continue;
            if (q[i] > max) max = q[i];
        }
        if (double.IsNegativeInfinity(max))
            return q.Length == 0 ? 0 : q.Max();
        return max;
    }

    private static Network BuildNet(int inputDim, int actionCount, int seed)
    {
        var net = new Network();
        net.Add(new LinearLayer(inputDim, 64, seed));
        net.Add(new ActivationLayer(64, ActivationType.ReLu));
        net.Add(new LinearLayer(64, 64, seed + 17));
        net.Add(new ActivationLayer(64, ActivationType.ReLu));
        net.Add(new LinearLayer(64, actionCount, seed + 31));
        return net;
    }

    private static void CopyWeights(Network from, Network to)
    {
        var src = from.Parameters().ToList();
        var dst = to.Parameters().ToList();
        if (src.Count != dst.Count) return;
        for (var i = 0; i < src.Count; i++)
        {
            var s = src[i];
            var d = dst[i];
            var len = Math.Min(s.Value.Length, d.Value.Length);
            Array.Copy(s.Value, d.Value, len);
        }
    }

    private static void ScaleGrads(Network net, double scale)
    {
        foreach (var p in net.Parameters())
        {
            for (var i = 0; i < p.Grad.Length; i++)
                p.Grad[i] *= scale;
        }
    }

    private static double GradNorm(Network net)
    {
        double sum = 0;
        foreach (var p in net.Parameters())
        {
            for (var i = 0; i < p.Grad.Length; i++)
                sum += p.Grad[i] * p.Grad[i];
        }
        return Math.Sqrt(sum);
    }

    private static void ClipGrads(Network net, double clip)
    {
        if (clip <= 0) return;
        double sum = 0;
        foreach (var p in net.Parameters())
        {
            for (var i = 0; i < p.Grad.Length; i++)
                sum += p.Grad[i] * p.Grad[i];
        }
        var norm = Math.Sqrt(sum);
        if (norm <= clip || norm <= 0) return;
        var scale = clip / norm;
        foreach (var p in net.Parameters())
        {
            for (var i = 0; i < p.Grad.Length; i++)
                p.Grad[i] *= scale;
        }
    }

    private sealed record CheckpointMeta(long TrainSteps, string WeightsPath);

    private sealed class ExperienceBuffer
    {
        private readonly Transition[] _buffer;
        private int _count;
        private int _index;

        public int Capacity => _buffer.Length;
        public int Count => _count;

        public ExperienceBuffer(int capacity)
        {
            if (capacity <= 0) throw new ArgumentException("capacity must be > 0", nameof(capacity));
            _buffer = new Transition[capacity];
        }

        public void Add(Transition transition)
        {
            _buffer[_index] = transition;
            _index = (_index + 1) % _buffer.Length;
            if (_count < _buffer.Length) _count++;
        }

        public IReadOnlyList<Transition> SampleBatch(int batchSize, Random rng)
        {
            if (_count == 0 || batchSize <= 0) return Array.Empty<Transition>();
            var n = Math.Min(batchSize, _count);
            var list = new List<Transition>(n);
            for (var i = 0; i < n; i++)
            {
                var idx = rng.Next(_count);
                list.Add(_buffer[idx]);
            }
            return list;
        }
    }

    private readonly record struct Transition(
        float[] State,
        int Action,
        float Reward,
        float[] NextState,
        bool Done,
        float[]? NextActionMask
    );

    private static float[]? ToFloatMask(bool[]? mask)
    {
        if (mask == null) return null;
        var arr = new float[mask.Length];
        for (var i = 0; i < mask.Length; i++)
            arr[i] = mask[i] ? 1f : 0f;
        return arr;
    }
}
