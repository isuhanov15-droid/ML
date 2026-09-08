using DeepBrain.Host.BrainLife;
using DeepBrain.Host.BrainLife.Ml;
using DeepBrain.Shared.BrainDtos.V6;

namespace DeepBrain.Experiments;

public sealed record PolicyObservation(long Tick, string HeuristicAction, string NetworkAction,
    string AdvisorAction, int LegalCount, double QRange, double TopTwoQGap,
    double ProbabilitySpread, double MaxProbability, double NetWeight, double Epsilon,
    double[] QValues, string[] LegalActions);
public sealed record DiagnosticSummary(long Decisions, long MultiChoiceDecisions, double MeanLegalCount, long NetworkVsHeuristic, long AdvisorVsHeuristic,
    long ExecutedVsHeuristic, long PostAdvisorChanges, double MeanQRange, double MeanTopTwoQGap,
    double MeanProbabilitySpread, double MeanMaxProbability);

// Observes the actual advisor call. It never changes Q, scores or the decision.
public sealed class DiagnosticAdvisor : IMlPolicyAdvisor
{
    private readonly IMlPolicyAdvisor _inner;
    private readonly SerializedBackend _backend;
    private long _multi, _legalTotal;
    private long _count, _networkDiff, _advisorDiff, _executedDiff, _postAdvisor;
    private double _range, _gap, _spread, _maxP;
    private long _lastCompleted = -1;
    public PolicyObservation? Pending { get; private set; }
    public DiagnosticAdvisor(IMlPolicyAdvisor inner, SerializedBackend backend) { _inner = inner; _backend = backend; }
    public PolicyDecision SelectAction(float[] stateVec, IReadOnlyDictionary<string,double> scores,
        IReadOnlyList<string> allowed, float[] mask, MlConfig config, long tick)
    {
        _backend.LastQ = Array.Empty<double>();
        var result = _inner.SelectAction(stateVec, scores, allowed, mask, config, tick);
        Pending = Analyze(tick, scores, allowed, config.ActionMasking ? mask : null, _backend.LastQ, result);
        return result;
    }
    public PolicyObservation? Complete(long tick, string executed)
    {
        if (Pending is not { } p || p.Tick != tick || _lastCompleted == tick) return null;
        _lastCompleted = tick; _count++;
        _legalTotal += p.LegalCount; if (p.LegalCount > 1) _multi++;
        if (p.NetworkAction != p.HeuristicAction) _networkDiff++;
        if (p.AdvisorAction != p.HeuristicAction) _advisorDiff++;
        if (executed != p.HeuristicAction) _executedDiff++;
        if (executed != p.AdvisorAction) _postAdvisor++;
        _range += p.QRange; _gap += p.TopTwoQGap; _spread += p.ProbabilitySpread; _maxP += p.MaxProbability;
        return p;
    }
    public DiagnosticSummary Summary() => new(_count, _multi, _legalTotal/(double)Math.Max(1,_count), _networkDiff, _advisorDiff, _executedDiff, _postAdvisor,
        _range/Math.Max(1,_count), _gap/Math.Max(1,_count), _spread/Math.Max(1,_count), _maxP/Math.Max(1,_count));
    public static PolicyObservation? Analyze(long tick, IReadOnlyDictionary<string,double> scores,
        IReadOnlyList<string> allowed, float[]? mask, double[] q, PolicyDecision decision)
    {
        if (q.Length != ActionCatalog.Count || q.Any(v => !double.IsFinite(v))) return null;
        var legal = Enumerable.Range(0, q.Length).Where(i =>
            (allowed.Count == 0 || allowed.Contains(ActionCatalog.Actions[i])) &&
            (mask is null || mask.Length != q.Length || mask[i] > 0.5f)).ToArray();
        if (legal.Length == 0) return null;
        // Match the advisor's stable heuristic tie-break order.
        var heuristic = scores.Where(kv => allowed.Count == 0 || allowed.Contains(kv.Key))
            .OrderByDescending(kv => kv.Value).Select(kv => kv.Key).FirstOrDefault() ?? ActionCatalog.Actions[legal[0]];
        var ordered = legal.OrderByDescending(i => q[i]).ToArray();
        var legalMask = new float[q.Length]; foreach (var i in legal) legalMask[i] = 1;
        var probs = MlMath.Softmax(q, legalMask);
        return new(tick, heuristic, ActionCatalog.Actions[ordered[0]], decision.ActionName, legal.Length,
            q[ordered[0]]-q[ordered[^1]], ordered.Length > 1 ? q[ordered[0]]-q[ordered[1]] : 0,
            legal.Max(i => probs[i])-legal.Min(i => probs[i]), legal.Max(i => probs[i]),
            decision.NetWeight, decision.Epsilon, q.ToArray(), legal.Select(i => ActionCatalog.Actions[i]).ToArray());
    }
    public float[] Encode(StateVectorInput input) => _inner.Encode(input);
    public void Observe(float[] s, int a, float r, float[] next, bool done, float[] mask, MlConfig cfg, long tick) => _inner.Observe(s,a,r,next,done,mask,cfg,tick);
    public bool TryLoad(string path, out int episode) => _inner.TryLoad(path,out episode);
    public void TrySave(string path, int episode) => _inner.TrySave(path,episode);
    public void Reset(MlConfig cfg) => _inner.Reset(cfg);
    public void ResetCounters() => _inner.ResetCounters();
    public bool TryConnectRemote() => _inner.TryConnectRemote();
    public void DisconnectRemote() => _inner.DisconnectRemote();
    public MlPolicyDto BuildTelemetry(bool enabled, bool core, int dim, int count, double reward, string? reason, string mode, bool train, int trainingEpisodes, int evalEpisodes)
        => _inner.BuildTelemetry(enabled,core,dim,count,reward,reason,mode,train,trainingEpisodes,evalEpisodes);
}

public sealed record SequenceComparison(int Ticks, int DifferentActions, long? FirstDifferentTick,
    int BothAwakeTicks, int DifferentActionsBothAwake);
public static class SequenceDiagnostics
{
    public static SequenceComparison Compare(string[] a, string[] b)
    {
        if (a.Length != b.Length) throw new InvalidDataException("Different sequence lengths");
        int diff=0, awake=0, awakeDiff=0; long? first=null;
        for (int i=0;i<a.Length;i++)
        {
            if (a[i]!=b[i]) { diff++; first ??= i; }
            if (a[i]!="sleep" && b[i]!="sleep") { awake++; if(a[i]!=b[i]) awakeDiff++; }
        }
        return new(a.Length,diff,first,awake,awakeDiff);
    }
}
