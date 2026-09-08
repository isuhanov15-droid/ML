using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using DeepBrain.Host.BrainLife;
using DeepBrain.Host.BrainLife.Ml;
using DeepBrain.Shared.Brain;
using DeepBrain.Experiments;

if (args.Length == 1 && args[0] == "--self-test") { DiagnosticTests.Run(); return; }

// Usage: <source-config> <NEW-output-directory> [training-ticks=24000] [evaluation-ticks=2400]
// No server, Ollama or production file is opened for writing.
if (args.Length is < 2 or > 4) throw new ArgumentException("config NEW-output-dir [train-ticks] [eval-ticks]");
var json = new JsonSerializerOptions(JsonSerializerDefaults.Web) { WriteIndented = true, PropertyNameCaseInsensitive = true };
var input = Path.GetFullPath(args[0]);
var root = Path.GetFullPath(args[1]);
if (Directory.Exists(root) || File.Exists(root)) throw new IOException("Output must not exist: " + root);
var source = JsonSerializer.Deserialize<BrainConfig>(File.ReadAllText(input), json) ?? throw new InvalidDataException("config");
var trainTicks = args.Length > 2 ? int.Parse(args[2]) : 24000;
var evalTicks = args.Length > 3 ? int.Parse(args[3]) : 2400;
if (trainTicks < 400 || evalTicks < 200 || source.Scenarios.Count == 0) throw new ArgumentException("invalid experiment limits");
Directory.CreateDirectory(root);
Directory.SetCurrentDirectory(root);
File.Copy(input, Path.Combine(root, "source-config.json"));
var tickMethod = typeof(LifeLoop).GetMethod("TickOnce", BindingFlags.Instance | BindingFlags.NonPublic)
    ?? throw new MissingMethodException("LifeLoop.TickOnce");
var results = new List<RunResult>();
try
{
    var training = Run("training", "round_robin", source.Scenarios.Keys.Order().First(), 1337, trainTicks, null, true);
    results.Add(training);
    if (training.TrainSteps <= 0) throw new InvalidOperationException("No training steps were performed");
    var checkpoint = Path.Combine(root, "training", "checkpoint");
    var weightsHash = Hash(WeightPath(checkpoint));
    foreach (var seed in new[] { 1337, 2027, 4099 })
    foreach (var scenario in source.Scenarios.Keys.Order(StringComparer.Ordinal))
    {
        var index = results.Count;
        results.Add(Run($"eval-{index}-heuristic", "fixed", scenario, seed, evalTicks, null, false));
        results.Add(Run($"eval-{index}-learned", "fixed", scenario, seed, evalTicks, checkpoint, false));
        if (weightsHash != Hash(WeightPath(checkpoint))) throw new InvalidOperationException("Training checkpoint changed during evaluation");
        SaveResults();
    }
    SaveResults();
    File.WriteAllText(Path.Combine(root, "DONE"), "Completed. Review report.json before promoting any weights.\n");
    Console.WriteLine("DONE: " + Path.Combine(root, "report.json"));
}
catch (Exception ex)
{
    File.WriteAllText(Path.Combine(root, "FAILED.txt"), ex.ToString());
    throw;
}

RunResult Run(string name, string mode, string scenario, int seed, int ticks, string? load, bool train)
{
    var dir = Path.Combine(root, name); Directory.CreateDirectory(dir);
    bool enabled = train || load is not null;
    var cfg = source with
    {
        UseMlAdvisor = false,
        ScenarioDefault = scenario, CurriculumMode = mode,
        Curriculum = source.Curriculum with { Mode = mode },
        Evaluation = source.Evaluation with { SaveReports = false },
        Memory = (source.Memory ?? MemoryConfig.Default) with { Enable = train, Path = Path.Combine(dir, "memory", "episodes.jsonl") },
        Habits = (source.Habits ?? HabitConfig.Default) with { Enable = train, Path = Path.Combine(dir, "memory", "habits.json"), ImportPath = "" },
        Llm = (source.Llm ?? LlmConfig.Default) with { Enable = false },
        ExternalApi = (source.ExternalApi ?? ExternalApiConfig.Default) with { Enable = false },
        Ml = source.Ml with { Enable = enabled, Backend = enabled ? "remote" : "off", Mode = train ? "training" : "evaluation",
            Seed = 1337, NetWeightWarmup = train ? source.Ml.NetWeightWarmup : 1,
            CheckpointPath = Path.Combine(dir, "checkpoint") }
    };
    // Evaluation starts with fresh agent state and frozen weights. Preload into a
    // fresh service, then write a local wrapper with EpisodeId=0, Epsilon=0.
    var backend = new SerializedBackend(cfg.Ml, _ => { });
    if (load is not null)
    {
        using var loadMeta = JsonDocument.Parse(File.ReadAllText(load));
        var backendLoad = loadMeta.RootElement.GetProperty("WeightsPath").GetString()!;
        if (!backend.TryLoad(backendLoad, out _)) throw new InvalidOperationException("Cannot load trained model");
        File.WriteAllText(cfg.Ml.CheckpointPath, JsonSerializer.Serialize(new { EpisodeId = 0, Epsilon = 0.0,
            TrainSteps = backend.TrainSteps, WeightsPath = backendLoad }));
    }
    var startSteps = backend.TrainSteps;
    var configPath = Path.Combine(dir, "config.json");
    File.WriteAllText(configPath, JsonSerializer.Serialize(cfg, json));
    using var log = new StreamWriter(Path.Combine(dir, "host.log"));
    using var decisionTrace = new StreamWriter(Path.Combine(dir, "decisions.jsonl"));
    var sequence = new List<string>();
    using var trace = new StreamWriter(Path.Combine(dir, "metrics.jsonl"));
    LifeStateDto? latest = null;
    var actions = new Dictionary<string, int>();
    var scenarioTicks = new Dictionary<string, int>();
    double reward = 0, pain = 0, safety = 0; int observed = 0, awake = 0, loops = 0, fallbacks = 0;
    var advisor = new DiagnosticAdvisor(new MlPolicyAdvisor(cfg.Ml, log.WriteLine, backend), backend);
    var loop = new LifeLoop(new WorldSim(seed), new HomeostasisEngine(), new InstinctEngine(), new EmotionEngine(),
        new ActionSelector(), new Actuator(), new RewardEngine(), new LearningEngine(), new BrainConfigLoader(configPath, log.WriteLine),
        (state, _) =>
        {
            latest = state; observed++;
            sequence.Add(state.LastDecision);
            var scenarioName = state.Scenario?.Name ?? "unknown";
            scenarioTicks[scenarioName] = scenarioTicks.GetValueOrDefault(scenarioName) + 1;
            var diagnostic = advisor.Complete(state.Tick, state.LastDecision);
            if (diagnostic is not null && (!train || state.Tick % 200 == 0))
                decisionTrace.WriteLine(JsonSerializer.Serialize(new { policy = diagnostic, executed = state.LastDecision }));
            reward += state.LastReward; pain += state.Homeostasis.Pain; safety += state.Homeostasis.Safety;
            if (state.LastDecision != "sleep") awake++;
            if (state.LoopInfo?.IsInLoop == true) loops++;
            actions[state.LastDecision] = actions.GetValueOrDefault(state.LastDecision) + 1;
            if (state.Ml is { } ml && (ml.NanSkips > 0 || ml.LastRemoteError is not null))
                throw new InvalidOperationException("ML diagnostic failure");
            fallbacks = Math.Max(fallbacks, state.Ml?.InvalidActionFallbackCount ?? 0);
            if (state.Tick % 200 == 0) trace.WriteLine(JsonSerializer.Serialize(new { state.Tick, state.Scenario,
                state.LastDecision, state.LastReward, state.Homeostasis, state.Ml, state.LoopInfo }));
        }, (_, _) => { }, (_, _) => { }, log.WriteLine, advisor);
    for (int i = 0; i < ticks; i++)
    {
        tickMethod.Invoke(loop, new object[] { 0.2, CancellationToken.None });
        if (i > 0 && i % 2000 == 0) Console.WriteLine($"{name}: {i}/{ticks}, trainSteps={backend.TrainSteps}, maxAbsQ={backend.MaxAbsQ:F3}");
    }
    loop.FlushPersistentState();
    if (train) advisor.TrySave(cfg.Ml.CheckpointPath, 0);
    if (!train && (backend.TrainCalls != 0 || backend.TrainSteps != startSteps))
        throw new InvalidOperationException("Evaluation trained the model");
    if (observed != ticks || latest is null) throw new InvalidOperationException("Missing states");
    File.WriteAllText(Path.Combine(dir, "last-state.json"), JsonSerializer.Serialize(latest, json));
    var result = new RunResult(name, scenario, seed, observed, awake, reward, reward/observed, pain/observed, safety/observed,
        loops, fallbacks, backend.TrainSteps, backend.TrainCalls, backend.MaxAbsQ, backend.MaxLoss, latest.Ml?.NetWeight ?? 0, actions, advisor.Summary(), scenarioTicks);
    File.WriteAllLines(Path.Combine(dir, "actions.txt"), sequence);
    if (enabled && advisor.Summary().Decisions == 0) throw new InvalidOperationException("No instrumented decisions");
    File.WriteAllText(Path.Combine(dir, "result.json"), JsonSerializer.Serialize(result, json));
    Console.WriteLine($"{name} / {scenario} seed={seed}: reward={reward:F4}, maxAbsQ={backend.MaxAbsQ:F4}");
    return result;
}
void SaveResults()
{
    var pairs = results.Skip(1).Chunk(2).Where(c => c.Length == 2).Select(c => new
    {
        c[0].Scenario, c[0].Seed, heuristicReward = c[0].Reward, learnedReward = c[1].Reward,
        rewardDifference = c[1].Reward - c[0].Reward, heuristicPain = c[0].MeanPain, learnedPain = c[1].MeanPain,
        c[1].MaxAbsQ, c[1].NetWeight, diagnostics = c[1].Diagnostics,
        sequence = SequenceDiagnostics.Compare(File.ReadAllLines(Path.Combine(root, c[0].Name, "actions.txt")),
            File.ReadAllLines(Path.Combine(root, c[1].Name, "actions.txt")))
    });
    File.WriteAllText(Path.Combine(root, "report.json"), JsonSerializer.Serialize(new
    {
        protocol = "deepbrain.clean-experiment.v2", trainingTicks = trainTicks, evaluationTicks = evalTicks,
        sourceConfigSha256 = Hash(input), results, pairs,
        notes = new[] { "Pilot, not a release gate. Identical initial state/scenario/world seed within each pair; actions may change later environment.",
            "Evaluation: no ML updates, epsilon=0; persistent memory/habits disabled in both arms. Learned arm uses the production blend at configured NetWeightMax; evaluation warmup=1 to test learned influence, not the restart ramp.",
            "LLM disabled. Training uses fresh memory and default habits with no legacy import. Training seed=1337.",
            "maxAbsQ measured across all inferred actions, including forbidden ones. Inspect legal-action metrics before diagnosing instability." }
    }, json));
}
static string WeightPath(string checkpoint)
{
    using var wrapper = JsonDocument.Parse(File.ReadAllText(checkpoint));
    var remote = wrapper.RootElement.GetProperty("WeightsPath").GetString()!;
    using var service = JsonDocument.Parse(File.ReadAllText(remote + ".service"));
    return service.RootElement.GetProperty("weightsPath").GetString()!;
}
static string Hash(string p) => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(p))).ToLowerInvariant();
record RunResult(string Name, string Scenario, int Seed, int Ticks, int AwakeTicks, double Reward, double MeanReward,
    double MeanPain, double MeanSafety, int LoopTicks, int InvalidFallbacks, long TrainSteps, long TrainCalls,
    double MaxAbsQ, double MaxLoss, double NetWeight, Dictionary<string,int> Actions, DiagnosticSummary Diagnostics, Dictionary<string,int> ScenarioTicks);
