using System.Text.Json;
using ML.Core;
using ML.Core.Layers;
using ML.Core.Serialization;
using ML.Host.Services;
using Server = ML.Shared.Protocol;
using Client = DeepBrain.Shared.MlBridge;

var failures = 0;
var cases = new (string Name, Action Run)[]
{
    ("DeepBrain nextActionMask survives serialization and controls Bellman target", () => CheckLoss(false, new[] { false, true, false }, false, 0.0004)),
    ("Legacy actionMask2 remains supported", () => CheckLoss(false, new[] { false, true, false }, true, 0.0004)),
    ("Terminal transition does not bootstrap", () => CheckLoss(true, new[] { false, true, false }, false, 1)),
    ("Empty next-action mask does not bootstrap forbidden actions", () => CheckLoss(false, new[] { false, false, false }, false, 1)),
    ("Masked inference normalizes legal actions despite extreme forbidden Q", CheckInference)
};
foreach (var test in cases)
{
    try { test.Run(); Console.WriteLine($"PASS: {test.Name}"); }
    catch (Exception ex) { failures++; Console.WriteLine($"FAIL: {test.Name}: {ex.Message}"); }
}
Console.WriteLine($"RESULT: {cases.Length - failures}/{cases.Length} passed");
return failures == 0 ? 0 : 1;

static void CheckLoss(bool done, bool[] mask, bool legacy, double expected)
{
    WithService(service =>
    {
        var request = new Client.MlTrainRequest(1,
            new Client.MlTransitionDto(new float[2], 1, 0, new float[2], done, mask),
            new Client.MlTrainConfigDto(1024, 1, 1, 1, 200, 0.98, 1, 1337), 2, 3);
        var json = JsonSerializer.Serialize(request, Server.JsonOptions.Default);
        if (legacy) json = json.Replace("nextActionMask", "actionMask2");
        var decoded = JsonSerializer.Deserialize<Server.MlTrainRequest>(json, Server.JsonOptions.Default)!;
        var result = service.Train(decoded);
        if (!result.Ok || !result.Trained) throw new Exception($"training failed: {result.Reason}");
        Near(expected, result.Loss);
        if (result.TrainSteps != 446343) throw new Exception("Restored trainSteps lost");
    });
}

static void CheckInference()
{
    WithService(service =>
    {
        var request = new Client.MlInferRequest(new float[2], new[] { false, true, true }, 2, 3);
        var json = JsonSerializer.Serialize(request, Server.JsonOptions.Default);
        var decoded = JsonSerializer.Deserialize<Server.MlInferRequest>(json, Server.JsonOptions.Default)!;
        var result = service.Infer(decoded);
        if (!result.Ok) throw new Exception(result.Reason);
        Near(1000, result.QValues[0]); // Loading/inference must preserve original weights.
        var p = result.Probabilities!;
        Near(0, p[0]);
        Near(1, p.Sum(x => (double)x), 1e-6);
        if (p[1] <= p[2] || result.ActionIndex != 1) throw new Exception("Legal action ranking lost");
    });
}

static void WithService(Action<BrainMlService> test)
{
    var dir = Path.Combine(Path.GetTempPath(), "deepbrain-bridge-test-" + Guid.NewGuid().ToString("N"));
    Directory.CreateDirectory(dir);
    try
    {
        var net = new Network();
        var layer = new LinearLayer(2, 3);
        Array.Clear(layer.Weights);
        layer.Bias[0] = 1000; layer.Bias[1] = 1; layer.Bias[2] = 0;
        net.Add(layer);
        var weights = Path.Combine(dir, "test.net");
        var meta = Path.Combine(dir, "test.chk");
        ModelStore.SaveToFile(weights, net);
        File.WriteAllText(meta, JsonSerializer.Serialize(new { trainSteps = 446342, weightsPath = weights }));
        var service = new BrainMlService();
        var loaded = service.Load(new Server.MlCheckpointRequest(meta));
        if (!loaded.Ok) throw new Exception(loaded.Meta);
        test(service);
    }
    finally { Directory.Delete(dir, true); }
}

static void Near(double expected, double actual, double tolerance = 1e-9)
{
    if (!double.IsFinite(actual) || Math.Abs(actual - expected) > tolerance)
        throw new Exception($"expected={expected}, actual={actual}");
}
