using System.Net;
using ML.Host.Networking;
using ML.Host.Rpc;
using ML.Host.Rpc.Handlers;
using ML.Host.Services;
using ML.Host.Storage;

namespace ML.Host;

internal static class Program
{
    public static async Task Main(string[] args)
    {
        int port = 5001;
        int rpcPort = 7777;
        if (args.Length > 0 && int.TryParse(args[0], out var p))
            port = p;
        if (args.Length > 1 && int.TryParse(args[1], out var rp))
            rpcPort = rp;

        var host = new HostServer(IPAddress.Any, port);
        host.Start();

        var router = new RpcRouter();
        var rpcServer = new RpcTcpServer(IPAddress.Any, rpcPort, router);
        var projects = new ProjectStore(AppContext.BaseDirectory);
        var experiments = new ExperimentStore(AppContext.BaseDirectory);
        var runs = new RunStore(AppContext.BaseDirectory);
        var training = new TrainingService();
        var inference = new InferenceService();
        var brainMl = new BrainMlService();
        var hostHandlers = new HostHandlers();
        var trainHandlers = new TrainHandlers(training, rpcServer.Events, experiments, runs, inference);
        var projectHandlers = new ProjectHandlers(projects, experiments, runs);
        var inferHandlers = new InferenceHandlers(inference, projects, rpcServer.Events);
        var brainMlHandlers = new BrainMlHandlers(brainMl, () => router.Methods);

        router.Register("host.info", _ => hostHandlers.GetInfoAsync());
        router.Register("host.capabilities", _ => hostHandlers.GetCapabilitiesAsync());
        router.Register("train.start", trainHandlers.StartAsync);
        router.Register("train.stop", trainHandlers.StopAsync);
        router.Register("project.create", projectHandlers.CreateProjectAsync);
        router.Register("project.list", projectHandlers.ListProjectsAsync);
        router.Register("project.update", projectHandlers.UpdateProjectAsync);
        router.Register("project.delete", projectHandlers.DeleteProjectAsync);
        router.Register("project.files", projectHandlers.ListProjectFilesAsync);
        router.Register("file.create", projectHandlers.CreateProjectFileAsync);
        router.Register("file.read", projectHandlers.ReadProjectFileAsync);
        router.Register("file.write", projectHandlers.WriteProjectFileAsync);
        router.Register("file.rename", projectHandlers.RenameProjectFileAsync);
        router.Register("file.delete", projectHandlers.DeleteProjectFileAsync);
        router.Register("experiment.create", projectHandlers.CreateExperimentAsync);
        router.Register("experiment.list", projectHandlers.ListExperimentsAsync);
        router.Register("run.list", projectHandlers.ListRunsAsync);
        router.Register("run.metrics", projectHandlers.GetRunMetricsAsync);
        router.Register("model.load", inferHandlers.LoadAsync);
        router.Register("infer.single", inferHandlers.InferSingleAsync);
        router.Register("infer.batch", inferHandlers.InferBatchAsync);
        router.Register("eval.dataset", inferHandlers.EvalDatasetAsync);
        router.Register("ml.ping", brainMlHandlers.PingAsync);
        router.Register("ml.status", brainMlHandlers.StatusAsync);
        router.Register("ml.infer", brainMlHandlers.InferAsync);
        router.Register("ml.train", brainMlHandlers.TrainAsync);
        router.Register("ml.checkpoint.save", brainMlHandlers.SaveAsync);
        router.Register("ml.checkpoint.load", brainMlHandlers.LoadAsync);
        router.Register("ml.reset", brainMlHandlers.ResetAsync);

        rpcServer.Start();

        Console.WriteLine($"ML.Host listening on 0.0.0.0:{port}");
        Console.WriteLine($"ML.Host RPC listening on 0.0.0.0:{rpcPort}");
        Console.WriteLine($"RPC methods: {string.Join(", ", router.Methods)}");
        Console.WriteLine("Press Ctrl+C to stop.");

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };

        try
        {
            await Task.Delay(Timeout.Infinite, cts.Token);
        }
        catch (OperationCanceledException)
        {
        }

        await rpcServer.DisposeAsync();
        await host.DisposeAsync();
    }
}
