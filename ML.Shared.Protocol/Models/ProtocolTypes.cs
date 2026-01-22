namespace ML.Shared.Protocol;

public static class ProtocolTypes
{
    public const int Version = 1;

    public const string HostInfo = "host.info";
    public const string HostInfoResult = "host.info.result";

    public const string HostCapabilities = "host.capabilities";
    public const string HostCapabilitiesResult = "host.capabilities.result";

    public const string TrainStart = "train.start";
    public const string TrainStartResult = "train.start.result";

    public const string TrainStop = "train.stop";
    public const string TrainStopResult = "train.stop.result";

    public const string LogAppend = "log.append";
    public const string MetricsUpdate = "metrics.update";
    public const string TrainStateChanged = "train.stateChanged";

    public const string ModelLoad = "model.load";
    public const string ModelLoadResult = "model.load.result";

    public const string InferSingle = "infer.single";
    public const string InferSingleResult = "infer.single.result";

    public const string InferBatch = "infer.batch";
    public const string InferBatchResult = "infer.batch.result";

    public const string EvalDataset = "eval.dataset";
    public const string EvalDatasetResult = "eval.dataset.result";
}
