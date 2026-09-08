namespace ML.Shared.Protocol;

public sealed record MlInferRequest(
    string? ModelId,
    float[]? Observation,
    float[]? State,
    bool[]? ActionMask,
    float[]? ActionMaskF,
    int? TopK,
    int? Seed,
    int? InputDim,
    int? ActionCount
)
{
    public double? LearningRate { get; init; }
}

public sealed record MlInferResponse(
    bool Ok,
    double[] QValues,
    int ActionIndex,
    string? Reason,
    float[]? Probabilities,
    double Entropy,
    double AvgQ
) { public string? ServerInstance { get; init; } }

public sealed record MlTransitionDto(
    float[] State,
    int ActionIndex,
    float Reward,
    float[] NextState,
    bool Done,
    bool[]? ActionMask,
    bool[]? ActionMask2
)
{
    // DeepBrain uses nextActionMask; retain actionMask2 for older clients.
    public bool[]? NextActionMask { get; init; }
}

public sealed record MlTrainConfigDto(
    int BufferSize,
    int BatchSize,
    int TrainStepsPerBatch,
    int TrainEveryTicks,
    int TargetUpdateTicks,
    double Gamma,
    double GradClip,
    int Seed
);

public sealed record MlTrainRequest(
    long EpisodeId,
    long Tick,
    float[]? S,
    int? A,
    float? R,
    float[]? S2,
    bool? Done,
    bool[]? ActionMask,
    bool[]? ActionMask2,
    float? Gamma,
    MlTransitionDto? Transition,
    MlTrainConfigDto? Config,
    int? InputDim,
    int? ActionCount
) { public string? ExpectedInstanceId { get; init; } }

public sealed record MlTrainResponse(
    bool Ok,
    double Loss,
    double AvgQ,
    long TrainSteps,
    double Epsilon,
    long InvalidActions,
    string? Reason,
    bool Trained = false,
    double GradNorm = 0,
    int BufferSize = 0
);

public sealed record MlCheckpointRequest(string Path) { public string? ExpectedInstanceId { get; init; } }

public sealed record MlCheckpointResponse(bool Ok, string? Meta);

public sealed record MlStatusRequest();

public sealed record MlStatusResponse(
    bool Ok,
    string[] Methods,
    string Version,
    bool HasCore,
    string? LastError
);

public sealed record MlResetRequest(int Seed, double LearningRate);
