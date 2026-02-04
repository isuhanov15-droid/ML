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
);

public sealed record MlInferResponse(
    bool Ok,
    double[] QValues,
    int ActionIndex,
    string? Reason,
    float[]? Probabilities,
    double Entropy,
    double AvgQ
);

public sealed record MlTransitionDto(
    float[] State,
    int ActionIndex,
    float Reward,
    float[] NextState,
    bool Done,
    bool[]? ActionMask,
    bool[]? ActionMask2
);

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
);

public sealed record MlTrainResponse(
    bool Ok,
    double Loss,
    double AvgQ,
    long TrainSteps,
    double Epsilon,
    long InvalidActions,
    string? Reason
);

public sealed record MlCheckpointRequest(string Path);

public sealed record MlCheckpointResponse(bool Ok, string? Meta);

public sealed record MlStatusRequest();

public sealed record MlStatusResponse(
    bool Ok,
    string[] Methods,
    string Version,
    bool HasCore,
    string? LastError
);
