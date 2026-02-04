namespace ML.Shared.Protocol;

public sealed record MlInferRequest(
    float[] State,
    float[]? ActionMask,
    int InputDim,
    int ActionCount
);

public sealed record MlInferResponse(
    int ActionIndex,
    double[] QValues,
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
    float[]? NextActionMask
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
    long Tick,
    MlTransitionDto Transition,
    MlTrainConfigDto Config,
    int InputDim,
    int ActionCount
);

public sealed record MlTrainResponse(
    bool Trained,
    double Loss,
    double GradNorm,
    long TrainSteps,
    int BufferSize
);

public sealed record MlCheckpointRequest(string Path);

public sealed record MlCheckpointResponse(bool Ok, string? Meta);
