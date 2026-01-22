namespace ML.Shared.Protocol;

public sealed record ModelLoadRequest(
    string ProjectId,
    string? RunId,
    string? ModelPath);

public sealed record ModelLoadResponse(
    string ModelId,
    string TaskType,
    int InputSize,
    int OutputSize,
    string[]? Labels);

public sealed record InferSingleRequest(
    string ModelId,
    double[] Input);

public sealed record InferSingleResponse(
    double[] Output,
    int? PredictedIndex,
    double[]? Probabilities);

public sealed record InferBatchRequest(
    string ModelId,
    double[][] Inputs);

public sealed record InferBatchResponse(
    double[][] Outputs,
    double[][]? Probabilities,
    int[]? PredictedIndices);

public sealed record EvalDatasetRequest(
    string ModelId,
    string DatasetPath,
    bool HasHeader,
    int[] InputCols,
    int LabelCol);

public sealed record EvalDatasetResponse(
    double? Accuracy,
    double? Loss,
    EvalSampleDto[] Samples);

public sealed record EvalSampleDto(
    double[] Input,
    double[]? Expected,
    double[]? Predicted);
