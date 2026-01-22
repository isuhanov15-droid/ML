namespace ML.Gui.Models;

public sealed record MetricPoint(
    int Epoch,
    double Loss,
    double? ValLoss,
    double? Accuracy,
    double? LearningRate,
    long? ElapsedMs,
    string? RunId);
