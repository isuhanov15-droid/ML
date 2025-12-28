namespace ML.Core.Abstractions;

public interface ILayer
{
    int InputSize { get; }
    int OutputSize { get; }

    double[] Forward(double[] input);
}
