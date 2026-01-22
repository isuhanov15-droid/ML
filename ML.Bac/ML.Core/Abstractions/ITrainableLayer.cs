namespace ML.Core.Abstractions;

public interface ITrainableLayer : ILayer
{
    // outputGradient = dL/dOut
    double[] Backward(double[] outputGradient, IOptimizer optimizer);
}
