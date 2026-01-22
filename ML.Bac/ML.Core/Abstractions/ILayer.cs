namespace ML.Core.Abstractions;

// ILayer.cs
public interface ILayer
{
    double[] Forward(double[] input, bool training = true);
    double[] Backward(double[] gradOutput);
    IEnumerable<IParameter> Parameters();
}

