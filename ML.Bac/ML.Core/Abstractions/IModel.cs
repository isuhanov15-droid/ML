namespace ML.Core.Abstractions;
public interface IModel
{
    double[] Forward(double[] input, bool training = true);
    double[] Backward(double[] gradOutput);
    IEnumerable<IParameter> Parameters();
}
