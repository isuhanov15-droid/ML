namespace ML.Core.Abstractions;

// IOptimizer.cs
public interface IOptimizer
{
    void Step(IEnumerable<IParameter> parameters);
    void ZeroGrad(IEnumerable<IParameter> parameters);
}

