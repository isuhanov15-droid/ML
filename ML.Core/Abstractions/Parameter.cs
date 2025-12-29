using System;

namespace ML.Core.Abstractions;

public sealed class Parameter : IParameter
{
    public double[] Value { get; }
    public double[] Grad { get; }

    public Parameter(double[] value, double[] grad)
    {
        Value = value ?? throw new ArgumentNullException(nameof(value));
        Grad = grad ?? throw new ArgumentNullException(nameof(grad));

        if (Value.Length != Grad.Length)
            throw new ArgumentException("Value and Grad must have same length.");
    }

    public void ZeroGrad()
    {
        Array.Clear(Grad, 0, Grad.Length);
    }
}
