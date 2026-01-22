namespace ML.Core.Abstractions
{// IParameter.cs
    public interface IParameter
    {
        double[] Value { get; }
        double[] Grad { get; }
        void ZeroGrad();
    }

}