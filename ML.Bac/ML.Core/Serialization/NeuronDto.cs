namespace ML.Core.Serialization;
public sealed class NeuronDto
{
    public double[] Weights { get; set; } = [];
    public double Bias { get; set; }
}