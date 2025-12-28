namespace ML.Core.Abstractions;

public interface IOptimizer
{
    void Step(Neuron neuron);
    void NextStep();
}
