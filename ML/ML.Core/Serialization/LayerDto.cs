namespace ML.Core.Serialization;

public sealed class LayerDto
{
    public string Type { get; set; } = "";      // "Dense" | "Softmax"

    public string? NeuronType { get; set; }     // <-- ДОБАВИЛИ: "Input" | "Hidden" | "Output"
    public string? Activation { get; set; }     // "ReLu" | "Linear" | ...

    public List<NeuronDto>? Neurons { get; set; }
}
