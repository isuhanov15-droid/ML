using System.Collections.Generic;

namespace ML.Core.Serialization;

public sealed class NetworkDto
{
    public int FormatVersion { get; set; } = 1;
    public int InputSize { get; set; }
    public List<LayerDto> Layers { get; set; } = new();
}

