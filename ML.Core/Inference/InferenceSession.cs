using ML.Core.Serialization;
using ML.Core;

namespace ML.Core.Inference;

public sealed class InferenceSession : IInferenceSession
{
    private Network? _network;

    public void LoadFromFile(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Не указан путь к модели.", nameof(path));

        _network = ModelStore.LoadFromFile(path);
    }

    public void LoadFromStore(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            throw new ArgumentException("Не указано имя модели.", nameof(modelName));

        _network = ModelStore.Load(modelName);
    }

    public double[] Predict(double[] input)
    {
        var net = _network ?? throw new InvalidOperationException("Сначала загрузите модель.");
        return net.Forward(input, training: false);
    }
}
