namespace ML.Core.Training.Callbacks;

public sealed class CallbackAdapter : ITrainCallback
{
    private readonly ICallback _inner;

    public CallbackAdapter(ICallback inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    public void OnEpochEnd(TrainEpochResult r)
    {
        // Старый интерфейс жил epoch=int (и у тебя там epoch 0-based)
        // В TrainEpochResult Epoch удобно хранить 1-based.
        // Поэтому пробрасываем как 0-based:
        _inner.OnEpochEnd(r.Epoch - 1);
    }
}
