namespace ML.Core.Training.Collbacks;
public interface ICallback
{
    void OnEpochEnd(int epoch);
}

public class StopFileCallback : ICallback
{
    private readonly string _path;

    public StopFileCallback(string path)
    {
        _path = path;
    }

    public void OnEpochEnd(int epoch)
    {
        if (File.Exists(_path))
            throw new OperationCanceledException("Training stopped by STOP file");
    }
}
public class ConsoleLoggerCallback : ICallback
{
    public void OnEpochEnd(int epoch)
    {
        Console.WriteLine($"Epoch {epoch} finished");
    }
}
