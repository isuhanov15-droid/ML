namespace ML.Core.Inference;

public interface IInferenceSession
{
    void LoadFromFile(string path);
    void LoadFromStore(string modelName);
    double[] Predict(double[] input);
}
