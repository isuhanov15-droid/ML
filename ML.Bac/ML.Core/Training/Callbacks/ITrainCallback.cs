namespace ML.Core.Training.Callbacks;

public interface ITrainCallback
{
    void OnEpochEnd(TrainEpochResult r);
}
