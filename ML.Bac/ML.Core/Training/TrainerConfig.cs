namespace ML.Core.Training;

public sealed class TrainerConfig
{
    public int Epochs { get; set; }
    public int LogEvery { get; set; }

    // Чекпоинты
    public bool AutoLoadCheckpoint { get; set; } = true;
    public bool SaveCheckpoint { get; set; } = true;
    public int CheckpointEvery { get; set; }
    public string CheckpointPath { get; set; } = "";
    // Мягкая остановка
    public bool EnableStopFile { get; set; } = true;
    public string StopFileName { get; set; } = "STOP";
    public bool DeleteStopFileOnExit { get; set; } = true;

    // Быстрая остановка внутри эпохи (проверка раз в N сэмплов)
    public bool StopCheckInsideEpoch { get; set; } = true;
    public int StopCheckEverySamples { get; set; } = 128;
    public bool ShuffleEachEpoch { get; set; } = true;
    public int ShuffleSeed { get; set; } = 42;

    public int BatchSize { get; set; } = 0; // 0 или 1 = full-batch (как сейчас)

}
