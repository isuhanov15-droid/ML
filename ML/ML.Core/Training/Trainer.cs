using ML.Core.Data;
using ML.Core.Losses;
using ML.Core.Optimizers;

namespace ML.Core.Training;

public class Trainer
{
    private readonly Network _net;
    private readonly CrossEntropyLoss _loss;
    private readonly AdamOptimizer _opt;
    private readonly bool _useMinMax;

    public Trainer(Network net, double lr = 0.01, bool useMinMax = false)
    {
        _net = net;
        _loss = new CrossEntropyLoss();
        _opt = new AdamOptimizer(lr);
        _useMinMax = useMinMax;

    }

    public void Train(Dataset dataset, TrainerConfig cfg)
    {
        // 1) Auto-load checkpoint
        if (cfg.AutoLoadCheckpoint && File.Exists(cfg.CheckpointPath))
        {
            _net.Load(cfg.CheckpointPath);
            Console.WriteLine("Checkpoint loaded.");

            if (!_net.EndsWithSoftmax)
            {
                _net.EnsureSoftmax();
                Console.WriteLine("Softmax layer appended to loaded checkpoint (compatibility).");
            }
        }

        for (int epoch = 0; epoch < cfg.Epochs; epoch++)
        {
            // STOP (между эпохами)
            if (StopRequested(cfg))
            {
                Console.WriteLine("STOP signal detected. Saving checkpoint and exiting...");

                if (cfg.SaveCheckpoint)
                {
                    _net.Save(cfg.CheckpointPath);
                    Console.WriteLine("Checkpoint saved (STOP).");
                }

                if (cfg.DeleteStopFileOnExit && File.Exists(cfg.StopFileName))
                    File.Delete(cfg.StopFileName);

                return;
            }

            // Перемешивание на эпоху (если добавишь поле в TrainerConfig)
            // Если полей нет — просто закомментируй этот блок или добавь их в cfg.
            var epochData = (cfg.ShuffleEachEpoch ? dataset.Shuffle(cfg.ShuffleSeed + epoch) : dataset);

            double totalLoss = 0.0;
            int seen = 0;

            // ===== FULL-BATCH режим (как сейчас) =====
            if (cfg.BatchSize <= 1)
            {
                int step = 0;

                foreach (var (x, y) in epochData.Samples)
                {
                    // STOP (внутри эпохи) — быстрое реагирование
                    if (cfg.StopCheckInsideEpoch
                        && cfg.StopCheckEverySamples > 0
                        && (step % cfg.StopCheckEverySamples == 0)
                        && StopRequested(cfg))
                    {
                        Console.WriteLine("STOP signal detected during epoch. Saving checkpoint and exiting...");

                        if (cfg.SaveCheckpoint)
                        {
                            _net.Save(cfg.CheckpointPath);
                            Console.WriteLine("Checkpoint saved (STOP).");
                        }

                        if (cfg.DeleteStopFileOnExit && File.Exists(cfg.StopFileName))
                            File.Delete(cfg.StopFileName);

                        return;
                    }

                    step++;

                    var input = x; // позже можно завязать на cfg.UseMinMax / _useMinMax
                    var probs = _net.Forward(input);

                    totalLoss += _loss.Forward(probs, y);
                    _net.Backward(probs, y, _opt);

                    seen++;
                }
            }
            // ===== MINI-BATCH режим =====
            else
            {
                int batchIndex = 0;

                foreach (var batch in epochData.GetBatches(cfg.BatchSize, shuffle: false))
                {
                    // STOP (по батчам)
                    if (cfg.StopCheckInsideEpoch
                        && cfg.StopCheckEverySamples > 0
                        && (batchIndex % Math.Max(1, cfg.StopCheckEverySamples / cfg.BatchSize) == 0)
                        && StopRequested(cfg))
                    {
                        Console.WriteLine("STOP signal detected during epoch (batch). Saving checkpoint and exiting...");

                        if (cfg.SaveCheckpoint)
                        {
                            _net.Save(cfg.CheckpointPath);
                            Console.WriteLine("Checkpoint saved (STOP).");
                        }

                        if (cfg.DeleteStopFileOnExit && File.Exists(cfg.StopFileName))
                            File.Delete(cfg.StopFileName);

                        return;
                    }

                    batchIndex++;

                    foreach (var (x, y) in batch)
                    {
                        var input = x;
                        var probs = _net.Forward(input);

                        totalLoss += _loss.Forward(probs, y);
                        _net.Backward(probs, y, _opt);

                        seen++;
                    }
                }
            }

            var avgLoss = (seen > 0) ? totalLoss / seen : 0.0;

            if (cfg.LogEvery > 0 && epoch % cfg.LogEvery == 0)
                Console.WriteLine($"Epoch {epoch}, Loss {avgLoss}");

            if (cfg.SaveCheckpoint
                && cfg.CheckpointEvery > 0
                && epoch > 0
                && epoch % cfg.CheckpointEvery == 0)
            {
                _net.Save(cfg.CheckpointPath);
                Console.WriteLine("Checkpoint saved.");
            }
        }
    }

    public void Train(Dataset dataset, int epochs, int logEvery = 100)
    {
        var cfg = new TrainerConfig
        {
            Epochs = epochs,
            LogEvery = logEvery
        };

        Train(dataset, cfg);
    }

    private static bool StopRequested(TrainerConfig cfg)
    {
        return cfg.EnableStopFile && File.Exists(cfg.StopFileName);
    }


}
