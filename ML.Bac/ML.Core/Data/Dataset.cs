using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Core.Data;

public partial class Dataset
{
    public (double[] X, int Y)[] Samples { get; }

    public Dataset((double[], int)[] samples)
    {
        Samples = samples;
    }
    public Dataset Shuffle(int seed = 42)
    {
        var rnd = new Random(seed);
        var arr = Samples.ToArray();

        for (int i = arr.Length - 1; i > 0; i--)
        {
            int j = rnd.Next(i + 1);
            (arr[i], arr[j]) = (arr[j], arr[i]);
        }

        return new Dataset(arr);
    }

    public (Dataset Train, Dataset Test) Split(double trainRatio = 0.8, int seed = 42)
    {
        if (trainRatio <= 0 || trainRatio >= 1)
            throw new ArgumentOutOfRangeException(nameof(trainRatio), "trainRatio must be in (0,1)");

        var shuffled = Shuffle(seed);
        var arr = shuffled.Samples.ToArray();

        int trainCount = (int)System.Math.Round(arr.Length * trainRatio);
        trainCount = System.Math.Clamp(trainCount, 1, arr.Length - 1);
        var train = arr.Take(trainCount).ToArray();
        var test = arr.Skip(trainCount).ToArray();

        return (new Dataset(train), new Dataset(test));
    }

    public IEnumerable<(double[] X, int Y)[]> GetBatches(int batchSize, bool shuffle = true, int seed = 42)
    {
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));

        var arr = (shuffle ? Shuffle(seed) : this).Samples.ToArray();

        for (int i = 0; i < arr.Length; i += batchSize)
        {
            int len = System.Math.Min(batchSize, arr.Length - i);
            var batch = new (double[] X, int Y)[len];
            Array.Copy(arr, i, batch, 0, len);
            yield return batch;
        }
    }
}
