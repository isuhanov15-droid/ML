using ML.Core.Data;

namespace ML.Core.Utils;

public static class Metrics
{
    public static double Accuracy(Network net, Dataset data)
    {
        int correct = 0, total = 0;
        foreach (var (x, y) in data.Samples)
        {
            var p = net.Forward(x);
            int pred = ArgMax(p);
            if (pred == y) correct++;
            total++;
        }
        return total == 0 ? 0 : (double)correct / total;
    }

    private static int ArgMax(double[] v)
    {
        int idx = 0;
        double max = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > max) { max = v[i]; idx = i; }
        return idx;
    }
}
