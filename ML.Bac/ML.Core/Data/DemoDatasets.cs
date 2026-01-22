namespace ML.Core.Data;

public static class DemoDatasets
{
    public static readonly (double[] x, int y)[] Xor =
    {
        (new[] {0.0, 0.0}, 0),
        (new[] {0.0, 1.0}, 1),
        (new[] {1.0, 0.0}, 1),
        (new[] {1.0, 1.0}, 0)
    };

    public static readonly (double[] x, int y)[] And =
    {
        (new[] {0.0, 0.0}, 0),
        (new[] {0.0, 1.0}, 0),
        (new[] {1.0, 0.0}, 0),
        (new[] {1.0, 1.0}, 1),
    };

    public static IEnumerable<(double[] x, int y)> Threshold(double min = 0, double max = 1, double step = 0.1, double threshold = 0.5)
    {
        for (double v = min; v <= max + 1e-9; v += step)
        {
            int cls = v >= threshold ? 1 : 0;
            yield return (new[] { v }, cls);
        }
    }
}
