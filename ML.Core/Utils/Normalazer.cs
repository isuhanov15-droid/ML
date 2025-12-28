namespace ML.Core.Utils;

public static class Normalizer
{
    public static double[] MinMax(double[] input)
    {
        double min = double.MaxValue;
        double max = double.MinValue;

        foreach (var v in input)
        {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        double range = max - min;
        if (range == 0) return input;

        var result = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
            result[i] = (input[i] - min) / range;

        return result;
    }
    public static double[] Identity(double[] input)
    {
        return input; // эмоции у нас уже 0..1
    }

}
