using ML.Core.Abstractions;
namespace ML.Core.Layers;

public class SoftmaxLayer : ILayer
{
    public int InputSize { get; }
    public int OutputSize => InputSize;

    // Кэш (пригодится позже для обучения/градиентов)
    public double[] LastInput { get; private set; } = Array.Empty<double>();
    public double[] LastOutput { get; private set; } = Array.Empty<double>();

    public SoftmaxLayer(int size)
    {
        if (size <= 0)
            throw new ArgumentException("size must be > 0");

        InputSize = size;
    }

    public double[] Forward(double[] input)
    {
        if (input.Length != InputSize)
            throw new ArgumentException("Input size mismatch for SoftmaxLayer");

        LastInput = (double[])input.Clone();

        // 1) max trick для численной устойчивости
        double max = input[0];
        for (int i = 1; i < input.Length; i++)
            if (input[i] > max) max = input[i];

        // 2) exp + сумма
        var exp = new double[input.Length];
        double sum = 0.0;

        for (int i = 0; i < input.Length; i++)
        {
            exp[i] = Math.Exp(input[i] - max);
            sum += exp[i];
        }

        // 3) нормализация
        var output = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
            output[i] = exp[i] / sum;

        LastOutput = output;
        return output;
    }
}
