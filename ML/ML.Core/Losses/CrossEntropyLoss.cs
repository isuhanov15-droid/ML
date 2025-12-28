namespace ML.Core.Losses;

public interface ILoss
{
    double Forward(double[] predicted, int targetClass);
}




public class CrossEntropyLoss : ILoss
{
    private const double Epsilon = 1e-15; // защита от log(0)

    public double Forward(double[] predicted, int targetClass)
    {
        if (predicted == null || predicted.Length == 0)
            throw new ArgumentException("Predicted array is empty");

        if (targetClass < 0 || targetClass >= predicted.Length)
            throw new ArgumentOutOfRangeException(nameof(targetClass));

        // Берём вероятность правильного класса
        double p = predicted[targetClass];

        // Клиппинг для численной устойчивости
        p = Math.Max(Epsilon, Math.Min(1.0 - Epsilon, p));

        // Cross-Entropy
        return -Math.Log(p);
    }
}
