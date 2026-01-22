namespace ML.Core.Abstractions;
public interface ILoss
{
    // Возвращает скалярное значение loss
    double Forward(double[] logits, int target);

    // Градиент по logits
    double[] Backward();
}
