namespace ML.Studio.Models;

public sealed class MetricPoint
{
    public int Epoch { get; }
    public double Loss { get; }
    public double? ValLoss { get; }
    public double? Accuracy { get; }
    public DateTimeOffset Utc { get; }

    public MetricPoint(int epoch, double loss, double? valLoss, double? accuracy, DateTimeOffset utc)
    {
        Epoch = epoch;
        Loss = loss;
        ValLoss = valLoss;
        Accuracy = accuracy;
        Utc = utc;
    }

    public string Display
    {
        get
        {
            string acc = Accuracy.HasValue ? Accuracy.Value.ToString("F4") : "-";
            string val = ValLoss.HasValue ? ValLoss.Value.ToString("F4") : "-";
            return $"epoch={Epoch} loss={Loss:F4} val={val} acc={acc}";
        }
    }

    public string TrainLossDisplay => FormatNumber(Loss);
    public string ValLossDisplay => ValLoss.HasValue ? FormatNumber(ValLoss.Value) : "-";
    public string AccDisplay => Accuracy.HasValue ? FormatNumber(Accuracy.Value) : "-";

    private static string FormatNumber(double value)
    {
        return value.ToString("0.0000##", System.Globalization.CultureInfo.InvariantCulture);
    }
}
