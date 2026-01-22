namespace ML.Studio.Models;

public sealed class CombinedMetricRow
{
    public int Epoch { get; }
    public double? LossA { get; }
    public double? LossB { get; }

    public CombinedMetricRow(int epoch, double? lossA, double? lossB)
    {
        Epoch = epoch;
        LossA = lossA;
        LossB = lossB;
    }

    public string Display
    {
        get
        {
            string a = LossA.HasValue ? LossA.Value.ToString("F4") : "-";
            string b = LossB.HasValue ? LossB.Value.ToString("F4") : "-";
            return $"epoch={Epoch} lossA={a} lossB={b}";
        }
    }
}
