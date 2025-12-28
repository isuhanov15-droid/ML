using ML.Core.Abstractions;
namespace ML.Core.Optimizers;


public class AdamOptimizer : IOptimizer
{
    private readonly double _lr;
    private readonly double _beta1;
    private readonly double _beta2;
    private readonly double _eps;

    private int _t = 0;

    public int StepIndex => _t;
    public void SetStepIndex(int t) => _t = t;


    private readonly Dictionary<double[], double[]> _m = new();
    private readonly Dictionary<double[], double[]> _v = new();

    private readonly Dictionary<Neuron, double> _mb = new();
    private readonly Dictionary<Neuron, double> _vb = new();

    public AdamOptimizer(
        double learningRate = 0.001,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8)
    {
        _lr = learningRate;
        _beta1 = beta1;
        _beta2 = beta2;
        _eps = epsilon;
    }

    public void Step(Neuron neuron)
    {
        if (!_m.ContainsKey(neuron.Weights))
        {
            _m[neuron.Weights] = new double[neuron.Weights.Length];
            _v[neuron.Weights] = new double[neuron.Weights.Length];
            _mb[neuron] = 0.0;
            _vb[neuron] = 0.0;
        }

        for (int i = 0; i < neuron.Weights.Length; i++)
        {
            double g = neuron.WeightGradients[i];

            _m[neuron.Weights][i] = _beta1 * _m[neuron.Weights][i] + (1 - _beta1) * g;
            _v[neuron.Weights][i] = _beta2 * _v[neuron.Weights][i] + (1 - _beta2) * g * g;

            double mHat = _m[neuron.Weights][i] / (1 - Math.Pow(_beta1, _t + 1));
            double vHat = _v[neuron.Weights][i] / (1 - Math.Pow(_beta2, _t + 1));

            neuron.Weights[i] -= _lr * mHat / (Math.Sqrt(vHat) + _eps);
        }

        // bias
        double gb = neuron.BiasGradient;

        _mb[neuron] = _beta1 * _mb[neuron] + (1 - _beta1) * gb;
        _vb[neuron] = _beta2 * _vb[neuron] + (1 - _beta2) * gb * gb;

        double mbHat = _mb[neuron] / (1 - Math.Pow(_beta1, _t + 1));
        double vbHat = _vb[neuron] / (1 - Math.Pow(_beta2, _t + 1));

        neuron.Bias -= _lr * mbHat / (Math.Sqrt(vbHat) + _eps);
    }

    public void NextStep() => _t++;
}
