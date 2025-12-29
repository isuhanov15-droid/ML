using ML.Core.Training.Collbacks;
using ML.Core.Abstractions;

namespace ML.Core.Training
{
    public class Trainer
    {
        private readonly IModel _model;
        private readonly IOptimizer _optimizer;
        private readonly ILoss _loss;

        public Trainer(IModel model, IOptimizer optimizer, ILoss loss)
        {
            _model = model;
            _optimizer = optimizer;
            _loss = loss;
        }

        public void Train(
            IEnumerable<(double[] x, int y)> dataset,
            int epochs,
            IEnumerable<ICallback> callbacks)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var (x, y) in dataset)
                {
                    var logits = _model.Forward(x, training: true);
                    var lossValue = _loss.Forward(logits, y);

                    var grad = _loss.Backward();
                    _model.Backward(grad);

                    _optimizer.Step(_model.Parameters());
                    foreach (var p in _model.Parameters())
                    {
                        if (p is ML.Core.Abstractions.Parameter pp)
                            pp.Sync();
                    }

                    _optimizer.ZeroGrad(_model.Parameters());
                }

                foreach (var cb in callbacks)
                    cb.OnEpochEnd(epoch);
            }
        }
    }
}
