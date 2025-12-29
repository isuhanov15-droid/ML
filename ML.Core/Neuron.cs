using System;

namespace ML.Core;


public class Neuron
{
    public NeuronType Type { get; init; }            //Хранит тип нейрона 
    public double[] Weights { get; set; }    //Хранит веса нейрона
    public double[] LastInput { get; private set; } = Array.Empty<double>();

    public double Bias { get; set; }         //Коэффицент смещения

    public ActivationType ActType { get; init; }     //Хранит функцию активации нейрона

    public double Z { get; private set; }   // сумма до активации
    public double A { get; private set; }   // после активации

    public double[] WeightGradients { get; set; } // Хранит градиенты весов
    public double BiasGradient { get; set; }      // Хранит градиент смещения
    public bool IsActive { get; set; } = true;    // Хранит состояние активаниции (да/нет)

    private static readonly Random Rnd = new Random();

    // Конструктор класса 
    public Neuron(int inputCount, NeuronType neuronType, ActivationType activationType)
    {
        Type = neuronType;

        ActType = activationType;
        // Высчитываем значение для случайного веса
        double scale = ActType == ActivationType.ReLu
        ? Math.Sqrt(2.0 / (inputCount))
        : Math.Sqrt(1.0 / (inputCount));
        //Выстовляем случайный коэфицент смещения
        Bias = Rnd.NextDouble() * 2 * scale - scale;
        //Выстовляем случайные веса
        Weights = new double[inputCount];
        for (int i = 0; i < Weights.Length; i++)
        {
            Weights[i] = Rnd.NextDouble() * 2 * scale - scale;
        }

        WeightGradients = new double[inputCount];

    }
    //Функция прохода по одному нейрону
    public double Forward(double[] inputs)
    {
        if (inputs.Length != Weights.Length)
            throw new ArgumentException("Input size does not match weights size");

        LastInput = (double[])inputs.Clone(); // ✅ вместо ссылки

        double sum = Bias;
        for (int i = 0; i < inputs.Length; i++)
            sum += inputs[i] * Weights[i];

        Z = sum;
        A = Activate(Z);
        return A;
    }

    //Выбираем функцию активации в зависимости от типа активации нейрона
    private double Activate(double x) => ActType switch
    {
        ActivationType.Linear => x,
        ActivationType.Sigmoid => 1.0 / (1.0 + Math.Exp(-x)),
        ActivationType.Tanh => Math.Tanh(x),
        ActivationType.ReLu => Math.Max(0, x),
        ActivationType.LeakyReLu => x > 0 ? x : 0.01 * x,
        ActivationType.Elu => x >= 0 ? x : Math.Exp(x) - 1,
        ActivationType.Gelu => 0.5 * x * (1 + Math.Tanh(
                                          Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))),
        ActivationType.Swish => x / (1 + Math.Exp(-x)),
        ActivationType.Softplus => x > 20 ? x : Math.Log(1 + Math.Exp(x)),
        _ => x
    };
    //--------------------------------------Функции Активации нерона по типам------------------------
    private static double Linear(double x) => x;

    private static double Sigmoid(double x) =>
        1.0 / (1.0 + Math.Exp(-x));
    private static double Tanh(double x) =>
    Math.Tanh(x);

    private static double ReLu(double x) =>
        x > 0 ? x : 0;
    private static double LeakyReLu(double x, double alpha = 0.01) =>
    x > 0 ? x : alpha * x;
    private static double Elu(double x, double alpha = 1.0) =>
    x >= 0 ? x : alpha * (Math.Exp(x) - 1);
    private static double Gelu(double x) =>
    0.5 * x * (1 + Math.Tanh(
        Math.Sqrt(2 / Math.PI) * (x + 0.044715 * Math.Pow(x, 3))
    ));
    private static double Swish(double x) =>
    x * Sigmoid(x);
    private static double Softplus(double x) =>
    x > 20 ? x : Math.Log(1 + Math.Exp(x));

    //-----------------------------------------------------------------------------------------------

    // Производная функции активвации
    public double ActivationDerivative() => ActType switch
    {
        ActivationType.Linear => 1,
        ActivationType.Sigmoid => A * (1 - A),
        ActivationType.Tanh => 1 - A * A,
        ActivationType.ReLu => Z > 0 ? 1 : 0,
        ActivationType.LeakyReLu => Z > 0 ? 1 : 0.01,
        ActivationType.Elu => Z >= 0 ? 1 : Math.Exp(Z),
        ActivationType.Softplus => Sigmoid(Z),
        _ => 1
    };
}

//Колекция типов неронов 
public enum NeuronType
{
    Input,
    Hidden,
    Output
}


//Колекция функций активации
public enum ActivationType
{
    Linear,
    Sigmoid,
    Tanh,
    ReLu,
    LeakyReLu,
    Elu,
    Gelu,
    Swish,
    Softplus
}
