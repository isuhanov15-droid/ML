using System;

namespace ML.Core.Math;

/// <summary>
/// Простая матрица в row-major: Data[r * Cols + c]
/// Без LINQ, без аллокаций внутри операций.
/// </summary>
public sealed class Matrix
{
    public int Rows { get; }
    public int Cols { get; }
    public double[] Data { get; }   // row-major

    public Matrix(int rows, int cols, bool initZeros = true)
    {
        if (rows <= 0) throw new ArgumentException("rows must be > 0", nameof(rows));
        if (cols <= 0) throw new ArgumentException("cols must be > 0", nameof(cols));

        Rows = rows;
        Cols = cols;
        Data = new double[rows * cols];

        if (!initZeros) return;
        // по умолчанию массив уже нулевой
    }

    public double this[int r, int c]
    {
        get => Data[r * Cols + c];
        set => Data[r * Cols + c] = value;
    }

    public static void MatVec(double[] w, int rows, int cols, double[] x, double[] y)
    {
        // y = W*x, где W shape = [rows, cols], w = row-major
        if (x.Length != cols) throw new ArgumentException("x length mismatch");
        if (y.Length != rows) throw new ArgumentException("y length mismatch");

        int idx = 0;
        for (int r = 0; r < rows; r++)
        {
            double sum = 0;
            for (int c = 0; c < cols; c++)
                sum += w[idx++] * x[c];
            y[r] = sum;
        }
    }

    public static void MatTVec(double[] w, int rows, int cols, double[] dy, double[] dx)
    {
        // dx = W^T * dy
        if (dy.Length != rows) throw new ArgumentException("dy length mismatch");
        if (dx.Length != cols) throw new ArgumentException("dx length mismatch");

        Array.Clear(dx, 0, dx.Length);

        // W row-major: w[r*cols + c]
        for (int r = 0; r < rows; r++)
        {
            double g = dy[r];
            int baseIdx = r * cols;
            for (int c = 0; c < cols; c++)
                dx[c] += w[baseIdx + c] * g;
        }
    }

    public static void OuterAdd(double[] dW, int rows, int cols, double[] dy, double[] x)
    {
        // dW += dy ⊗ x   (rows x cols)
        if (dy.Length != rows) throw new ArgumentException("dy length mismatch");
        if (x.Length != cols) throw new ArgumentException("x length mismatch");
        if (dW.Length != rows * cols) throw new ArgumentException("dW length mismatch");

        int idx = 0;
        for (int r = 0; r < rows; r++)
        {
            double g = dy[r];
            for (int c = 0; c < cols; c++)
                dW[idx++] += g * x[c];
        }
    }
}
