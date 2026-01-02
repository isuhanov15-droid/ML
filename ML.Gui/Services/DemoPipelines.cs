namespace ML.Gui.Services;

internal static class DemoPipelines
{
    // Простейший XOR, чтобы приложение "заводилось" из коробки.
    public static readonly (double[] x, int y)[] XorData =
    {
        (new[] {0.0, 0.0}, 0),
        (new[] {0.0, 1.0}, 1),
        (new[] {1.0, 0.0}, 1),
        (new[] {1.0, 1.0}, 0)
    };
}
