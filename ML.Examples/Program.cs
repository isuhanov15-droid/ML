using ML.Examples.Common;

namespace ML.Examples;

internal static class Program
{
    private static void Main()
    {
        Console.WriteLine("ML Example Emotion started");


        Console.WriteLine();

        Emotion.Run();

        Console.WriteLine("\nDone. Press any key to exit.");
        Console.ReadKey();
    }
}
