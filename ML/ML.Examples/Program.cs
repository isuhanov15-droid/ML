using ML.Examples.Common;

namespace ML.Examples;

internal static class Program
{
    private static void Main()
    {
        Console.WriteLine("ML Examples started");

        var choice = Cli.Menu(
            "Select example",
            "AND (binary logic)",
            "XOR (non-linear logic)",
            "Threshold classification",
            "Emotions (Softmax)"
        );

        Console.WriteLine();

        switch (choice)
        {
            case 1:
                And.Run();
                break;
            case 2:
                Xor.Run();
                break;
            case 3:
                Threshold.Run();
                break;
            case 4:
                Emotion.Run();
                break;
        }

        Console.WriteLine("\nDone. Press any key to exit.");
        Console.ReadKey();
    }
}
