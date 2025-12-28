using System;

namespace ML.Examples.Common;

public static class Cli
{
    public static int Menu(string title, params string[] options)
    {
        Console.WriteLine();
        Console.WriteLine(title);
        Console.WriteLine(new string('-', title.Length));

        for (int i = 0; i < options.Length; i++)
            Console.WriteLine($"{i + 1}. {options[i]}");

        Console.Write("\nSelect option: ");

        while (true)
        {
            var input = Console.ReadLine();
            if (int.TryParse(input, out int choice)
                && choice >= 1
                && choice <= options.Length)
            {
                return choice;
            }

            Console.Write("Invalid input. Try again: ");
        }
    }
}
