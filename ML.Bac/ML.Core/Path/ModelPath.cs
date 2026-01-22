
using System;
using System.IO;

namespace ML.Core.Path; 

public static class ModelPath
{
    public static string ModelsRoot
    {
        get           
        {
            var root = System.IO.Path.Combine(       
                AppContext.BaseDirectory,
                "..", "..", "..", "..", // выходим из bin
                "ML.Models"
            );
            root = System.IO.Path.GetFullPath(root);
            Directory.CreateDirectory(root);
            return root;
        }
    }

    public static string Xor => System.IO.Path.Combine(ModelsRoot, "xor");
    public static string Emotion => System.IO.Path.Combine(ModelsRoot, "emotion");
    public static string Threshold => System.IO.Path.Combine(ModelsRoot, "threshold");
}
