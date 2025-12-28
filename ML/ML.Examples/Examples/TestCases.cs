using System;
using System.Linq;
using ML.Core;
using ML.Core.Data;
using ML.Core.Training;
using ML.Core.Utils;

namespace ML.Examples
{
    static class TestCases
    {
        public static void RunAll()
        {
            //RunAND();
            //RunXOR();
            //RunThreshold();
            RunEmotions();
        }

        public static void RunAND()
        {
            Console.WriteLine("=== TEST 1: AND ===");

            var data = new Dataset(new[]
            {
                (new double[]{0,0}, 0),
                (new double[]{0,1}, 0),
                (new double[]{1,0}, 0),
                (new double[]{1,1}, 1),
            });

            var net = BuildAND();
            var trainer = new Trainer(net, lr: 0.01);

            trainer.Train(data, epochs: 2000, logEvery: 100);
            Evaluate(net, data);
            Console.WriteLine();
        }

        public static void RunXOR()
        {
            Console.WriteLine("=== TEST 2: XOR ===");

            var data = new Dataset(new[]
            {
                (new double[]{0,0}, 0),
                (new double[]{0,1}, 1),
                (new double[]{1,0}, 1),
                (new double[]{1,1}, 0),
            });

            var net = BuildXOR();
            var trainer = new Trainer(net, lr: 0.01);

            trainer.Train(data, epochs: 2000, logEvery: 100);
            Evaluate(net, data);
            Console.WriteLine();
        }

        public static void RunThreshold()
        {
            Console.WriteLine("=== TEST 3: Threshold classification ===");

            var data = new Dataset(new[]
            {
                (new double[]{0.2,0.3,0.1,0.2}, 0),
                (new double[]{0.5,0.6,0.4,0.2}, 1),
                (new double[]{1.0,0.8,0.9,0.7}, 2),
                (new double[]{0.9,0.3,0.2,0.1}, 0),
                (new double[]{0.6,0.6,0.6,0.5}, 2),  // можешь тут класс подправить по своей логике
                (new double[]{0.9,0.9,0.9,0.9}, 2),
            });

            var net = BuildThreshold();
            var trainer = new Trainer(net, lr: 0.01);

            trainer.Train(data, epochs: 2000, logEvery: 100);
            Evaluate(net, data);
            Console.WriteLine();
        }

        // -------------------- Архитектуры --------------------

        private static Network BuildAND()
        {
            var net = new Network(2);
            net.AddInputLayer();
            net.AddHiddenLayer(4, ActivationType.ReLu);
            net.AddOutputLayer(2, ActivationType.Linear);
            net.AddSoftmax();
            return net;
        }

        private static Network BuildXOR()
        {
            var net = new Network(2);
            net.AddInputLayer();
            net.AddHiddenLayer(8, ActivationType.ReLu);
            net.AddOutputLayer(2, ActivationType.Linear);
            net.AddSoftmax();
            return net;
        }

        private static Network BuildThreshold()
        {
            var net = new Network(4);
            net.AddInputLayer();
            net.AddHiddenLayer(8, ActivationType.ReLu);
            net.AddHiddenLayer(8, ActivationType.ReLu);
            net.AddOutputLayer(3, ActivationType.Linear);
            net.AddSoftmax();
            return net;
        }

        // -------------------- Evaluate --------------------

        private static void Evaluate(Network net, Dataset data)
        {
            foreach (var (x, y) in data.Samples)
            {
                var input = Normalizer.MinMax(x);
                var probs = net.Forward(input);

                int pred = ArgMax(probs);

                Console.WriteLine(
                    $"Input [{string.Join(",", x)}] " +
                    $"→ predicted={pred}, target={y}, probs=[{string.Join(", ", probs.Select(p => p.ToString("F3")))}]"
                );
            }
        }
        public static void RunEmotions()
        {
            Console.WriteLine("=== TEST 4: Emotions (Softmax) ===");

            var samples = Emotion.GenerateSamples(count: 20000, seed: 10, noiseStd: 0.03);
            var data = new Dataset(samples);

            var net = BuildEmotions();
            var trainer = new Trainer(net, lr: 0.0005, useMinMax: false);

            trainer.Train(data, epochs: 1500, logEvery: 100);

            // ручной ввод (как ты просил)
            Console.WriteLine("\n--- Manual input test ---");
            var manual = Emotion.Make(
                threat: 0.1,
                loss: 0.5,
                physicalDiscomfort: 0.1,
                socialSupport: 0.6,
                energy: 0.5,
                control: 0.6,
                surprise: 0.2,
                humor: 0.2,
                goalProgress: 0.4,
                fatigue: 0.5
            );

            var probs = net.Forward(manual); // без Normalizer
            Console.WriteLine($"Predicted: {Emotion.Names[Emotion.ArgMax(probs)]}");
            Emotion.PrintTopK(probs, k: 5);

            // несколько сценариев для кайфа
            Console.WriteLine("\n--- Scenarios ---");
            var scenarios = new (string Name, double[] X)[]
            {
        ("WarmSuccess", Emotion.Make(0.1,0.1,0.1, 0.9,0.8,0.8, 0.2,0.3,0.9, 0.2)),
        ("DarkNoiseAlone", Emotion.Make(0.9,0.1,0.2, 0.1,0.6,0.2, 0.7,0.1,0.4, 0.3)),
        ("HeavyLossTired", Emotion.Make(0.3,0.9,0.2, 0.2,0.2,0.3, 0.2,0.0,0.2, 0.9)),
        ("JokeAndSurprise", Emotion.Make(0.05,0.1,0.0, 0.6,0.8,0.7, 0.9,0.95,0.6, 0.2)),
        ("PhysicalPain", Emotion.Make(0.2,0.1,0.95, 0.3,0.3,0.5, 0.2,0.0,0.3, 0.7)),
        ("BlockedAndIrritated", Emotion.Make(0.6,0.2,0.1, 0.3,0.8,0.15, 0.3,0.1,0.1, 0.4)),
            };

            foreach (var (name, x) in scenarios)
            {
                var p = net.Forward(x);
                Console.WriteLine($"\nScenario: {name}");
                Console.WriteLine($"Predicted: {Emotion.Names[Emotion.ArgMax(p)]}");
                Emotion.PrintTopK(p, 3);
            }

            Console.WriteLine();
        }

        private static Network BuildEmotions()
        {
            var net = new Network(Emotion.InputSize);
            net.AddInputLayer();
            net.AddHiddenLayer(32, ActivationType.ReLu);
            net.AddHiddenLayer(16, ActivationType.ReLu);
            net.AddOutputLayer(Emotion.Classes, ActivationType.Linear);
            net.AddSoftmax();
            return net;
        }


        private static int ArgMax(double[] v)
        {
            int idx = 0;
            double max = v[0];
            for (int i = 1; i < v.Length; i++)
            {
                if (v[i] > max)
                {
                    max = v[i];
                    idx = i;
                }
            }
            return idx;
        }
    }
}
