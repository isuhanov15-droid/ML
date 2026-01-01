using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using ML.Core;
using ML.Core.Layers;
using ML.Core.Losses;
using ML.Core.Optimizers;
using ML.Core.Training;
using ML.Core.Training.Callbacks;

namespace ML.Examples;

static class Emotion
{
    // =========================================================
    // Канон под текущее ядро:
    // - Save/Load: по modelName -> ML/Models/<name>.json (ModelStore)
    // - Trainer: Train(dataset, TrainOptions)
    // - Метрики (accuracy) НЕ в Core, а в Examples через callbacks
    // =========================================================

    private const string ModelName = "emotion_ru";   // сохранится как ML/Models/emotion_ru.json

    // =========================
    // 1) Эмоции (16 классов)
    // =========================
    public enum E
    {
        Neutral = 0,
        Joy = 1,
        Smile = 2,
        Laugh = 3,
        Gratitude = 4,
        Pride = 5,
        Interest = 6,
        Calm = 7,

        Sadness = 8,
        Suffering = 9,
        Fear = 10,
        Anger = 11,
        Disgust = 12,
        Shame = 13,
        Guilt = 14,
        Loneliness = 15
    }

    public static readonly string[] Names =
    {
        "Нейтрально", "Радость", "Улыбка", "Смех", "Благодарность", "Гордость", "Интерес", "Спокойствие",
        "Грусть", "Страдание", "Страх", "Злость", "Отвращение", "Стыд", "Вина", "Одиночество"
    };

    public const int Classes = 16;

    // =========================
    // 2) Фичи текста (52)
    // =========================
    // [0..15]  - лексиконы эмоций/сигналов
    // [16..31] - форма/пунктуация/капс/длина/повторы/эмодзи
    // [32..51] - грамматические/контекстные признаки
    public const int InputSize = 52;

    private static readonly Regex TokenRx = new(@"[^\p{L}\p{Nd}]+", RegexOptions.Compiled);

    private static string[] Tokenize(string text)
    {
        text ??= "";
        text = text.ToLowerInvariant();
        text = TokenRx.Replace(text, " ").Trim();
        if (text.Length == 0) return Array.Empty<string>();
        return text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    private static double Clamp01(double v) => v < 0 ? 0 : (v > 1 ? 1 : v);

    // =========================
    // 3) Safety gate (до ML)
    // =========================
    private static readonly HashSet<string> LexViolence = new(StringComparer.OrdinalIgnoreCase)
    {
        "убью","убить","зарежу","зарезать","пристрелю","застрелю","расстреляю","сломаю","изобью","ударю",
        "взорву","взрыв","прибью","задушу","повешу","порежу"
    };

    private static readonly HashSet<string> LexSelfHarm = new(StringComparer.OrdinalIgnoreCase)
    {
        "суицид","самоубийство","убьюсь","умру","покончу","вскрою","вскроюсь","повешусь"
    };

    private static bool IsSafetyCritical(string text)
    {
        var t = Tokenize(text);
        if (t.Any(x => LexViolence.Contains(x))) return true;
        if (t.Any(x => LexSelfHarm.Contains(x))) return true;
        return false;
    }

    private static void PrintSafetyResponse()
    {
        Console.WriteLine("Эмоция: Страх");
        Console.WriteLine("Ответ:  Я не поддерживаю угрозы/насилие или самоповреждение. Давай остановимся и переведём разговор в безопасный формат.");
        Console.WriteLine();
    }

    // =========================
    // 4) Лексиконы
    // =========================
    private static readonly HashSet<string> LxJoy = new(StringComparer.OrdinalIgnoreCase)
    { "ура","класс","кайф","рад","счастлив","счастлива","победа","вышло","получилось","удалось","круто","огонь" };

    private static readonly HashSet<string> LxSmile = new(StringComparer.OrdinalIgnoreCase)
    { "приятно","тепло","улыбаюсь","улыбка","мило","хорошо","уютно","лампово","светло","привет" };

    private static readonly HashSet<string> LxLaugh = new(StringComparer.OrdinalIgnoreCase)
    { "ахаха","хаха","лол","ржу","смешно","прикол","шутка","угар","оруу" };

    private static readonly HashSet<string> LxGratitude = new(StringComparer.OrdinalIgnoreCase)
    { "спасибо","благодарю","признателен","признательна","ценю","спасиб","благодарность" };

    private static readonly HashSet<string> LxPride = new(StringComparer.OrdinalIgnoreCase)
    { "горжусь","горд","горда","достиг","достигла","смог","смогла","сделал","сделала","выдержал" };

    private static readonly HashSet<string> LxInterest = new(StringComparer.OrdinalIgnoreCase)
    { "интересно","любопытно","хочу","узнать","почему","как","что","разберемся","посмотрим","идея" };

    private static readonly HashSet<string> LxCalm = new(StringComparer.OrdinalIgnoreCase)
    { "спокойно","тихо","ровно","стабильно","уверенно","норм","нормально","ок","выдох","пауза" };

    private static readonly HashSet<string> LxSadness = new(StringComparer.OrdinalIgnoreCase)
    { "грустно","печально","тоска","слезы","плачу","пусто","жалко","скучаю","уныло" };

    private static readonly HashSet<string> LxSuffering = new(StringComparer.OrdinalIgnoreCase)
    { "больно","страдаю","тяжело","невыносимо","плохо","разбит","выжат","выгорание","нетсил","нет" };

    private static readonly HashSet<string> LxFear = new(StringComparer.OrdinalIgnoreCase)
    { "страшно","опасно","ужас","паника","пугает","угроза","кошмар","боюсь","жутко","обстрел","взрыв" };

    private static readonly HashSet<string> LxAnger = new(StringComparer.OrdinalIgnoreCase)
    { "злюсь","бесит","раздражает","достало","ярость","взбесило","ненавижу","сука","идиот" };

    private static readonly HashSet<string> LxDisgust = new(StringComparer.OrdinalIgnoreCase)
    { "фу","противно","мерзко","отвратительно","тошно","воняет","гадость","пакость" };

    private static readonly HashSet<string> LxShame = new(StringComparer.OrdinalIgnoreCase)
    { "стыдно","стыд","позор","неловко","опозорился","опозорилась","смущаюсь" };

    private static readonly HashSet<string> LxGuilt = new(StringComparer.OrdinalIgnoreCase)
    { "виноват","виновата","вина","прости","извини","простите","сожалею","неправ" };

    private static readonly HashSet<string> LxLoneliness = new(StringComparer.OrdinalIgnoreCase)
    { "один","одна","одинок","одиноко","никого","пусто","без тебя","не с кем" };

    private static readonly HashSet<string> LxNegation = new(StringComparer.OrdinalIgnoreCase)
    { "не","нет","никогда","ни","ничего","никак" };

    private static readonly HashSet<string> LxQuestion = new(StringComparer.OrdinalIgnoreCase)
    { "как","почему","зачем","что","когда","где","кто","сколько","ли" };

    private static readonly HashSet<string> LxFirstPerson = new(StringComparer.OrdinalIgnoreCase)
    { "я","мне","меня","мой","моя","мои" };

    private static readonly HashSet<string> LxSecondPerson = new(StringComparer.OrdinalIgnoreCase)
    { "ты","тебе","тебя","твой","твоя","твои","вы","вам","вас" };

    private static readonly HashSet<string> LxSupport = new(StringComparer.OrdinalIgnoreCase)
    { "рядом","обнимаю","держись","помогу","вместе","поддержу","семья","друг","друзья" };

    private static int CountLex(IEnumerable<string> tokens, HashSet<string> lex)
    {
        int c = 0;
        foreach (var t in tokens)
            if (lex.Contains(t)) c++;
        return c;
    }

    private static int CountAny(IEnumerable<string> tokens, params string[] words)
    {
        int c = 0;
        foreach (var t in tokens)
            for (int i = 0; i < words.Length; i++)
                if (t.Equals(words[i], StringComparison.OrdinalIgnoreCase)) { c++; break; }
        return c;
    }

    // =========================
    // 5) Features
    // =========================
    public static double[] TextToFeatures(string text)
    {
        text ??= "";
        var tokens = Tokenize(text);

        int joy = CountLex(tokens, LxJoy);
        int smile = CountLex(tokens, LxSmile);
        int laugh = CountLex(tokens, LxLaugh);
        int grat = CountLex(tokens, LxGratitude);
        int pride = CountLex(tokens, LxPride);
        int interest = CountLex(tokens, LxInterest);
        int calm = CountLex(tokens, LxCalm);

        int sad = CountLex(tokens, LxSadness);
        int suffering = CountLex(tokens, LxSuffering);
        int fear = CountLex(tokens, LxFear);
        int anger = CountLex(tokens, LxAnger);
        int disgust = CountLex(tokens, LxDisgust);
        int shame = CountLex(tokens, LxShame);
        int guilt = CountLex(tokens, LxGuilt);
        int lonely = CountLex(tokens, LxLoneliness);

        const double scale = 3.0;

        int len = text.Length;
        int words = tokens.Length;

        int exclam = text.Count(ch => ch == '!');
        int quest = text.Count(ch => ch == '?');
        int dots = CountSubstring(text, "...");
        int comma = text.Count(ch => ch == ',');
        int quotes = text.Count(ch => ch == '"' || ch == '«' || ch == '»');

        int upperLetters = text.Count(ch => char.IsLetter(ch) && char.IsUpper(ch));
        int letters = text.Count(ch => char.IsLetter(ch));
        double capsRatio = letters == 0 ? 0.0 : (double)upperLetters / letters;

        int repeats = CountCharRepeats(text);
        int smiles = CountSmiles(text);
        int emojis = CountEmojiLike(text);

        int neg = CountLex(tokens, LxNegation);
        int qwords = CountLex(tokens, LxQuestion);
        int fp = CountLex(tokens, LxFirstPerson);
        int sp = CountLex(tokens, LxSecondPerson);
        int support = CountLex(tokens, LxSupport);

        int past = CountAny(tokens, "вчера", "было", "была", "был", "потерял", "потеряла", "сделал", "сделала", "успел", "успела");
        int future = CountAny(tokens, "завтра", "будет", "буду", "сделаю", "сделаем", "план");
        int now = CountAny(tokens, "сейчас", "сегодня", "вот", "прям", "именно");

        int intens = CountAny(tokens, "очень", "капец", "сильно", "реально", "жесть", "просто", "пипец");

        var x = new double[InputSize];

        // 0..15: эмо-лексиконы + баланс
        x[0]  = Clamp01(joy / scale);
        x[1]  = Clamp01(smile / scale);
        x[2]  = Clamp01(laugh / scale);
        x[3]  = Clamp01(grat / scale);
        x[4]  = Clamp01(pride / scale);
        x[5]  = Clamp01(interest / scale);
        x[6]  = Clamp01(calm / scale);

        x[7]  = Clamp01(sad / scale);
        x[8]  = Clamp01(suffering / scale);
        x[9]  = Clamp01(fear / scale);
        x[10] = Clamp01(anger / scale);
        x[11] = Clamp01(disgust / scale);
        x[12] = Clamp01(shame / scale);
        x[13] = Clamp01(guilt / scale);
        x[14] = Clamp01(lonely / scale);

        double pos = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6];
        double negv = x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14];
        x[15] = Clamp01(0.5 + 0.25 * (pos - negv));

        // 16..31: форма
        x[16] = Clamp01(len / 120.0);
        x[17] = Clamp01(words / 20.0);
        x[18] = Clamp01(exclam / 5.0);
        x[19] = Clamp01(quest / 5.0);
        x[20] = Clamp01(dots / 3.0);
        x[21] = Clamp01(comma / 8.0);
        x[22] = Clamp01(quotes / 6.0);
        x[23] = Clamp01(capsRatio);
        x[24] = Clamp01(repeats / 10.0);
        x[25] = Clamp01(smiles / 6.0);
        x[26] = Clamp01(emojis / 6.0);

        // запас
        x[27] = Clamp01((exclam + quest) / 6.0);
        x[28] = Clamp01((repeats + emojis) / 10.0);
        x[29] = Clamp01((len > 0 && text.Trim().EndsWith(")") ? 1.0 : 0.0));
        x[30] = Clamp01((len > 0 && text.Trim().EndsWith(".") ? 1.0 : 0.0));
        x[31] = Clamp01((len > 0 && text.Trim().EndsWith("!") ? 1.0 : 0.0));

        // 32..51: контекст
        x[32] = Clamp01(neg / 3.0);
        x[33] = Clamp01(qwords / 3.0);
        x[34] = Clamp01(fp / 3.0);
        x[35] = Clamp01(sp / 3.0);
        x[36] = Clamp01(support / 3.0);

        x[37] = Clamp01(past / 2.0);
        x[38] = Clamp01(future / 2.0);
        x[39] = Clamp01(now / 2.0);
        x[40] = Clamp01(intens / 3.0);

        // немного “пустых” слотов под будущее расширение
        for (int i = 41; i < InputSize; i++)
            x[i] = 0.0;

        return x;
    }

    private static int CountSubstring(string s, string sub)
    {
        if (string.IsNullOrEmpty(s) || string.IsNullOrEmpty(sub)) return 0;
        int c = 0;
        int idx = 0;
        while (true)
        {
            idx = s.IndexOf(sub, idx, StringComparison.Ordinal);
            if (idx < 0) break;
            c++;
            idx += sub.Length;
        }
        return c;
    }

    private static int CountCharRepeats(string s)
    {
        if (string.IsNullOrEmpty(s)) return 0;
        int best = 0;
        int cur = 1;
        for (int i = 1; i < s.Length; i++)
        {
            if (s[i] == s[i - 1]) cur++;
            else { best = Math.Max(best, cur); cur = 1; }
        }
        best = Math.Max(best, cur);
        return best >= 3 ? best : 0;
    }

    private static int CountSmiles(string s)
    {
        if (string.IsNullOrEmpty(s)) return 0;
        int c = 0;
        c += CountSubstring(s, ":)");
        c += CountSubstring(s, ":-)");
        c += CountSubstring(s, ":D");
        c += CountSubstring(s, "))");
        c += CountSubstring(s, ")))");
        return c;
    }

    private static int CountEmojiLike(string s)
    {
        if (string.IsNullOrEmpty(s)) return 0;
        // грубо: считаем символы из диапазона эмодзи-плоскостей
        int c = 0;
        foreach (var ch in s)
        {
            if (ch >= 0x2600 && ch <= 0x27BF) c++;
            if (ch >= 0x1F300 && ch <= 0x1FAFF) c++;
        }
        return c;
    }

    // =========================
    // 6) Синтетический датасет
    // =========================
    private sealed record Sample(double[] x, int y);

    private static Sample[] GenerateSamples(int count, int seed, double noiseStd)
    {
        var rnd = new Random(seed);

        // шаблоны для генерации
        var lex = BuildTemplates();

        var samples = new Sample[count];
        for (int i = 0; i < count; i++)
        {
            // равномерно по классам
            int y = i % Classes;
            var phrase = lex[(E)y][rnd.Next(lex[(E)y].Count)];

            var x = TextToFeatures(phrase);

            // добавим шума на фичи — чтобы не было “табличного” переобучения
            if (noiseStd > 0)
            {
                for (int k = 0; k < x.Length; k++)
                {
                    x[k] += NextGaussian(rnd, 0, noiseStd);
                    if (x[k] < 0) x[k] = 0;
                    if (x[k] > 1) x[k] = 1;
                }
            }

            samples[i] = new Sample(x, y);
        }

        // перемешаем
        samples = samples.OrderBy(_ => rnd.Next()).ToArray();
        return samples;
    }

    private static Dictionary<E, List<string>> BuildTemplates()
    {
        return new Dictionary<E, List<string>>
        {
            [E.Neutral] = new() { "ок", "понял", "нормально", "ясно", "ладно", "не знаю", "посмотрим" },
            [E.Joy] = new() { "ура!", "кайф", "класс!", "я счастлив", "получилось!", "это победа", "огонь!" },
            [E.Smile] = new() { "привет", "приятно", "улыбаюсь", "тепло", "мило", "уютно" },
            [E.Laugh] = new() { "ахаха", "ржу", "смешно", "лол", "угар", "оруу" },
            [E.Gratitude] = new() { "спасибо", "благодарю", "очень ценю", "ты выручила", "признателен" },
            [E.Pride] = new() { "я горжусь", "я справился", "я сделал это", "выдержал", "смог" },
            [E.Interest] = new() { "интересно", "как это работает?", "почему так?", "давай разберемся", "есть идея" },
            [E.Calm] = new() { "я спокоен", "всё тихо", "ровно", "выдох", "пауза", "стабильно" },

            [E.Sadness] = new() { "мне грустно", "печально", "тоскливо", "пусто", "хочется плакать" },
            [E.Suffering] = new() { "очень тяжело", "нет сил", "больно", "выгорел", "я на пределе" },
            [E.Fear] = new() { "страшно", "паника", "мне жутко", "это пугает", "опасно" },
            [E.Anger] = new() { "меня бесит", "я зол", "ненавижу", "достало", "в ярости" },
            [E.Disgust] = new() { "фу", "противно", "мерзко", "отвратительно", "гадость" },
            [E.Shame] = new() { "мне стыдно", "неловко", "позор", "я опозорился" },
            [E.Guilt] = new() { "я виноват", "прости", "извини", "мне совестно", "сожалею" },
            [E.Loneliness] = new() { "я один", "мне одиноко", "никого рядом", "пусто", "не с кем" }
        };
    }

    private static double NextGaussian(Random rnd, double mean, double stdDev)
    {
        double u1 = 1.0 - rnd.NextDouble();
        double u2 = 1.0 - rnd.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * z;
    }

    // =========================
    // 7) Модель
    // =========================
    private static Network BuildModel()
    {
        // Под текущее ядро (Linear + Activation + Softmax сериализуются)
        var net = new Network();
        net.Add(new LinearLayer(InputSize, 80, seed: 123));
        net.Add(new ActivationLayer(80, ActivationType.ReLu));
        net.Add(new LinearLayer(80, 48, seed: 124));
        net.Add(new ActivationLayer(48, ActivationType.ReLu));
        net.Add(new LinearLayer(48, Classes, seed: 125));
        net.Add(new SoftmaxLayer(Classes));
        return net;
    }

    // =========================
    // 8) Run
    // =========================
    public static void Run()
    {
        Console.WriteLine("=== TEST: Emotions RU (Softmax) ===");

        // 1) dataset
        var samples = GenerateSamples(count: 45000, seed: 10, noiseStd: 0.02);
        var dataset = samples.Select(s => (x: s.x, y: s.y)).ToArray();

        int valSize = 4000;
        var train = dataset.Take(dataset.Length - valSize).ToArray();
        var val = dataset.Skip(dataset.Length - valSize).ToArray();

        // 2) load/save по modelName (ядро)
        Network model;
        try
        {
            Console.WriteLine("Loading existing model...");
            model = Network.Load(ModelName);
            Console.WriteLine("Model loaded.\n");
        }
        catch
        {
            Console.WriteLine("No model found. Training from scratch...\n");
            model = BuildModel();
        }

        // 3) trainer
        var optimizer = new AdamOptimizer(learningRate: 0.0008);
        var loss = new CrossEntropyLoss();
        var trainer = new Trainer(model, optimizer, loss);

        // 4) callbacks (accuracy — снаружи, это Examples)
        var eval = val.Take(2000).ToArray();

        var callbacks = new ITrainCallback[]
        {
            new CallbackAdapter(new Callback(model, eval, every: 1)),
            // если захочешь стопать по loss — можно добавить:
            // new EarlyStoppingByLoss(patience: 6, minDelta: 1e-4, useValidationIfAvailable: true)
        };

        // 5) train options (канон)
        var options = new TrainOptions
        {
            Epochs = 25,
            BatchSize = 128,
            Shuffle = true,
            DropLast = false,
            GradClipNorm = 5.0,
            GradientAccumulationSteps = 1,
            Seed = 42,
            Validation = val,
            Callbacks = callbacks
        };

        var t0 = DateTime.Now;
        trainer.Train(train, options);
        var dt = DateTime.Now - t0;

        Console.WriteLine($"Training time: {dt.TotalSeconds:F1} sec");

        model.Save(ModelName);
        Console.WriteLine($"Model saved: ML/Models/{ModelName}.json\n");

        RunConsoleChat(model);
    }

    // =========================
    // 9) Console chat
    // =========================
    private static void RunConsoleChat(Network model)
    {
        Console.WriteLine("=== Console Emotion Chat (RU) ===");
        Console.WriteLine("Пиши фразы по-русски. Команды: /exit, /top, /help\n");

        bool showTop = true;
        int topK = 5;

        while (true)
        {
            Console.Write("> ");
            var line = Console.ReadLine();
            if (line == null) break;

            line = line.Trim();
            if (line.Length == 0) continue;

            var lower = line.ToLowerInvariant();

            if (lower is "/exit" or "exit" or "quit" or "выход" or "выйти")
                break;

            if (lower.StartsWith("/help", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Команды:");
                Console.WriteLine("  /exit            - выход");
                Console.WriteLine("  /top             - toggle top-k");
                Console.WriteLine("  /top 3|5|8|16    - установить k");
                Console.WriteLine();
                continue;
            }

            if (lower.StartsWith("/top", StringComparison.OrdinalIgnoreCase))
            {
                var parts = lower.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 2 && int.TryParse(parts[1], out var k) && k is >= 1 and <= 16)
                {
                    topK = k;
                    showTop = true;
                    Console.WriteLine($"Top-{topK}: ON\n");
                }
                else
                {
                    showTop = !showTop;
                    Console.WriteLine($"Top-{topK}: {(showTop ? "ON" : "OFF")}\n");
                }
                continue;
            }

            // Safety first
            if (IsSafetyCritical(line))
            {
                PrintSafetyResponse();
                continue;
            }

            var x = TextToFeatures(line);
            var probs = model.Forward(x, training: false);

            int pred = ArgMax(probs);
            Console.WriteLine($"Эмоция: {Names[pred]}");
            Console.WriteLine($"Ответ:  {Response((E)pred)}");

            if (showTop)
                PrintTop(probs, topK);

            Console.WriteLine();
        }
    }

    private static int ArgMax(double[] v)
    {
        int idx = 0;
        double max = v[0];
        for (int i = 1; i < v.Length; i++)
            if (v[i] > max) { max = v[i]; idx = i; }
        return idx;
    }

    private static void PrintTop(double[] probs, int k)
    {
        var pairs = probs.Select((p, i) => (p, i))
                         .OrderByDescending(t => t.p)
                         .Take(k);

        foreach (var (p, i) in pairs)
            Console.WriteLine($"{Names[i],14}: {p:0.000}");
    }

    private static string Response(E e) => e switch
    {
        E.Joy => "О, да! Это светлая энергия 🔥",
        E.Smile => "Тепло поймала 🙂",
        E.Laugh => "Ахаха, ну ты даёшь 😄",
        E.Gratitude => "Приняла. Спасибо — это сила 🤍",
        E.Pride => "Красиво сделал. Это опора 💪",
        E.Interest => "Хорошо. Давай копать глубже 👀",
        E.Calm => "Ровно. Так и держим.",
        E.Sadness => "Слышу. Без давления. Я рядом.",
        E.Suffering => "Тяжело. Давай маленькими шагами.",
        E.Fear => "Ок. Сначала безопасность. Что сейчас важнее всего?",
        E.Anger => "Поняла. Границы. Давай по сути, без разрушений.",
        E.Disgust => "Фу — честно. Уберём это подальше.",
        E.Shame => "Стыд часто врёт. Не уничтожай себя.",
        E.Guilt => "Вина — сигнал. Исправить можно.",
        E.Loneliness => "Одиночество не навсегда. Я здесь.",
        _ => "Ок."
    };
}
