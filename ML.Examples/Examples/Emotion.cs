using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using ML.Core;
using ML.Core.Abstractions;
using ML.Core.Layers;
using ML.Core.Losses;
using ML.Core.Optimizers;
using ML.Core.Training;
using ML.Core.Training.Callbacks;

namespace ML.Examples;

static class Emotion
{
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
    // 2) Фичи текста (≈52)
    // =========================
    // Структура:
    // [0..15]  - лексиконы эмоций/сигналов
    // [16..31] - форма/пунктуация/капс/длина/повторы/эмодзи
    // [32..51] - грамматические/контекстные признаки (не/вопросительные/я-ты/время/соц)
    public const int InputSize = 52;

    // --- токенизация ---
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
    // Это НЕ “эмоции”, это приоритет безопасности.
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

    private static void PrintSafetyResponse(string text)
    {
        Console.WriteLine("Эмоция: Страх");
        Console.WriteLine("Ответ:  Я не могу поддерживать угрозы или вред. Давай остановимся, выдохнем и переключимся на безопасный разговор.");
        Console.WriteLine();
    }

    // =========================
    // 4) Лексиконы (ядро смысла)
    // =========================
    // Небольшие, но “сильные”: расширять можно по мере необходимости.
    private static readonly HashSet<string> LxJoy = new(StringComparer.OrdinalIgnoreCase)
    { "ура","класс","кайф","рад","счастлив","счастлива","победа","вышло","получилось","удалось","круто","огонь" };

    private static readonly HashSet<string> LxSmile = new(StringComparer.OrdinalIgnoreCase)
    { "приятно","тепло","улыбаюсь","улыбка","милота","хорошо","уютно","лампово","светло" };

    private static readonly HashSet<string> LxLaugh = new(StringComparer.OrdinalIgnoreCase)
    { "ахаха","хаха","лол","ржу","смешно","прикол","шутка","угар","анекдот" };

    private static readonly HashSet<string> LxGratitude = new(StringComparer.OrdinalIgnoreCase)
    { "спасибо","благодарю","признателен","признательна","ценю","спасиб","благодарность","лучший","лучшая" };

    private static readonly HashSet<string> LxPride = new(StringComparer.OrdinalIgnoreCase)
    { "горжусь","горд","горда","достиг","достигла","смог","смогла","сделал","сделала","вынес","выдержал" };

    private static readonly HashSet<string> LxInterest = new(StringComparer.OrdinalIgnoreCase)
    { "интересно","любопытно","хочу","узнать","почему","как","что","разберемся","посмотрим","идея" };

    private static readonly HashSet<string> LxCalm = new(StringComparer.OrdinalIgnoreCase)
    { "спокойно","тихо","ровно","стабильно","уверенно","норм","нормально","ок","выдох","пауза" };

    private static readonly HashSet<string> LxSadness = new(StringComparer.OrdinalIgnoreCase)
    { "грустно","печально","тоска","слезы","плачу","пусто","жалко","скучаю","уныло" };

    private static readonly HashSet<string> LxSuffering = new(StringComparer.OrdinalIgnoreCase)
    { "больно","страдаю","тяжело","невыносимо","плохо","разбит","выжат","выгорание","кошмарно","нетсил" };

    private static readonly HashSet<string> LxFear = new(StringComparer.OrdinalIgnoreCase)
    { "страшно","опасно","ужас","паника","пугает","угроза","кошмар","боюсь","жутко","обстрел","взрыв" };

    private static readonly HashSet<string> LxAnger = new(StringComparer.OrdinalIgnoreCase)
    { "злюсь","бесит","раздражает","достало","ярость","взбесило","ненавижу","сука","идиот","тварь" };

    private static readonly HashSet<string> LxDisgust = new(StringComparer.OrdinalIgnoreCase)
    { "фу","противно","мерзко","отвратительно","тошно","воняет","грязь","гадость","пакость" };

    private static readonly HashSet<string> LxShame = new(StringComparer.OrdinalIgnoreCase)
    { "стыдно","стыд","позор","неловко","опозорился","опозорилась","смущаюсь","смущение" };

    private static readonly HashSet<string> LxGuilt = new(StringComparer.OrdinalIgnoreCase)
    { "виноват","виновата","вина","прости","извини","простите","сожалею","жальчто","неправ" };

    private static readonly HashSet<string> LxLoneliness = new(StringComparer.OrdinalIgnoreCase)
    { "один","одна","одинок","одиноко","никого","пусто","нетникого","втроемне","не с кем","без тебя" };

    // общие сигналы
    private static readonly HashSet<string> LxNegation = new(StringComparer.OrdinalIgnoreCase)
    { "не","нет","никогда","ни","нифига","ничего","никак" };

    private static readonly HashSet<string> LxQuestion = new(StringComparer.OrdinalIgnoreCase)
    { "как","почему","зачем","что","когда","где","кто","сколько","ли" };

    private static readonly HashSet<string> LxFirstPerson = new(StringComparer.OrdinalIgnoreCase)
    { "я","мне","меня","мой","моя","мои","со мной","мною" };

    private static readonly HashSet<string> LxSecondPerson = new(StringComparer.OrdinalIgnoreCase)
    { "ты","тебе","тебя","твой","твоя","твои","вы","вам","вас" };

    private static readonly HashSet<string> LxSupport = new(StringComparer.OrdinalIgnoreCase)
    { "рядом","с тобой","обнимаю","держись","помогу","вместе","поддержу","семья","друг","друзья" };

    private static int CountLex(IEnumerable<string> tokens, HashSet<string> lex)
    {
        int c = 0;
        foreach (var t in tokens)
            if (lex.Contains(t)) c++;
        return c;
    }

    // =========================
    // 5) Улучшенный экстрактор фич
    // =========================
    public static double[] TextToFeatures(string text)
    {
        text ??= "";
        var tokens = Tokenize(text);

        // Лексиконы (0..15)
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

        // scale: 3 сигнальных слова ≈ 1.0
        const double scale = 3.0;

        // Форма/пунктуация/капс/длина/повторы/эмодзи (16..31)
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

        int repeats = CountCharRepeats(text);      // "ааа", "!!!", "))))"
        int smiles = CountSmiles(text);            // :) :-) :D ))))
        int emojis = CountEmojiLike(text);         // грубый подсчёт эмодзи-символов

        // Контекст (32..51)
        int neg = CountLex(tokens, LxNegation);
        int qwords = CountLex(tokens, LxQuestion);
        int fp = CountLex(tokens, LxFirstPerson);
        int sp = CountLex(tokens, LxSecondPerson);
        int support = CountLex(tokens, LxSupport);

        // простые временные/модальные маркеры
        int past = CountAny(tokens, "вчера", "было", "была", "был", "потерял", "потеряла", "сделал", "сделала", "успел", "успела");
        int future = CountAny(tokens, "завтра", "будет", "буду", "сделаю", "сделаем", "хочу", "план");
        int now = CountAny(tokens, "сейчас", "сегодня", "вот", "прям", "именно");

        // “интенсивность” (мат/усилители) — грубо
        int intens = CountAny(tokens, "очень", "капец", "сильно", "реально", "жесть", "просто", "пипец");

        var x = new double[InputSize];

        // 0..15
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

        // 15 — базовый “позитив/негатив баланс”
        double pos = x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6];
        double negv = x[7] + x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14];
        x[15] = Clamp01(0.5 + 0.25 * (pos - negv)); // центр 0.5

        // 16..31 (форма)
        x[16] = Clamp01(len / 140.0);
        x[17] = Clamp01(words / 24.0);
        x[18] = Clamp01(exclam / 6.0);
        x[19] = Clamp01(quest / 6.0);
        x[20] = Clamp01(dots / 3.0);
        x[21] = Clamp01(comma / 6.0);
        x[22] = Clamp01(quotes / 4.0);
        x[23] = Clamp01(capsRatio * 1.5);
        x[24] = Clamp01(repeats / 6.0);
        x[25] = Clamp01(smiles / 6.0);
        x[26] = Clamp01(emojis / 4.0);

        // 27..31 — “сглаженные агрегаты”
        x[27] = Clamp01((x[18] + x[23] + x[24]) / 3.0); // возбуждение/накал
        x[28] = Clamp01((x[19] + Clamp01(qwords / 4.0)) / 2.0); // вопросительность
        x[29] = Clamp01((x[25] + x[26] + x[1]) / 3.0); // дружелюбность/соц-сигнал
        x[30] = Clamp01((x[8] + x[9] + x[14]) / 3.0); // “тяжёлость”
        x[31] = Clamp01((x[0] + x[4] + x[6]) / 3.0); // “уверенный позитив”

        // 32..51 (контекст)
        x[32] = Clamp01(neg / 3.0);
        x[33] = Clamp01(qwords / 4.0);
        x[34] = Clamp01(fp / 4.0);
        x[35] = Clamp01(sp / 4.0);
        x[36] = Clamp01(support / 3.0);

        x[37] = Clamp01(past / 3.0);
        x[38] = Clamp01(future / 3.0);
        x[39] = Clamp01(now / 3.0);
        x[40] = Clamp01(intens / 4.0);

        // Отрицание влияет на позитивные лексиконы (легкая эвристика)
        // "не смешно", "не рад" — уменьшаем смех/радость
        double negFactor = 1.0 - 0.5 * x[32];
        x[41] = Clamp01(x[0] * negFactor);
        x[42] = Clamp01(x[2] * negFactor);
        x[43] = Clamp01(x[1] * negFactor);

        // Конфликт “ты” + злость
        x[44] = Clamp01(x[35] * x[10] * 2.0);

        // Конфликт “я” + вина/стыд
        x[45] = Clamp01(x[34] * (x[13] + x[12]) * 1.2);

        // Одиночество без поддержки
        x[46] = Clamp01(x[14] * (1.0 - x[36]));

        // Спокойствие + поддержка
        x[47] = Clamp01(x[6] * (0.6 + 0.6 * x[36]));

        // Интерес + вопросительность
        x[48] = Clamp01((x[5] + x[28]) / 2.0);

        // “Накал” + злость/страх
        x[49] = Clamp01(x[27] * (x[10] + x[9]) * 0.9);

        // “Тяжёлость” + страдание/грусть
        x[50] = Clamp01(x[30] * (x[8] + x[7]) * 0.9);

        // Резервный “сигнал неопределенности”: мало слов, мало лексиконов
        double lexSum = pos + negv;
        x[51] = Clamp01((1.0 - Clamp01(words / 6.0)) * (1.0 - Clamp01(lexSum)));

        return x;
    }

    private static int CountSubstring(string s, string sub)
    {
        if (string.IsNullOrEmpty(s) || string.IsNullOrEmpty(sub)) return 0;
        int count = 0;
        int idx = 0;
        while ((idx = s.IndexOf(sub, idx, StringComparison.Ordinal)) >= 0)
        {
            count++;
            idx += sub.Length;
        }
        return count;
    }

    private static int CountAny(string[] tokens, params string[] words)
    {
        int c = 0;
        foreach (var t in tokens)
            for (int i = 0; i < words.Length; i++)
                if (string.Equals(t, words[i], StringComparison.OrdinalIgnoreCase)) { c++; break; }
        return c;
    }

    private static int CountCharRepeats(string s)
    {
        if (string.IsNullOrEmpty(s)) return 0;
        int repeats = 0;
        int run = 1;
        for (int i = 1; i < s.Length; i++)
        {
            if (s[i] == s[i - 1]) run++;
            else
            {
                if (run >= 3) repeats++;
                run = 1;
            }
        }
        if (run >= 3) repeats++;
        return repeats;
    }

    private static int CountSmiles(string s)
    {
        if (string.IsNullOrEmpty(s)) return 0;
        int c = 0;
        c += CountSubstring(s, ":)");
        c += CountSubstring(s, ":-)");
        c += CountSubstring(s, ":D");
        c += CountSubstring(s, ":-D");
        c += CountSubstring(s, ")))");
        c += CountSubstring(s, "(((");
        return c;
    }

    // грубый счёт эмодзи: Unicode диапазоны + суррогаты
    private static int CountEmojiLike(string s)
    {
        if (string.IsNullOrEmpty(s)) return 0;
        int c = 0;
        foreach (var ch in s)
        {
            // очень приблизительно, но достаточно как “сигнал”
            if (ch >= 0x2600 && ch <= 0x27BF) c++;      // dingbats etc
            if (ch >= 0x1F300) c++;                     // может не сработать на char (surrogate), ок
        }
        // дополнительно: если есть суррогаты — вероятно эмодзи
        for (int i = 0; i < s.Length; i++)
            if (char.IsSurrogate(s[i])) { c++; break; }

        return c > 3 ? 3 : c; // ограничим
    }

    // =========================
    // 6) Датасет фраз (RU)
    // =========================
    private static readonly Dictionary<E, string[]> PhrasePool = new()
    {
        [E.Neutral] = new[]
        {
            "привет", "нормально", "в целом ок", "обычный день", "без особых эмоций", "ровно", "как обычно",
            "что нового", "как дела", "пока не знаю", "посмотрим"
        },

        [E.Joy] = new[]
        {
            "ура получилось", "я счастлив", "как же круто", "кайф", "вот это победа", "я рад", "вышло отлично",
            "супер новость", "это реально класс", "всё получилось!"
        },

        [E.Smile] = new[]
        {
            "улыбаюсь", "приятно", "тепло на душе", "мне уютно", "так мило", "хорошо стало", "лампово",
            "тихий кайф", "спокойная радость"
        },

        [E.Laugh] = new[]
        {
            "ахаха", "смешно", "лол", "я ржу", "угар", "шутка огонь", "прикол", "хаха да",
            "это так смешно"
        },

        [E.Gratitude] = new[]
        {
            "спасибо", "спасибо тебе", "я благодарю", "очень ценю", "признателен", "ты лучшая", "спасибо большое",
            "спасиб, выручил", "благодарность огромная"
        },

        [E.Pride] = new[]
        {
            "горжусь собой", "я смог", "я выдержал", "я сделал это", "достиг цели", "не сдался", "справился",
            "закрыл задачу", "я молодец"
        },

        [E.Interest] = new[]
        {
            "интересно", "любопытно", "как это работает", "хочу понять", "давай разберемся", "почему так",
            "есть идея", "можно попробовать", "а если так сделать"
        },

        [E.Calm] = new[]
        {
            "спокойно", "ровно", "всё под контролем", "я выдохнул", "пауза", "стабильно", "без паники",
            "держим курс", "тихо и ясно"
        },

        [E.Sadness] = new[]
        {
            "грустно", "печально", "тоска", "пусто внутри", "слезы", "мне жаль", "сердце тяжелеет",
            "не по себе", "скучаю"
        },

        [E.Suffering] = new[]
        {
            "очень тяжело", "мне больно", "невыносимо", "я выжат", "разбит", "нет сил", "выгорание",
            "плохо", "не могу"
        },

        [E.Fear] = new[]
        {
            "мне страшно", "опасно", "паника", "пугает", "кошмар", "я боюсь", "угроза", "жутко",
            "это тревожит"
        },

        [E.Anger] = new[]
        {
            "я злюсь", "меня бесит", "раздражает", "достало", "ярость", "взбесило", "ненавижу это",
            "как же бесит"
        },

        [E.Disgust] = new[]
        {
            "фу противно", "мерзко", "отвратительно", "тошно", "гадость", "воняет", "противно смотреть",
            "пакость"
        },

        [E.Shame] = new[]
        {
            "мне стыдно", "стыд", "неловко", "опозорился", "опозорилась", "как я мог", "позор",
            "смущаюсь"
        },

        [E.Guilt] = new[]
        {
            "я виноват", "виновата", "прости", "извини", "моя вина", "сожалею", "неправ",
            "жаль что так вышло"
        },

        [E.Loneliness] = new[]
        {
            "мне одиноко", "я один", "я одна", "нет никого", "пусто", "не с кем поговорить",
            "без тебя пусто", "никто не пишет"
        },
    };

    private static readonly Dictionary<E, string[]> ReactionPool = new()
    {
        [E.Neutral]    = new[] { "Окей 🙂", "Принято.", "Поняла. Держим ровно." },
        [E.Joy]        = new[] { "Ооо да! 🔥", "Вот это победа 😄", "Кайф! Так держать." },
        [E.Smile]      = new[] { "Тепло 😊", "Улыбка — хороший знак.", "Лампово, понимаю." },
        [E.Laugh]      = new[] { "Ахаха 😂", "Окей, смешно 😄", "Хороший прикол." },
        [E.Gratitude]  = new[] { "Принято 🙏", "Спасибо услышала.", "Тёплая благодарность — сила." },
        [E.Pride]      = new[] { "Вот это характер 💪", "Горжусь твоим усилием.", "Сильный ход." },
        [E.Interest]   = new[] { "О, это интересно 👀", "Давай копнём глубже.", "Хороший вопрос." },
        [E.Calm]       = new[] { "Выдох — да ✅", "Спокойный режим.", "Держим контроль." },
        [E.Sadness]    = new[] { "Я рядом. Тихо.", "Понимаю… это грустно.", "Без давления. Я с тобой." },
        [E.Suffering]  = new[] { "Окей… это тяжело.", "Давай стабилизируемся шаг за шагом.", "Я рядом, дышим." },
        [E.Fear]       = new[] { "Сначала безопасность.", "Выдох. Мы справимся.", "Поняла. Снизим шум." },
        [E.Anger]      = new[] { "Окей. Границы.", "Сделаем паузу и решим.", "Злость — энергия. Направим." },
        [E.Disgust]    = new[] { "Фу, да.", "Отдаляемся от источника.", "Поняла. Убираем это." },
        [E.Shame]      = new[] { "Стыд — сигнал, не приговор.", "Мягче к себе.", "Давай без самобичевания." },
        [E.Guilt]      = new[] { "Принято. Можно исправить.", "Давай разложим, что сделать дальше.", "Окей. Шаг за шагом." },
        [E.Loneliness] = new[] { "Я рядом. Слышишь?", "Давай поговорим. Ты не один.", "Поняла. Держу рядом." },
    };

    private static string PickReaction(E e)
    {
        if (!ReactionPool.TryGetValue(e, out var arr) || arr.Length == 0)
            return "Поняла.";
        return arr[(int)(DateTime.Now.Ticks % arr.Length)];
    }

    // Генерация обучающих примеров: только внутри своего класса + лёгкие усилители
    public static (double[] x, int y)[] GenerateSamples(int count, int seed = 42, double noiseStd = 0.02)
    {
        var rnd = new Random(seed);
        var kinds = PhrasePool.Keys.ToArray();

        var data = new (double[] x, int y)[count];

        for (int i = 0; i < count; i++)
        {
            var k = kinds[rnd.Next(kinds.Length)];
            var phrase = PhrasePool[k][rnd.Next(PhrasePool[k].Length)];

            // лёгкая аугментация (НЕ противоречит метке)
            phrase = Augment(phrase, k, rnd);

            var x = TextToFeatures(phrase);

            // небольшой гауссов шум (делает устойчивее)
            for (int j = 0; j < x.Length; j++)
                x[j] = Clamp01(x[j] + NextGaussian(rnd, 0, noiseStd));

            data[i] = (x, (int)k);
        }

        return data;
    }

    private static string Augment(string phrase, E k, Random rnd)
    {
        if (rnd.NextDouble() < 0.25)
            phrase = AddIntensifier(phrase, rnd);

        if (rnd.NextDouble() < 0.15)
            phrase = AddPunctuation(phrase, k, rnd);

        if (rnd.NextDouble() < 0.10)
            phrase = AddEmoji(phrase, k, rnd);

        // “не” — добавляем осторожно и только где уместно (не ломаем метку)
        if (rnd.NextDouble() < 0.08 && (k == E.Calm || k == E.Neutral || k == E.Sadness))
            phrase = "не знаю... " + phrase;

        return phrase;
    }

    private static string AddIntensifier(string s, Random rnd)
    {
        var a = new[] { "очень", "реально", "сильно", "прям", "капец", "жесть" };
        return $"{a[rnd.Next(a.Length)]} {s}";
    }

    private static string AddPunctuation(string s, E k, Random rnd)
    {
        return k switch
        {
            E.Joy or E.Anger => s + new string('!', 1 + rnd.Next(3)),
            E.Interest => s + "?",
            E.Sadness or E.Suffering => s + "...",
            _ => s
        };
    }

    private static string AddEmoji(string s, E k, Random rnd)
    {
        return k switch
        {
            E.Joy or E.Smile => s + " 😊",
            E.Laugh => s + " 😂",
            E.Sadness => s + " 😔",
            E.Anger => s + " 😡",
            E.Fear => s + " 😨",
            _ => s
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
        var net = new Network();
        net.Add(new LinearLayer(InputSize, 80));
        net.Add(new ActivationLayer(80, ActivationType.ReLu));
        net.Add(new LinearLayer(80, 48));
        net.Add(new ActivationLayer(48, ActivationType.ReLu));
        net.Add(new LinearLayer(48, Classes));
        net.Add(new SoftmaxLayer(Classes));
        return net;
    }

    // =========================
    // 8) Run
    // =========================
    public static void Run()
    {
        Console.WriteLine("=== TEST: Emotions RU MAX (Softmax) ===");

        // 1) dataset
        var samples = GenerateSamples(count: 45000, seed: 10, noiseStd: 0.02);
        var dataset = samples.Select(s => (x: s.x, y: s.y)).ToArray();

        int valSize = 4000;
        var train = dataset.Take(dataset.Length - valSize).ToArray();
        var val = dataset.Skip(dataset.Length - valSize).ToArray();

        // 2) path
        var modelPath = GetModelPath();
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);

        Network model;

        if (File.Exists(modelPath))
        {
            Console.WriteLine("Loading existing model...");
            model = Network.Load(modelPath);
            Console.WriteLine("Model loaded.\n");
        }
        else
        {
            model = BuildModel();
            Console.WriteLine("No model found. Training from scratch...\n");
        }

        // 3) trainer
        var optimizer = new AdamOptimizer(learningRate: 0.0008);
        ILoss loss = new CrossEntropyLoss();
        var trainer = new Trainer(model, optimizer, loss);

        // 4) callbacks
        // Старый Callback печатает accuracy (это Examples и это нормально).
        // Но Trainer ждёт ITrainCallback, поэтому используем Adapter.
        var eval = val.Take(2000).ToArray();

        var callbacks = new ITrainCallback[]
        {
            new CallbackAdapter(new Callback(model, eval, every: 1))
        };

        var t0 = DateTime.Now;

        trainer.Train(dataset: dataset, TrainOptions() );

        var dt = DateTime.Now - t0;
        Console.WriteLine($"Training time: {dt.TotalSeconds:F1} sec");

        model.Save(modelPath);
        Console.WriteLine($"Model saved: {modelPath}\n");

        // 5) interactive
        RunConsoleChat(model);
    }

    private static string GetModelPath()
    {
        // 1) если у вас есть Core.Path.ModelPath.Emotion — используем его
        // 2) иначе пишем в ML.Models/emotion_ru_max относительно рабочей папки
        try
        {
            // если в проекте реально есть такой путь — отлично
            return Core.Path.ModelPath.Emotion;
        }
        catch
        {
            var root = Directory.GetCurrentDirectory();
            return System.IO.Path.Combine(root, "ML.Models", "emotion_ru_max");
        }
    }

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
                Console.WriteLine("  /top 3|5|8       - установить k");
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
                PrintSafetyResponse(line);
                continue;
            }

            var x = TextToFeatures(line);
            var probs = model.Forward(x, training: false);

            int pred = ArgMax(probs);
            var e = (E)pred;

            Console.WriteLine($"Эмоция: {Names[pred]}");
            Console.WriteLine($"Ответ:  {PickReaction(e)}");

            if (showTop)
                PrintTopK(probs, topK);

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

    private static void PrintTopK(double[] probs, int k)
    {
        var top = probs.Select((p, i) => (p, i))
                       .OrderByDescending(t => t.p)
                       .Take(k);

        foreach (var (p, i) in top)
            Console.WriteLine($"{Names[i],14}: {p:F3}");
    }
}
