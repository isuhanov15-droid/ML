using Avalonia;
using Avalonia.Markup.Xaml;
using Avalonia.Markup.Xaml.Styling;
using Avalonia.Styling;

namespace ML.Studio.Services;

public static class ThemeService
{
    private static readonly Dictionary<string, Uri> Themes = new()
    {
        { "vs-dark", new Uri("avares://ML.Studio/Styles/Theme.VsDark.axaml") },
        { "graphite", new Uri("avares://ML.Studio/Styles/Theme.Graphite.axaml") },
        { "midnight", new Uri("avares://ML.Studio/Styles/Theme.Midnight.axaml") }
    };

    public static void Apply(string name)
    {
        if (Application.Current == null)
            return;

        if (!Themes.TryGetValue(name, out var uri))
            return;

        var resources = Application.Current.Resources;
        var merged = resources.MergedDictionaries;
        var existing = merged.OfType<ResourceInclude>().FirstOrDefault(x => x.Source?.ToString()?.Contains("Theme.") == true);
        if (existing != null)
            merged.Remove(existing);

        merged.Add(new ResourceInclude(uri) { Source = uri });
        Application.Current.RequestedThemeVariant = ThemeVariant.Dark;
    }
}
