using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace ML.Studio.Models;

public sealed class EditorTab : INotifyPropertyChanged
{
    private string _title;
    private string _fullPath;
    private string _text;
    private bool _isDirty;
    private bool _suppressDirty;

    public string Title
    {
        get => _title;
        set
        {
            if (SetField(ref _title, value))
                OnPropertyChanged(nameof(DisplayTitle));
        }
    }

    public string FullPath
    {
        get => _fullPath;
        set => SetField(ref _fullPath, value);
    }

    public string Text
    {
        get => _text;
        set
        {
            if (!SetField(ref _text, value))
                return;

            if (!_suppressDirty)
                IsDirty = true;
        }
    }

    public bool IsDirty
    {
        get => _isDirty;
        private set
        {
            if (SetField(ref _isDirty, value))
                OnPropertyChanged(nameof(DisplayTitle));
        }
    }

    public string DisplayTitle => IsDirty ? $"{Title}*" : Title;

    public EditorTab(string title, string fullPath, string text)
    {
        _title = title;
        _fullPath = fullPath;
        _suppressDirty = true;
        _text = text;
        _suppressDirty = false;
        _isDirty = false;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public void MarkClean()
    {
        IsDirty = false;
    }

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
            return false;
        field = value;
        OnPropertyChanged(name);
        return true;
    }

    private void OnPropertyChanged(string? name)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
