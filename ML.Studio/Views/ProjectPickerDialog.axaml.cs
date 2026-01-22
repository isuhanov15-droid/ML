using System;
using Avalonia.Controls;
using ML.Shared.Protocol;

namespace ML.Studio.Views;

public sealed partial class ProjectPickerDialog : Window
{
    public IReadOnlyList<ProjectDto> Projects { get; }
    public ProjectDto? SelectedProject { get; set; }

    public ProjectPickerDialog() : this(Array.Empty<ProjectDto>())
    {
    }

    public ProjectPickerDialog(IReadOnlyList<ProjectDto> projects)
    {
        InitializeComponent();
        Projects = projects;
        DataContext = this;
    }

    private void OnOk(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        Close(true);
    }

    private void OnCancel(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        Close(false);
    }
}
