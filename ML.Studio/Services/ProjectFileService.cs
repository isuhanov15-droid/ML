using System.Diagnostics;
using ML.Studio.Models;

namespace ML.Studio.Services;

public sealed class ProjectFileService
{
    private readonly WorkspaceService _workspace;

    public ProjectFileService(WorkspaceService workspace)
    {
        _workspace = workspace;
    }

    public async Task<ProjectFileItem> CreateFileAsync(StudioProject project, string relativePath, string templateContent, string kind)
    {
        var full = Path.Combine(project.RootPath, relativePath);
        var dir = Path.GetDirectoryName(full);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        if (!File.Exists(full))
            await File.WriteAllTextAsync(full, templateContent);

        var item = WorkspaceService.ToItem(project.RootPath, relativePath, kind);
        var files = project.Files.Concat(new[] { item }).ToArray();
        var updated = project with { Files = files, UpdatedUtc = DateTime.UtcNow };
        await _workspace.SaveProjectAsync(updated);
        return item;
    }

    public async Task RenameFileAsync(StudioProject project, string oldRelPath, string newRelPath)
    {
        var oldFull = Path.Combine(project.RootPath, oldRelPath);
        var newFull = Path.Combine(project.RootPath, newRelPath);
        var dir = Path.GetDirectoryName(newFull);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        File.Move(oldFull, newFull, true);

        var files = project.Files.Select(f =>
            f.RelativePath == oldRelPath ? f with { RelativePath = newRelPath, FullPath = newFull, DisplayName = Path.GetFileName(newRelPath) } : f).ToArray();
        var updated = project with { Files = files, UpdatedUtc = DateTime.UtcNow };
        await _workspace.SaveProjectAsync(updated);
    }

    public async Task DeleteFileAsync(StudioProject project, string relPath)
    {
        var full = Path.Combine(project.RootPath, relPath);
        if (File.Exists(full))
            File.Delete(full);

        var files = project.Files.Where(f => f.RelativePath != relPath).ToArray();
        var updated = project with { Files = files, UpdatedUtc = DateTime.UtcNow };
        await _workspace.SaveProjectAsync(updated);
    }

    public async Task<string> ReadTextAsync(string fullPath)
    {
        return await File.ReadAllTextAsync(fullPath);
    }

    public async Task WriteTextAsync(string fullPath, string text)
    {
        await File.WriteAllTextAsync(fullPath, text);
    }

    public void RevealInFileManager(string fullPath)
    {
        try
        {
            var dir = File.Exists(fullPath) ? Path.GetDirectoryName(fullPath) : fullPath;
            if (string.IsNullOrWhiteSpace(dir))
                return;

            if (OperatingSystem.IsWindows())
            {
                Process.Start(new ProcessStartInfo("explorer.exe", $"\"{dir}\"") { UseShellExecute = true });
            }
            else if (OperatingSystem.IsMacOS())
            {
                Process.Start("open", $"\"{dir}\"");
            }
            else
            {
                Process.Start("xdg-open", $"\"{dir}\"");
            }
        }
        catch
        {
        }
    }
}
