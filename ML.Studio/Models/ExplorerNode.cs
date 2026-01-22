using System.Collections.ObjectModel;

namespace ML.Studio.Models;

public sealed class ExplorerNode
{
    public string Name { get; }
    public string? FullPath { get; }
    public ObservableCollection<ExplorerNode> Children { get; }

    public bool IsFile => FullPath != null;

    public ExplorerNode(string name, string? fullPath = null, IEnumerable<ExplorerNode>? children = null)
    {
        Name = name;
        FullPath = fullPath;
        Children = children != null ? new ObservableCollection<ExplorerNode>(children) : new ObservableCollection<ExplorerNode>();
    }
}
