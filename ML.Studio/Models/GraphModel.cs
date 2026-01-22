namespace ML.Studio.Models;

public sealed record GraphNode(
    int Id,
    string Title,
    string Subtitle,
    Dictionary<string, string> Meta,
    double X,
    double Y,
    double Width,
    double Height);

public sealed record GraphEdge(int FromId, int ToId);

public sealed class GraphModel
{
    public IReadOnlyList<GraphNode> Nodes { get; }
    public IReadOnlyList<GraphEdge> Edges { get; }
    public GraphBounds Bounds { get; }

    public static GraphModel Empty { get; } = new(Array.Empty<GraphNode>(), Array.Empty<GraphEdge>());

    public GraphModel(IReadOnlyList<GraphNode> nodes, IReadOnlyList<GraphEdge> edges)
    {
        Nodes = nodes;
        Edges = edges;
        Bounds = GraphBounds.FromNodes(nodes);
    }
}

public sealed record GraphBounds(double MinX, double MinY, double MaxX, double MaxY)
{
    public double Width => MaxX - MinX;
    public double Height => MaxY - MinY;

    public static GraphBounds FromNodes(IReadOnlyList<GraphNode> nodes)
    {
        if (nodes.Count == 0)
            return new GraphBounds(0, 0, 0, 0);

        double minX = nodes.Min(n => n.X);
        double minY = nodes.Min(n => n.Y);
        double maxX = nodes.Max(n => n.X + n.Width);
        double maxY = nodes.Max(n => n.Y + n.Height);
        return new GraphBounds(minX, minY, maxX, maxY);
    }
}
