using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using ML.Studio.Models;

namespace ML.Studio.Controls;

public sealed class NetworkGraphView : Control
{
    public static readonly StyledProperty<GraphModel?> GraphProperty =
        AvaloniaProperty.Register<NetworkGraphView, GraphModel?>(nameof(Graph));

    public static readonly StyledProperty<int?> SelectedNodeIdProperty =
        AvaloniaProperty.Register<NetworkGraphView, int?>(nameof(SelectedNodeId));

    private const double MinScale = 0.2;
    private const double MaxScale = 3.0;
    private double _scale = 1.0;
    private Vector _translate = new(20, 20);
    private bool _isPanning;
    private Point _lastPointer;

    public GraphModel? Graph
    {
        get => GetValue(GraphProperty);
        set => SetValue(GraphProperty, value);
    }

    public int? SelectedNodeId
    {
        get => GetValue(SelectedNodeIdProperty);
        set => SetValue(SelectedNodeIdProperty, value);
    }

    public NetworkGraphView()
    {
        ClipToBounds = true;
        GraphProperty.Changed.AddClassHandler<NetworkGraphView>((x, _) => x.InvalidateVisual());
        SelectedNodeIdProperty.Changed.AddClassHandler<NetworkGraphView>((x, _) => x.InvalidateVisual());
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);
        var graph = Graph;
        if (graph == null || graph.Nodes.Count == 0)
            return;

        var matrix = Matrix.CreateTranslation(_translate.X, _translate.Y) * Matrix.CreateScale(_scale, _scale);
        using (context.PushPostTransform(matrix))
        {
            DrawEdges(context, graph);
            DrawNodes(context, graph);
        }
    }

    private void DrawEdges(DrawingContext context, GraphModel graph)
    {
        var pen = new Pen(new SolidColorBrush(Color.Parse("#6B6B6B")), 1);
        foreach (var edge in graph.Edges)
        {
            var from = graph.Nodes.FirstOrDefault(n => n.Id == edge.FromId);
            var to = graph.Nodes.FirstOrDefault(n => n.Id == edge.ToId);
            if (from == null || to == null)
                continue;

            var start = new Point(from.X + from.Width, from.Y + from.Height / 2);
            var end = new Point(to.X, to.Y + to.Height / 2);
            context.DrawLine(pen, start, end);

            var arrowSize = 6;
            var arrowP1 = new Point(end.X - arrowSize, end.Y - arrowSize / 2);
            var arrowP2 = new Point(end.X - arrowSize, end.Y + arrowSize / 2);
            var geometry = new StreamGeometry();
            using (var geo = geometry.Open())
            {
                geo.BeginFigure(end, true);
                geo.LineTo(arrowP1);
                geo.LineTo(arrowP2);
                geo.EndFigure(true);
            }
            context.DrawGeometry(new SolidColorBrush(Color.Parse("#6B6B6B")), null, geometry);
        }
    }

    private void DrawNodes(DrawingContext context, GraphModel graph)
    {
        var titleBrush = new SolidColorBrush(Color.Parse("#E0E0E0"));
        var subtitleBrush = new SolidColorBrush(Color.Parse("#A0A0A0"));

        foreach (var node in graph.Nodes)
        {
            bool isSelected = SelectedNodeId == node.Id;
            var fill = new SolidColorBrush(isSelected ? Color.Parse("#2A3B55") : Color.Parse("#2B2B2B"));
            var border = new Pen(new SolidColorBrush(isSelected ? Color.Parse("#4AA3FF") : Color.Parse("#444444")), 2);
            var rect = new Rect(node.X, node.Y, node.Width, node.Height);
            context.DrawRectangle(fill, border, rect, 8, 8);

            var title = new FormattedText(
                node.Title,
                System.Globalization.CultureInfo.InvariantCulture,
                FlowDirection.LeftToRight,
                new Typeface("Segoe UI", FontStyle.Normal, FontWeight.SemiBold),
                13,
                titleBrush);
            var subtitle = new FormattedText(
                node.Subtitle,
                System.Globalization.CultureInfo.InvariantCulture,
                FlowDirection.LeftToRight,
                new Typeface("Segoe UI"),
                11,
                subtitleBrush);

            context.DrawText(title, new Point(node.X + 10, node.Y + 8));
            context.DrawText(subtitle, new Point(node.X + 10, node.Y + 30));
        }
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        var point = e.GetPosition(this);
        var hit = HitTestNode(point);
        if (hit != null)
        {
            SelectedNodeId = hit.Id;
            e.Handled = true;
            return;
        }

        SelectedNodeId = null;
        _isPanning = true;
        _lastPointer = point;
        e.Pointer.Capture(this);
        e.Handled = true;
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        if (!_isPanning)
            return;

        var pos = e.GetPosition(this);
        var delta = pos - _lastPointer;
        _translate = new Vector(_translate.X + delta.X, _translate.Y + delta.Y);
        _lastPointer = pos;
        InvalidateVisual();
        e.Handled = true;
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        _isPanning = false;
        e.Pointer.Capture(null);
    }

    protected override void OnPointerWheelChanged(PointerWheelEventArgs e)
    {
        base.OnPointerWheelChanged(e);
        var pos = e.GetPosition(this);
        double zoomDelta = e.Delta.Y > 0 ? 1.1 : 0.9;
        double newScale = Math.Clamp(_scale * zoomDelta, MinScale, MaxScale);
        if (Math.Abs(newScale - _scale) < 1e-6)
            return;

        var world = ScreenToWorld(pos);
        _scale = newScale;
        _translate = new Vector(
            pos.X - world.X * _scale,
            pos.Y - world.Y * _scale);
        InvalidateVisual();
        e.Handled = true;
    }

    public void FitToView(double padding = 20)
    {
        var graph = Graph;
        if (graph == null || graph.Nodes.Count == 0 || Bounds.Width <= 0 || Bounds.Height <= 0)
            return;

        var bounds = graph.Bounds;
        var availableW = Bounds.Width - padding * 2;
        var availableH = Bounds.Height - padding * 2;
        if (availableW <= 0 || availableH <= 0)
            return;

        var scaleX = availableW / Math.Max(1, bounds.Width);
        var scaleY = availableH / Math.Max(1, bounds.Height);
        _scale = Math.Clamp(Math.Min(scaleX, scaleY), MinScale, MaxScale);

        _translate = new Vector(
            padding - bounds.MinX * _scale,
            padding - bounds.MinY * _scale);

        InvalidateVisual();
    }

    public void ResetView()
    {
        _scale = 1.0;
        _translate = new Vector(20, 20);
        InvalidateVisual();
    }

    private GraphNode? HitTestNode(Point screenPoint)
    {
        var graph = Graph;
        if (graph == null)
            return null;

        var world = ScreenToWorld(screenPoint);
        foreach (var node in graph.Nodes)
        {
            var rect = new Rect(node.X, node.Y, node.Width, node.Height);
            if (rect.Contains(world))
                return node;
        }

        return null;
    }

    private Point ScreenToWorld(Point screen)
    {
        return new Point(
            (screen.X - _translate.X) / _scale,
            (screen.Y - _translate.Y) / _scale);
    }
}
