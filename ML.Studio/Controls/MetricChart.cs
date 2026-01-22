using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media;
using Avalonia.Input;
using ML.Studio.Models;

namespace ML.Studio.Controls;

public sealed class MetricChart : Control
{
    public static readonly StyledProperty<IList<MetricPoint>?> ItemsProperty =
        AvaloniaProperty.Register<MetricChart, IList<MetricPoint>?>(nameof(Items));
    public static readonly StyledProperty<bool> ShowTrainLossProperty =
        AvaloniaProperty.Register<MetricChart, bool>(nameof(ShowTrainLoss), true);
    public static readonly StyledProperty<bool> ShowValLossProperty =
        AvaloniaProperty.Register<MetricChart, bool>(nameof(ShowValLoss), true);
    public static readonly StyledProperty<bool> ShowAccuracyProperty =
        AvaloniaProperty.Register<MetricChart, bool>(nameof(ShowAccuracy), true);
    public static readonly StyledProperty<bool> CompareModeProperty =
        AvaloniaProperty.Register<MetricChart, bool>(nameof(CompareMode), false);
    public static readonly StyledProperty<IList<MetricPoint>?> CompareItemsAProperty =
        AvaloniaProperty.Register<MetricChart, IList<MetricPoint>?>(nameof(CompareItemsA));
    public static readonly StyledProperty<IList<MetricPoint>?> CompareItemsBProperty =
        AvaloniaProperty.Register<MetricChart, IList<MetricPoint>?>(nameof(CompareItemsB));
    public static readonly StyledProperty<bool> CompareShowTrainLossProperty =
        AvaloniaProperty.Register<MetricChart, bool>(nameof(CompareShowTrainLoss), true);
    public static readonly StyledProperty<bool> CompareShowValLossProperty =
        AvaloniaProperty.Register<MetricChart, bool>(nameof(CompareShowValLoss), true);
    public static readonly StyledProperty<bool> CompareShowAccuracyProperty =
        AvaloniaProperty.Register<MetricChart, bool>(nameof(CompareShowAccuracy), true);

    private static readonly Pen TrainPen = new(new SolidColorBrush(Color.Parse("#5FB3FF")), 2);
    private static readonly Pen ValPen = new(new SolidColorBrush(Color.Parse("#FFB84D")), 2);
    private static readonly Pen AccPen = new(new SolidColorBrush(Color.Parse("#7DD36F")), 2);
    private static readonly Pen TrainDashPen = new(new SolidColorBrush(Color.Parse("#5FB3FF")), 2, new DashStyle(new[] { 6.0, 4.0 }, 0));
    private static readonly Pen ValDashPen = new(new SolidColorBrush(Color.Parse("#FFB84D")), 2, new DashStyle(new[] { 6.0, 4.0 }, 0));
    private static readonly Pen AccDashPen = new(new SolidColorBrush(Color.Parse("#7DD36F")), 2, new DashStyle(new[] { 6.0, 4.0 }, 0));
    private static readonly Pen HoverPen = new(new SolidColorBrush(Color.Parse("#8A8A8A")), 1);
    private static readonly IBrush HoverBrush = new SolidColorBrush(Color.Parse("#E0E0E0"));

    private IList<MetricPoint>? _items;
    private IList<MetricPoint>? _compareA;
    private IList<MetricPoint>? _compareB;
    private int? _hoverIndex;

    public IList<MetricPoint>? Items
    {
        get => GetValue(ItemsProperty);
        set => SetValue(ItemsProperty, value);
    }

    public bool ShowTrainLoss
    {
        get => GetValue(ShowTrainLossProperty);
        set => SetValue(ShowTrainLossProperty, value);
    }

    public bool ShowValLoss
    {
        get => GetValue(ShowValLossProperty);
        set => SetValue(ShowValLossProperty, value);
    }

    public bool ShowAccuracy
    {
        get => GetValue(ShowAccuracyProperty);
        set => SetValue(ShowAccuracyProperty, value);
    }

    public bool CompareMode
    {
        get => GetValue(CompareModeProperty);
        set => SetValue(CompareModeProperty, value);
    }

    public IList<MetricPoint>? CompareItemsA
    {
        get => GetValue(CompareItemsAProperty);
        set => SetValue(CompareItemsAProperty, value);
    }

    public IList<MetricPoint>? CompareItemsB
    {
        get => GetValue(CompareItemsBProperty);
        set => SetValue(CompareItemsBProperty, value);
    }

    public bool CompareShowTrainLoss
    {
        get => GetValue(CompareShowTrainLossProperty);
        set => SetValue(CompareShowTrainLossProperty, value);
    }

    public bool CompareShowValLoss
    {
        get => GetValue(CompareShowValLossProperty);
        set => SetValue(CompareShowValLossProperty, value);
    }

    public bool CompareShowAccuracy
    {
        get => GetValue(CompareShowAccuracyProperty);
        set => SetValue(CompareShowAccuracyProperty, value);
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);

        if (CompareMode)
        {
            RenderCompare(context);
            return;
        }

        if (_items == null || _items.Count < 2)
            return;

        var bounds = new Rect(0, 0, Bounds.Width, Bounds.Height);
        var plot = bounds.Deflate(8);
        if (plot.Width <= 0 || plot.Height <= 0)
            return;

        int minEpoch = _items.Min(m => m.Epoch);
        int maxEpoch = _items.Max(m => m.Epoch);
        if (maxEpoch == minEpoch)
            maxEpoch = minEpoch + 1;

        var values = new List<double>();
        foreach (var m in _items)
        {
            if (ShowTrainLoss)
                values.Add(m.Loss);
            if (ShowValLoss && m.ValLoss.HasValue)
                values.Add(m.ValLoss.Value);
            if (ShowAccuracy && m.Accuracy.HasValue)
                values.Add(m.Accuracy.Value);
        }

        if (values.Count == 0)
            return;

        double minY = values.Min();
        double maxY = values.Max();
        if (Math.Abs(maxY - minY) < 1e-9)
            maxY = minY + 1e-3;

        if (ShowTrainLoss)
            DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, TrainPen, m => m.Loss);
        if (ShowValLoss)
            DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, ValPen, m => m.ValLoss);
        if (ShowAccuracy)
            DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, AccPen, m => m.Accuracy);

        DrawHover(context, plot, minEpoch, maxEpoch, minY, maxY);
        DrawAxisLabels(context, plot);
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == ItemsProperty)
        {
            if (_items is INotifyCollectionChanged oldObs)
                oldObs.CollectionChanged -= OnCollectionChanged;
            _items = change.NewValue as IList<MetricPoint>;
            if (_items is INotifyCollectionChanged newObs)
                newObs.CollectionChanged += OnCollectionChanged;
            InvalidateVisual();
        }
        if (change.Property == CompareItemsAProperty)
        {
            if (_compareA is INotifyCollectionChanged oldObs)
                oldObs.CollectionChanged -= OnCollectionChanged;
            _compareA = change.NewValue as IList<MetricPoint>;
            if (_compareA is INotifyCollectionChanged newObs)
                newObs.CollectionChanged += OnCollectionChanged;
            InvalidateVisual();
        }
        if (change.Property == CompareItemsBProperty)
        {
            if (_compareB is INotifyCollectionChanged oldObs)
                oldObs.CollectionChanged -= OnCollectionChanged;
            _compareB = change.NewValue as IList<MetricPoint>;
            if (_compareB is INotifyCollectionChanged newObs)
                newObs.CollectionChanged += OnCollectionChanged;
            InvalidateVisual();
        }
        if (change.Property == ShowTrainLossProperty ||
            change.Property == ShowValLossProperty ||
            change.Property == ShowAccuracyProperty ||
            change.Property == CompareModeProperty ||
            change.Property == CompareShowTrainLossProperty ||
            change.Property == CompareShowValLossProperty ||
            change.Property == CompareShowAccuracyProperty)
        {
            InvalidateVisual();
        }
    }

    private void OnCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
    {
        InvalidateVisual();
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        if (CompareMode)
            return;
        if (_items == null || _items.Count == 0)
            return;

        var plot = new Rect(0, 0, Bounds.Width, Bounds.Height).Deflate(8);
        if (plot.Width <= 0 || plot.Height <= 0)
            return;

        int minEpoch = _items.Min(m => m.Epoch);
        int maxEpoch = _items.Max(m => m.Epoch);
        if (maxEpoch == minEpoch)
            maxEpoch = minEpoch + 1;

        var pos = e.GetPosition(this);
        int closestIndex = 0;
        double closestDist = double.MaxValue;
        for (int i = 0; i < _items.Count; i++)
        {
            var m = _items[i];
            double x = plot.Left + (m.Epoch - minEpoch) / (double)(maxEpoch - minEpoch) * plot.Width;
            double dist = Math.Abs(pos.X - x);
            if (dist < closestDist)
            {
                closestDist = dist;
                closestIndex = i;
            }
        }

        var point = _items[closestIndex];
        var tip = BuildTooltip(point);
        ToolTip.SetTip(this, tip);
        if (_hoverIndex != closestIndex)
        {
            _hoverIndex = closestIndex;
            InvalidateVisual();
        }
    }

    protected override void OnPointerExited(PointerEventArgs e)
    {
        base.OnPointerExited(e);
        if (CompareMode)
            return;
        ToolTip.SetTip(this, null);
        if (_hoverIndex.HasValue)
        {
            _hoverIndex = null;
            InvalidateVisual();
        }
    }

    private void RenderCompare(DrawingContext context)
    {
        if (_compareA == null || _compareB == null || _compareA.Count < 2 || _compareB.Count < 2)
            return;

        var bounds = new Rect(0, 0, Bounds.Width, Bounds.Height);
        var plot = bounds.Deflate(8);
        if (plot.Width <= 0 || plot.Height <= 0)
            return;

        int maxEpochA = _compareA.Max(m => m.Epoch);
        int maxEpochB = _compareB.Max(m => m.Epoch);
        int maxEpoch = Math.Min(maxEpochA, maxEpochB);
        int minEpoch = Math.Min(_compareA.Min(m => m.Epoch), _compareB.Min(m => m.Epoch));
        if (maxEpoch == minEpoch)
            maxEpoch = minEpoch + 1;

        var values = new List<double>();
        foreach (var m in _compareA.Where(m => m.Epoch <= maxEpoch))
        {
            if (CompareShowTrainLoss)
                values.Add(m.Loss);
            if (CompareShowValLoss && m.ValLoss.HasValue)
                values.Add(m.ValLoss.Value);
            if (CompareShowAccuracy && m.Accuracy.HasValue)
                values.Add(m.Accuracy.Value);
        }
        foreach (var m in _compareB.Where(m => m.Epoch <= maxEpoch))
        {
            if (CompareShowTrainLoss)
                values.Add(m.Loss);
            if (CompareShowValLoss && m.ValLoss.HasValue)
                values.Add(m.ValLoss.Value);
            if (CompareShowAccuracy && m.Accuracy.HasValue)
                values.Add(m.Accuracy.Value);
        }

        if (values.Count == 0)
            return;

        double minY = values.Min();
        double maxY = values.Max();
        if (Math.Abs(maxY - minY) < 1e-9)
            maxY = minY + 1e-3;

        if (CompareShowTrainLoss)
        {
            DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, TrainPen, _compareA, m => m.Loss);
            DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, TrainDashPen, _compareB, m => m.Loss);
        }
        if (CompareShowValLoss)
        {
            DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, ValPen, _compareA, m => m.ValLoss);
            DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, ValDashPen, _compareB, m => m.ValLoss);
        }
        if (CompareShowAccuracy)
        {
            DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, AccPen, _compareA, m => m.Accuracy);
            DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, AccDashPen, _compareB, m => m.Accuracy);
        }
    }

    private void DrawSeries(
        DrawingContext context,
        Rect plot,
        int minEpoch,
        int maxEpoch,
        double minY,
        double maxY,
        Pen pen,
        Func<MetricPoint, double?> selector)
    {
        if (_items == null || _items.Count == 0)
            return;

        DrawSeries(context, plot, minEpoch, maxEpoch, minY, maxY, pen, _items, selector);
    }

    private void DrawSeries(
        DrawingContext context,
        Rect plot,
        int minEpoch,
        int maxEpoch,
        double minY,
        double maxY,
        Pen pen,
        IList<MetricPoint> items,
        Func<MetricPoint, double?> selector)
    {
        if (items.Count == 0)
            return;

        var geometry = new StreamGeometry();
        using var ctx = geometry.Open();
        bool started = false;

        foreach (var m in items)
        {
            var value = selector(m);
            if (!value.HasValue)
            {
                started = false;
                continue;
            }

            double x = plot.Left + (m.Epoch - minEpoch) / (double)(maxEpoch - minEpoch) * plot.Width;
            double y = plot.Bottom - (value.Value - minY) / (maxY - minY) * plot.Height;
            if (!started)
            {
                ctx.BeginFigure(new Point(x, y), false);
                started = true;
            }
            else
            {
                ctx.LineTo(new Point(x, y));
            }
        }

        if (started)
        {
            ctx.EndFigure(false);
            context.DrawGeometry(null, pen, geometry);
        }
    }

    private void DrawHover(
        DrawingContext context,
        Rect plot,
        int minEpoch,
        int maxEpoch,
        double minY,
        double maxY)
    {
        if (_hoverIndex == null || _items == null || _items.Count == 0)
            return;

        int index = _hoverIndex.Value;
        if (index < 0 || index >= _items.Count)
            return;

        var m = _items[index];
        double x = plot.Left + (m.Epoch - minEpoch) / (double)(maxEpoch - minEpoch) * plot.Width;
        context.DrawLine(HoverPen, new Point(x, plot.Top), new Point(x, plot.Bottom));

        DrawHoverPoint(context, x, plot, minY, maxY, m.Loss, ShowTrainLoss);
        DrawHoverPoint(context, x, plot, minY, maxY, m.ValLoss, ShowValLoss);
        DrawHoverPoint(context, x, plot, minY, maxY, m.Accuracy, ShowAccuracy);
    }

    private void DrawHoverPoint(
        DrawingContext context,
        double x,
        Rect plot,
        double minY,
        double maxY,
        double? value,
        bool visible)
    {
        if (!visible || !value.HasValue)
            return;

        double y = plot.Bottom - (value.Value - minY) / (maxY - minY) * plot.Height;
        var center = new Point(x, y);
        context.DrawEllipse(HoverBrush, null, center, 3, 3);
    }

    private void DrawAxisLabels(DrawingContext context, Rect plot)
    {
        var epochText = new FormattedText(
            "epoch",
            System.Globalization.CultureInfo.InvariantCulture,
            FlowDirection.LeftToRight,
            Typeface.Default,
            12,
            Brushes.Gray);
        var lossText = new FormattedText(
            "loss/acc",
            System.Globalization.CultureInfo.InvariantCulture,
            FlowDirection.LeftToRight,
            Typeface.Default,
            12,
            Brushes.Gray);

        context.DrawText(epochText, new Point(plot.Right - epochText.Width, plot.Bottom + 2));
        context.DrawText(lossText, new Point(plot.Left, plot.Top - lossText.Height - 2));
    }

    private string BuildTooltip(MetricPoint point)
    {
        var parts = new List<string> { $"epoch={point.Epoch}" };
        if (ShowTrainLoss)
            parts.Add($"train={point.TrainLossDisplay}");
        if (ShowValLoss)
            parts.Add($"val={point.ValLossDisplay}");
        if (ShowAccuracy)
            parts.Add($"acc={point.AccDisplay}");
        return string.Join("  ", parts);
    }
}
