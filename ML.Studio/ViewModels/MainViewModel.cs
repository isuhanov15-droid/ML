using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Runtime.InteropServices;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using ML.Shared.Protocol;
using ML.Shared.Protocol.Training;
using ML.Studio.Models;
using ML.Studio.Services;

namespace ML.Studio.ViewModels;

public sealed class MainViewModel : INotifyPropertyChanged
{
    private sealed record PendingMetric(string RunId, MetricPoint Point);
    public sealed record DeleteResult(bool Deleted, bool Required, string? RelativePath);

    public enum StudioConnectionState
    {
        Disconnected,
        Connecting,
        Connected
    }

    public enum TrainingState
    {
        Idle,
        Running,
        Finished,
        Stopped,
        Failed
    }

    private readonly HostClient _client = new();
    private readonly WorkspaceService _workspace = new();
    private readonly ProjectFileService _files;
    private readonly DispatcherTimer _structureDebounce;
    private readonly SettingsService _settings = new();
    private readonly LoggingService _logger = new();
    private readonly ConcurrentQueue<PendingMetric> _metricsQueue = new();
    private readonly DispatcherTimer _metricsTimer;
    private string _host = "127.0.0.1";
    private string _portText = "7777";
    private string _status = "Отключено";
    private bool _isConnected;
    private StudioConnectionState _connectionState = StudioConnectionState.Disconnected;
    private TrainingState _trainingState = TrainingState.Idle;
    private CancellationTokenSource? _connectCts;
    private string? _activeRunId;
    private DateTimeOffset? _runStartedUtc;
    private StudioProject? _currentProject;
    private ExplorerNode? _selectedNode;
    private EditorTab? _selectedTab;
    private int _modeIndex;
    private string _paramEpochs = "50";
    private string _paramLearningRate = "0.05";
    private string _paramBatchSize = "8";
    private string _paramOptimizer = "adam";
    private bool _useFileParams = true;
    private TrainStartConfig? _lastTrainConfig;
    private bool _epochsValid = true;
    private bool _learningRateValid = true;
    private bool _batchSizeValid = true;
    private bool _loggedEpochsInvalid;
    private bool _loggedLearningRateInvalid;
    private bool _loggedBatchSizeInvalid;
    private bool _followMetrics = true;
    private bool _showTrainLoss = true;
    private bool _showValLoss = true;
    private bool _showAccuracy = true;
    private string _metricsEveryText = "1";
    private bool _metricsEveryValid = true;
    private bool _loggedMetricsEveryInvalid;
    private double? _lastAcc;
    private double? _lastValLoss;
    private bool _resumeTraining;
    private string? _loadedModelId;
    private int? _loadedInputSize;
    private int? _loadedOutputSize;
    private Bitmap? _decisionBoundaryImage;
    private string _heatmapStatus = "";
    private int _selectedGridSize = 100;
    private double _heatmapThreshold = 0.5;
    private string _selectedHeatmapMode = "Probability(Class 1)";
    private bool _autoRangeFromDataset = true;
    private string _rangeXMin = "0";
    private string _rangeXMax = "1";
    private string _rangeYMin = "0";
    private string _rangeYMax = "1";
    private bool _heatmapRunning;
    private CancellationTokenSource? _heatmapCts;
    private string _inferInput = "0,1";
    private string _inferOutput = "";
    private string _inferPredicted = "";
    private string _inferTopProbabilities = "";
    private string _inferStatus = "Модель не загружена.";
    private string _evalStatus = "";
    private string _evalInputColsText = "";
    private string _evalLabelColText = "";
    private readonly Dictionary<string, Bitmap> _heatmapCache = new();
    private bool _compareMode;
    private string? _selectedRunA;
    private string? _selectedRunB;
    private bool _compareShowTrain = true;
    private bool _compareShowVal = true;
    private bool _compareShowAcc = true;
    private IList<MetricPoint>? _compareItemsA;
    private IList<MetricPoint>? _compareItemsB;
    private string? _remoteProjectId;
    private string? _remoteExperimentId;
    private string _structurePreview = "";
    private GraphModel? _structureGraph;
    private int? _selectedGraphNodeId;
    private string _selectedLayerTitle = "";
    private string _selectedLayerSubtitle = "";
    private readonly ObservableCollection<string> _selectedLayerMeta = new();

    public ObservableCollection<ExplorerNode> ProjectTree { get; } = new();
    public ObservableCollection<EditorTab> EditorTabs { get; } = new();
    public ObservableCollection<ExplorerNode> JsonTree { get; } = new();
    public ObservableCollection<string> GraphItems { get; } = new();
    public ObservableCollection<MetricPoint> Metrics { get; } = new();
    public ObservableCollection<string> StructureItems { get; } = new();
    public ObservableCollection<string> SelectedLayerMeta => _selectedLayerMeta;
    public ObservableCollection<InferInputField> InferInputs { get; } = new();
    public ObservableCollection<InferHistoryItem> InferHistory { get; } = new();
    public ObservableCollection<string> EvalSamples { get; } = new();
    public ObservableCollection<SoftmaxBarItem> SoftmaxBars { get; } = new();
    public ObservableCollection<int> GridSizes { get; } = new();
    public ObservableCollection<string> RunIds { get; } = new();
    public ObservableCollection<string> HeatmapModes { get; } = new();
    public LoggingService Logger => _logger;
    public ObservableCollection<string> Logs => _logger.Entries;

    public MainViewModel()
    {
        _files = new ProjectFileService(_workspace);
        _client.EventReceived += OnEventReceived;
        _client.Disconnected += OnDisconnected;

        GridSizes.Add(50);
        GridSizes.Add(100);
        GridSizes.Add(200);
        _selectedGridSize = 100;
        HeatmapModes.Add("Probability(Class 1)");
        HeatmapModes.Add("Argmax class");

        _metricsTimer = new DispatcherTimer
        {
            Interval = TimeSpan.FromMilliseconds(150)
        };
        _metricsTimer.Tick += (_, _) => FlushMetrics();
        _metricsTimer.Start();

        _structureDebounce = new DispatcherTimer
        {
            Interval = TimeSpan.FromMilliseconds(300)
        };
        _structureDebounce.Tick += async (_, _) =>
        {
            _structureDebounce.Stop();
            await LoadStructureAsync();
        };
    }

    public string Host
    {
        get => _host;
        set => SetField(ref _host, value);
    }

    public string PortText
    {
        get => _portText;
        set => SetField(ref _portText, value);
    }

    public string Status
    {
        get => _status;
        set => SetField(ref _status, value);
    }

    public bool IsConnected
    {
        get => _isConnected;
        private set => SetField(ref _isConnected, value);
    }

    public StudioConnectionState ConnectionState
    {
        get => _connectionState;
        private set
        {
            if (SetField(ref _connectionState, value))
                RaiseComputedState();
        }
    }

    public TrainingState TrainingStatus
    {
        get => _trainingState;
        private set
        {
            if (SetField(ref _trainingState, value))
                RaiseComputedState();
        }
    }

    public string CurrentProjectTitle
    {
        get => _currentProject == null ? "Проект не выбран" : _currentProject.Name;
    }

    public ExplorerNode? SelectedNode
    {
        get => _selectedNode;
        set => SetField(ref _selectedNode, value);
    }

    public EditorTab? SelectedTab
    {
        get => _selectedTab;
        set => SetField(ref _selectedTab, value);
    }

    public int ModeIndex
    {
        get => _modeIndex;
        set
        {
            if (SetField(ref _modeIndex, value))
            {
                OnPropertyChanged(nameof(IsEditorMode));
                OnPropertyChanged(nameof(IsGraphMode));
                OnPropertyChanged(nameof(IsStructureMode));
                OnPropertyChanged(nameof(IsRunMode));
            }
        }
    }

    public bool IsEditorMode => ModeIndex == 0;
    public bool IsGraphMode => ModeIndex == 1;
    public bool IsStructureMode => ModeIndex == 2;
    public bool IsRunMode => ModeIndex == 3;

    public string ParamEpochs
    {
        get => _paramEpochs;
        set
        {
            if (SetField(ref _paramEpochs, value))
                ValidateEpochs();
        }
    }

    public string ParamLearningRate
    {
        get => _paramLearningRate;
        set
        {
            if (SetField(ref _paramLearningRate, value))
                ValidateLearningRate();
        }
    }

    public string ParamBatchSize
    {
        get => _paramBatchSize;
        set
        {
            if (SetField(ref _paramBatchSize, value))
                ValidateBatchSize();
        }
    }

    public string ParamOptimizer
    {
        get => _paramOptimizer;
        set
        {
            if (SetField(ref _paramOptimizer, value))
                ValidateOptimizer();
        }
    }

    public bool UseFileParams
    {
        get => _useFileParams;
        set => SetField(ref _useFileParams, value);
    }

    public bool EpochsInvalid => !_epochsValid;
    public bool LearningRateInvalid => !_learningRateValid;
    public bool BatchSizeInvalid => !_batchSizeValid;

    public IReadOnlyList<string> OptimizerOptions { get; } = new[] { "adam", "sgd", "adamw" };

    public bool FollowMetrics
    {
        get => _followMetrics;
        set => SetField(ref _followMetrics, value);
    }

    public bool ShowTrainLoss
    {
        get => _showTrainLoss;
        set => SetField(ref _showTrainLoss, value);
    }

    public bool ShowValLoss
    {
        get => _showValLoss;
        set => SetField(ref _showValLoss, value);
    }

    public bool ShowAccuracy
    {
        get => _showAccuracy;
        set => SetField(ref _showAccuracy, value);
    }

    public string MetricsEveryText
    {
        get => _metricsEveryText;
        set
        {
            if (SetField(ref _metricsEveryText, value))
                ValidateMetricsEvery();
        }
    }

    public bool MetricsEveryInvalid => !_metricsEveryValid;

    public bool ResumeTraining
    {
        get => _resumeTraining;
        set => SetField(ref _resumeTraining, value);
    }

    public string InferInput
    {
        get => _inferInput;
        set => SetField(ref _inferInput, value);
    }

    public string InferOutput
    {
        get => _inferOutput;
        private set => SetField(ref _inferOutput, value);
    }

    public string InferPredicted
    {
        get => _inferPredicted;
        private set => SetField(ref _inferPredicted, value);
    }

    public string InferTopProbabilities
    {
        get => _inferTopProbabilities;
        private set => SetField(ref _inferTopProbabilities, value);
    }

    public string InferStatus
    {
        get => _inferStatus;
        private set => SetField(ref _inferStatus, value);
    }

    public string EvalStatus
    {
        get => _evalStatus;
        private set => SetField(ref _evalStatus, value);
    }

    public string EvalInputColsText
    {
        get => _evalInputColsText;
        set => SetField(ref _evalInputColsText, value);
    }

    public string EvalLabelColText
    {
        get => _evalLabelColText;
        set => SetField(ref _evalLabelColText, value);
    }

    public bool ShowSoftmaxBars => SoftmaxBars.Count > 0;
    public bool ShowDecisionBoundary => _loadedInputSize == 2;

    public Bitmap? DecisionBoundaryImage
    {
        get => _decisionBoundaryImage;
        private set => SetField(ref _decisionBoundaryImage, value);
    }

    public string HeatmapStatus
    {
        get => _heatmapStatus;
        private set => SetField(ref _heatmapStatus, value);
    }

    public int SelectedGridSize
    {
        get => _selectedGridSize;
        set => SetField(ref _selectedGridSize, value);
    }

    public double HeatmapThreshold
    {
        get => _heatmapThreshold;
        set => SetField(ref _heatmapThreshold, value);
    }

    public string SelectedHeatmapMode
    {
        get => _selectedHeatmapMode;
        set
        {
            if (SetField(ref _selectedHeatmapMode, value))
            {
                OnPropertyChanged(nameof(UseProbabilityMode));
                OnPropertyChanged(nameof(ShowThreshold));
            }
        }
    }

    public bool UseProbabilityMode => SelectedHeatmapMode == "Probability(Class 1)";
    public bool ShowThreshold => UseProbabilityMode;

    public bool AutoRangeFromDataset
    {
        get => _autoRangeFromDataset;
        set
        {
            if (SetField(ref _autoRangeFromDataset, value))
                OnPropertyChanged(nameof(ManualRangeEnabled));
        }
    }

    public bool ManualRangeEnabled => !AutoRangeFromDataset;

    public string RangeXMin
    {
        get => _rangeXMin;
        set => SetField(ref _rangeXMin, value);
    }

    public string RangeXMax
    {
        get => _rangeXMax;
        set => SetField(ref _rangeXMax, value);
    }

    public string RangeYMin
    {
        get => _rangeYMin;
        set => SetField(ref _rangeYMin, value);
    }

    public string RangeYMax
    {
        get => _rangeYMax;
        set => SetField(ref _rangeYMax, value);
    }

    public bool HeatmapRunning
    {
        get => _heatmapRunning;
        private set
        {
            if (SetField(ref _heatmapRunning, value))
                OnPropertyChanged(nameof(CanCancelHeatmap));
        }
    }

    public bool CanCancelHeatmap => HeatmapRunning;

    public bool CompareMode
    {
        get => _compareMode;
        private set
        {
            if (SetField(ref _compareMode, value))
                OnPropertyChanged(nameof(StatusLine));
        }
    }

    public string? SelectedRunA
    {
        get => _selectedRunA;
        set => SetField(ref _selectedRunA, value);
    }

    public string? SelectedRunB
    {
        get => _selectedRunB;
        set => SetField(ref _selectedRunB, value);
    }

    public bool CompareShowTrain
    {
        get => _compareShowTrain;
        set => SetField(ref _compareShowTrain, value);
    }

    public bool CompareShowVal
    {
        get => _compareShowVal;
        set => SetField(ref _compareShowVal, value);
    }

    public bool CompareShowAcc
    {
        get => _compareShowAcc;
        set => SetField(ref _compareShowAcc, value);
    }

    public IList<MetricPoint>? CompareItemsA
    {
        get => _compareItemsA;
        private set => SetField(ref _compareItemsA, value);
    }

    public IList<MetricPoint>? CompareItemsB
    {
        get => _compareItemsB;
        private set => SetField(ref _compareItemsB, value);
    }

    public string ActiveRunLabel => string.IsNullOrWhiteSpace(_activeRunId) ? "" : $"запуск={_activeRunId}";
    public string? CurrentRunId => _activeRunId;
    public string ShortRunId => string.IsNullOrWhiteSpace(_activeRunId)
        ? ""
        : _activeRunId.Length <= 8
            ? _activeRunId
            : _activeRunId[..8];
    public bool HasRunId => !string.IsNullOrWhiteSpace(_activeRunId);

    public string StatusLine
    {
        get
        {
            string run = string.IsNullOrWhiteSpace(ShortRunId) ? "" : $" ({ShortRunId})";
            return $"{FormatConnectionState(ConnectionState)} | {FormatTrainingState(TrainingStatus)}{run}";
        }
    }

    public bool CanConnect => ConnectionState == StudioConnectionState.Disconnected;
    public bool CanDisconnect => ConnectionState == StudioConnectionState.Connected || ConnectionState == StudioConnectionState.Connecting;
    public bool CanRun => ConnectionState == StudioConnectionState.Connected
                         && _currentProject != null
                         && RequiredFilesPresent()
                         && TrainingStatus != TrainingState.Running;
    public bool CanStop => TrainingStatus == TrainingState.Running;
    public string StructurePreview
    {
        get => _structurePreview;
        private set => SetField(ref _structurePreview, value);
    }

    public GraphModel? StructureGraph
    {
        get => _structureGraph;
        private set => SetField(ref _structureGraph, value);
    }

    public int? SelectedGraphNodeId
    {
        get => _selectedGraphNodeId;
        set
        {
            if (SetField(ref _selectedGraphNodeId, value))
                UpdateSelectedLayer();
        }
    }

    public string SelectedLayerTitle
    {
        get => _selectedLayerTitle;
        private set => SetField(ref _selectedLayerTitle, value);
    }

    public string SelectedLayerSubtitle
    {
        get => _selectedLayerSubtitle;
        private set => SetField(ref _selectedLayerSubtitle, value);
    }

    public async Task InitializeAsync()
    {
        var settings = _settings.Load();
        if (!string.IsNullOrWhiteSpace(settings.LastProjectPath) && File.Exists(settings.LastProjectPath))
        {
            try
            {
                var project = await _workspace.OpenProjectAsync(settings.LastProjectPath);
                await SetCurrentProjectAsync(project);
            }
            catch
            {
                _logger.Append("warn", "Не удалось загрузить последний проект.");
            }
        }
    }

    public async Task ConnectAsync()
    {
        if (!int.TryParse(PortText, out var port))
        {
            Status = "Неверный порт.";
            _logger.Append("error", "Неверный порт для подключения.");
            return;
        }

        if (ConnectionState == StudioConnectionState.Connecting)
            return;

        Status = "Подключение...";
        ConnectionState = StudioConnectionState.Connecting;
        _connectCts?.Cancel();
        _connectCts = new CancellationTokenSource();
        try
        {
            await _client.ConnectAsync(Host, port, _connectCts.Token);
            IsConnected = true;
            ConnectionState = StudioConnectionState.Connected;
            Status = "Подключено";
            _logger.Append("info", $"Подключено к {Host}:{port}.");

            var info = await _client.CallAsync<HostInfoDto>("host.info", null);
            _logger.Append("info", $"Инфо хоста: {info.appName} {info.hostVersion} протокол={info.protocolVersion}");

            var caps = await _client.CallAsync<HostCapabilitiesDto>("host.capabilities", null);
            _logger.Append("info", $"Возможности: cores={caps.cpu.logicalCores} backends=[{string.Join(',', caps.supportedBackends)}]");
        }
        catch (OperationCanceledException)
        {
            Status = "Подключение отменено.";
            ConnectionState = StudioConnectionState.Disconnected;
            IsConnected = false;
            _logger.Append("warn", "Подключение отменено.");
        }
        catch (Exception ex)
        {
            Status = $"Ошибка подключения: {ex.Message}";
            await _client.DisconnectAsync();
            IsConnected = false;
            ConnectionState = StudioConnectionState.Disconnected;
            _logger.Append("error", $"Ошибка подключения: {ex.Message}");
        }
    }

    public async Task DisconnectAsync()
    {
        _connectCts?.Cancel();
        await _client.DisconnectAsync();
        Status = "Отключено";
        IsConnected = false;
        ConnectionState = StudioConnectionState.Disconnected;
        TrainingStatus = TrainingState.Idle;
        _activeRunId = null;
        OnPropertyChanged(nameof(ShortRunId));
        OnPropertyChanged(nameof(HasRunId));
        OnPropertyChanged(nameof(StatusLine));
        _logger.Append("info", "Отключено от хоста.");
    }

    public async Task CreateProjectAsync(string folderPath, string name, string projectType)
    {
        try
        {
            var project = await _workspace.CreateProjectAsync(folderPath, name, projectType);
            await SetCurrentProjectAsync(project);
            _logger.Append("info", $"Проект создан: {project.Name}");
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка создания проекта: {ex.Message}");
        }
    }

    public async Task OpenProjectAsync(string mlprojPath)
    {
        try
        {
            var project = await _workspace.OpenProjectAsync(mlprojPath);
            await SetCurrentProjectAsync(project);
            _logger.Append("info", $"Проект открыт: {project.Name}");
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка открытия проекта: {ex.Message}");
        }
    }

    private async Task SetCurrentProjectAsync(StudioProject project)
    {
        _currentProject = project;
        _remoteProjectId = null;
        _remoteExperimentId = null;
        OnPropertyChanged(nameof(CurrentProjectTitle));
        RaiseComputedState();
        ResetCompareState();
        await RefreshProjectFilesAsync();
        BuildProjectTree(project);
        await LoadParamsAsync();
        await LoadGraphAsync();
        await LoadStructureAsync();
        await RefreshRunListAsync();

        var settings = _settings.Load();
        settings.LastProjectPath = _workspace.GetProjectFilePath(project);
        _settings.Save(settings);
    }

    public async Task CreateFileAsync(string relativePath)
    {
        if (_currentProject == null)
            return;

        try
        {
            var item = await _files.CreateFileAsync(_currentProject, relativePath, "", "Other");
            _currentProject = _currentProject with { Files = _currentProject.Files.Concat(new[] { item }).ToArray() };
            BuildProjectTree(_currentProject);
            RaiseComputedState();
            await OpenFileAsync(item.FullPath);
            _logger.Append("info", $"Файл создан: {relativePath}");
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка создания файла: {ex.Message}");
        }
    }

    public async Task RenameSelectedFileAsync(string newRelativePath)
    {
        if (_currentProject == null || SelectedNode?.FullPath == null)
            return;

        try
        {
            var oldRel = GetRelative(SelectedNode.FullPath);
            var newRel = newRelativePath.Replace("\\", "/");
            await _files.RenameFileAsync(_currentProject, oldRel, newRel);
            _currentProject = await _workspace.OpenProjectAsync(_workspace.GetProjectFilePath(_currentProject));
            BuildProjectTree(_currentProject);
            RaiseComputedState();
            UpdateOpenTabPath(oldRel, newRel);
            _logger.Append("info", $"Файл переименован: {oldRel} -> {newRel}");
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка переименования файла: {ex.Message}");
        }
    }

    public async Task<DeleteResult> DeleteSelectedFileAsync()
    {
        if (_currentProject == null || SelectedNode?.FullPath == null)
            return new DeleteResult(false, false, null);

        try
        {
            var selectedFullPath = SelectedNode.FullPath;
            var rel = GetRelative(selectedFullPath);
            await _files.DeleteFileAsync(_currentProject, rel);
            _currentProject = await _workspace.OpenProjectAsync(_workspace.GetProjectFilePath(_currentProject));
            BuildProjectTree(_currentProject);
            RaiseComputedState();
            CloseTabByPath(selectedFullPath);
            _logger.Append("info", $"Файл удален: {rel}");
            return new DeleteResult(true, IsRequiredFile(rel), rel);
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка удаления файла: {ex.Message}");
            return new DeleteResult(false, false, null);
        }
    }

    public void RevealSelectedFile()
    {
        if (SelectedNode?.FullPath == null)
            return;

        _files.RevealInFileManager(SelectedNode.FullPath);
    }

    public async Task OpenSelectedFileAsync()
    {
        if (SelectedNode?.FullPath == null)
            return;

        try
        {
            await OpenFileAsync(SelectedNode.FullPath);
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка открытия файла: {ex.Message}");
        }
    }

    public async Task SaveCurrentFileAsync()
    {
        if (SelectedTab == null)
            return;

        try
        {
            await _files.WriteTextAsync(SelectedTab.FullPath, SelectedTab.Text);
            SelectedTab.MarkClean();
            Status = "Файл сохранен.";
            _logger.Append("info", $"Файл сохранен: {SelectedTab.Title}");
            InvalidateHeatmapCacheIfNeeded(SelectedTab.FullPath);
            if (IsNetworkFile(SelectedTab.FullPath))
                ScheduleStructureReload();
            await LoadParamsAsync();
            await LoadGraphAsync();
            BuildJsonTree(SelectedTab.Text);
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка сохранения файла: {ex.Message}");
        }
    }

    public async Task SaveAllAsync()
    {
        var dirtyTabs = EditorTabs.Where(t => t.IsDirty).ToList();
        bool touchedNetwork = false;
        foreach (var tab in dirtyTabs)
        {
            try
            {
                await _files.WriteTextAsync(tab.FullPath, tab.Text);
                tab.MarkClean();
                InvalidateHeatmapCacheIfNeeded(tab.FullPath);
                if (IsNetworkFile(tab.FullPath))
                    touchedNetwork = true;
            }
            catch (Exception ex)
            {
                _logger.Append("error", $"Ошибка сохранения {tab.Title}: {ex.Message}");
            }
        }

        if (dirtyTabs.Count > 0)
        {
            Status = $"Сохранено файлов: {dirtyTabs.Count}.";
            _logger.Append("info", $"Сохранено файлов: {dirtyTabs.Count}.");
        }

        if (touchedNetwork)
            ScheduleStructureReload();
    }

    public async Task SaveTabAsync(EditorTab tab)
    {
        try
        {
            await _files.WriteTextAsync(tab.FullPath, tab.Text);
            tab.MarkClean();
            InvalidateHeatmapCacheIfNeeded(tab.FullPath);
            if (IsNetworkFile(tab.FullPath))
                ScheduleStructureReload();
            if (ReferenceEquals(SelectedTab, tab))
                Status = "Файл сохранен.";
            _logger.Append("info", $"Файл сохранен: {tab.Title}");
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка сохранения файла: {ex.Message}");
        }
    }

    public async Task SaveTabAsAsync(EditorTab tab, string relativePath)
    {
        if (_currentProject == null)
            return;

        var item = await _files.CreateFileAsync(_currentProject, relativePath, tab.Text, "Other");
        _currentProject = _currentProject with { Files = _currentProject.Files.Concat(new[] { item }).ToArray() };
        BuildProjectTree(_currentProject);
        RaiseComputedState();
        _logger.Append("info", $"Файл сохранен как: {relativePath}");
        await OpenFileAsync(item.FullPath);
    }

    public void CloseTab(EditorTab tab)
    {
        var wasSelected = ReferenceEquals(SelectedTab, tab);
        EditorTabs.Remove(tab);
        if (wasSelected)
            SelectedTab = EditorTabs.LastOrDefault();
    }

    public EditorTab? GetSelectedFileTab()
    {
        if (SelectedNode?.FullPath == null)
            return null;
        return EditorTabs.FirstOrDefault(t => string.Equals(t.FullPath, SelectedNode.FullPath, StringComparison.OrdinalIgnoreCase));
    }

    public async Task RecreateRequiredFileAsync(string relativePath)
    {
        if (_currentProject == null)
            return;

        var content = WorkspaceService.GetDefaultContentFor(relativePath);
        if (string.IsNullOrWhiteSpace(content))
            return;

        var kind = WorkspaceService.GetDefaultKindFor(relativePath);
        var item = await _files.CreateFileAsync(_currentProject, relativePath, content, kind);
        _currentProject = _currentProject with { Files = _currentProject.Files.Concat(new[] { item }).ToArray() };
        BuildProjectTree(_currentProject);
        RaiseComputedState();
        _logger.Append("info", $"Файл пересоздан: {relativePath}");
    }

    public async Task SaveCurrentFileAsAsync(string relativePath)
    {
        if (_currentProject == null || SelectedTab == null)
            return;

        try
        {
            var item = await _files.CreateFileAsync(_currentProject, relativePath, SelectedTab.Text, "Other");
            _currentProject = _currentProject with { Files = _currentProject.Files.Concat(new[] { item }).ToArray() };
            BuildProjectTree(_currentProject);
            await OpenFileAsync(item.FullPath);
            _logger.Append("info", $"Файл сохранен как: {relativePath}");
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка сохранения файла: {ex.Message}");
        }
    }

    public async Task SaveParamsAsync()
    {
        if (_currentProject == null)
            return;

        try
        {
            if (!EnsureParamsValid())
            {
                _logger.Append("warn", "Некорректные параметры. Сохранение отменено.");
                return;
            }

            var epochs = TryParseInt(ParamEpochs, 50, "epochs");
            var batch = TryParseInt(ParamBatchSize, 8, "batchSize");
            var lr = TryParseDoubleFlexible(ParamLearningRate, 0.05, "learningRate");

            var path = Path.Combine(_currentProject.RootPath, "params.json");
            var json = "{\n" +
                       $"  \"epochs\": {epochs.ToString(CultureInfo.InvariantCulture)},\n" +
                       $"  \"learningRate\": {lr.ToString(CultureInfo.InvariantCulture)},\n" +
                       $"  \"batchSize\": {batch.ToString(CultureInfo.InvariantCulture)},\n" +
                       $"  \"optimizer\": \"{ParamOptimizer}\"\n" +
                       "}\n";
            await _files.WriteTextAsync(path, json);
            Status = "Параметры сохранены.";
            _logger.Append("info", "Параметры сохранены.");
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка сохранения параметров: {ex.Message}");
        }
    }

    public async Task ReloadParamsAsync()
    {
        try
        {
            await LoadParamsAsync();
            Status = "Параметры загружены.";
            _logger.Append("info", "Параметры загружены.");
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка загрузки параметров: {ex.Message}");
        }
    }

    public async Task StartTrainingAsync()
    {
        if (!CanRun)
            return;

        try
        {
            if (!EnsureParamsValid())
            {
                _logger.Append("warn", "Некорректные параметры. Запуск отменен.");
                return;
            }

            if (ResumeTraining && _currentProject != null)
            {
                var modelPath = Path.Combine(_currentProject.RootPath, "model.json");
                if (!File.Exists(modelPath))
                    _logger.Append("warn", $"Файл модели не найден: {modelPath}. Запуск без resume.");
            }

            var (projectId, experimentId) = await EnsureRemoteExperimentAsync();
            var config = await BuildTrainConfigAsync();
            _lastTrainConfig = config;
            var result = await _client.CallAsync<TrainStartResult>("train.start", new { projectId, experimentId, config });
            _activeRunId = result.runId;
            _runStartedUtc = DateTimeOffset.Now;
            TrainingStatus = TrainingState.Running;
            OnPropertyChanged(nameof(ActiveRunLabel));
            OnPropertyChanged(nameof(ShortRunId));
            OnPropertyChanged(nameof(HasRunId));
            OnPropertyChanged(nameof(StatusLine));
            AddRunId(_activeRunId);
            Status = "Запуск начат.";
            _logger.Append("info", $"Запуск обучения: runId={_activeRunId}");
        }
        catch (Exception ex)
        {
            Status = $"Ошибка запуска: {ex.Message}";
            TrainingStatus = TrainingState.Failed;
            _logger.Append("error", $"Ошибка запуска: {ex.Message}");
        }
    }

    public async Task StopTrainingAsync()
    {
        if (!CanStop)
            return;

        try
        {
            await _client.CallAsync<object>("train.stop", new { runId = _activeRunId });
            Status = "Остановка запрошена.";
            if (_runStartedUtc.HasValue)
            {
                var duration = DateTimeOffset.Now - _runStartedUtc.Value;
                _logger.Append("info", $"Длительность: {duration:g}");
            }
            _logger.Append("info", "Остановка обучения запрошена.");
        }
        catch (Exception ex)
        {
            Status = $"Ошибка остановки: {ex.Message}";
            _logger.Append("error", $"Ошибка остановки: {ex.Message}");
        }
    }

    public async Task LoadModelAsync()
    {
        if (!IsConnected || _currentProject == null)
            return;

        try
        {
            var modelPath = Path.Combine(_currentProject.RootPath, "model.json");
            var request = new ModelLoadRequest(
                ProjectId: _currentProject.Name ?? "local",
                RunId: _activeRunId,
                ModelPath: modelPath);
            var response = await _client.CallAsync<ModelLoadResponse>("model.load", request);
            InvalidateHeatmapCache("model.load");
            _loadedModelId = response.ModelId;
            _loadedInputSize = response.InputSize;
            _loadedOutputSize = response.OutputSize;
            BuildInferInputs();
            InferStatus = $"Модель загружена: {modelPath} (in={response.InputSize}, out={response.OutputSize})";
            OnPropertyChanged(nameof(LoadedModelShortId));
            OnPropertyChanged(nameof(LoadedModelInfo));
            OnPropertyChanged(nameof(ShowDecisionBoundary));
            _logger.Append("info", $"model.load: {modelPath}");
        }
        catch (Exception ex)
        {
            InferStatus = $"Ошибка загрузки модели: {ex.Message}";
            _logger.Append("error", $"Ошибка model.load: {ex.Message}");
        }
    }

    public async Task InferSingleAsync()
    {
        if (!IsConnected)
            return;

        try
        {
            if (string.IsNullOrWhiteSpace(_loadedModelId))
                throw new InvalidOperationException("Модель не загружена.");

            var input = BuildInferInputVector();
            var request = new InferSingleRequest(_loadedModelId, input);
            var result = await _client.CallAsync<InferSingleResponse>("infer.single", request);
            InferOutput = string.Join(", ", result.Output.Select(v => v.ToString("0.0000##", CultureInfo.InvariantCulture)));
            InferPredicted = result.PredictedIndex?.ToString(CultureInfo.InvariantCulture) ?? "-";
            InferTopProbabilities = FormatTopProbabilities(result.Probabilities);
            InferStatus = "Инференс выполнен.";

            var historyItem = new InferHistoryItem(
                string.Join(", ", input.Select(v => v.ToString("0.###", CultureInfo.InvariantCulture))),
                InferPredicted);
            AddInferHistory(historyItem);
            UpdateSoftmaxBars(result.Probabilities);
        }
        catch (Exception ex)
        {
            InferStatus = $"Ошибка инференса: {ex.Message}";
            _logger.Append("error", $"Ошибка infer.single: {ex.Message}");
        }
    }

    public async Task EvalDatasetAsync()
    {
        if (!IsConnected || _currentProject == null)
            return;

        try
        {
            if (string.IsNullOrWhiteSpace(_loadedModelId))
                throw new InvalidOperationException("Модель не загружена.");

            var datasetPath = Path.Combine(_currentProject.RootPath, "dataset.csv");
            var schema = DetectDatasetSchema(datasetPath);
            if (string.IsNullOrWhiteSpace(EvalInputColsText))
                EvalInputColsText = string.Join(",", schema.InputCols);
            if (string.IsNullOrWhiteSpace(EvalLabelColText))
                EvalLabelColText = schema.LabelCol.ToString(CultureInfo.InvariantCulture);
            if (TryParseColumns(EvalInputColsText, out var inputCols))
                schema = schema with { InputCols = inputCols };
            if (TryParseColumnIndex(EvalLabelColText, out var labelCol))
                schema = schema with { LabelCol = labelCol };

            var request = new EvalDatasetRequest(
                ModelId: _loadedModelId,
                DatasetPath: datasetPath,
                HasHeader: schema.HasHeader,
                InputCols: schema.InputCols,
                LabelCol: schema.LabelCol);
            var result = await _client.CallAsync<EvalDatasetResponse>("eval.dataset", request);
            var accText = result.Accuracy.HasValue
                ? result.Accuracy.Value.ToString("0.0000##", CultureInfo.InvariantCulture)
                : "-";
            EvalStatus = $"accuracy={accText} (samples={result.Samples.Length})";
            EvalSamples.Clear();
            foreach (var sample in result.Samples.Take(10))
            {
                var input = string.Join(", ", sample.Input.Select(v => v.ToString("0.###", CultureInfo.InvariantCulture)));
                var expected = sample.Expected != null
                    ? string.Join(", ", sample.Expected.Select(v => v.ToString("0.###", CultureInfo.InvariantCulture)))
                    : "-";
                var predicted = sample.Predicted != null
                    ? string.Join(", ", sample.Predicted.Select(v => v.ToString("0.###", CultureInfo.InvariantCulture)))
                    : "-";
                EvalSamples.Add($"{input} -> exp[{expected}] pred[{predicted}]");
            }
        }
        catch (Exception ex)
        {
            EvalStatus = $"Ошибка eval.dataset: {ex.Message}";
            _logger.Append("error", $"Ошибка eval.dataset: {ex.Message}");
        }
    }

    public async Task BuildHeatmapAsync()
    {
        if (!IsConnected || _currentProject == null)
            return;
        if (string.IsNullOrWhiteSpace(_loadedModelId))
        {
            HeatmapStatus = "Сначала загрузите модель.";
            return;
        }
        if (_loadedInputSize != 2)
        {
            HeatmapStatus = "Decision boundary доступен только для inputSize=2.";
            return;
        }

        try
        {
            _heatmapCts?.Cancel();
            _heatmapCts = new CancellationTokenSource();
            var ct = _heatmapCts.Token;
            HeatmapRunning = true;

            var datasetPath = Path.Combine(_currentProject.RootPath, "dataset.csv");
            var schema = DetectDatasetSchema(datasetPath);
            if (string.IsNullOrWhiteSpace(EvalInputColsText))
                EvalInputColsText = string.Join(",", schema.InputCols);
            if (string.IsNullOrWhiteSpace(EvalLabelColText))
                EvalLabelColText = schema.LabelCol.ToString(CultureInfo.InvariantCulture);
            if (TryParseColumns(EvalInputColsText, out var inputCols))
                schema = schema with { InputCols = inputCols };
            if (TryParseColumnIndex(EvalLabelColText, out var labelCol))
                schema = schema with { LabelCol = labelCol };

            var points = LoadDatasetPoints(datasetPath, schema);
            var range = AutoRangeFromDataset
                ? GetRange(points)
                : GetManualRange();
            var key = BuildHeatmapKey(_loadedModelId, SelectedGridSize, range, HeatmapThreshold, UseProbabilityMode);
            if (_heatmapCache.TryGetValue(key, out var cached))
            {
                DecisionBoundaryImage = cached;
                HeatmapStatus = "Heatmap (cache).";
                return;
            }

            var inputs = BuildGridInputs(SelectedGridSize, range);
            var request = new InferBatchRequest(_loadedModelId, inputs);
            var response = await _client.CallAsync<InferBatchResponse>("infer.batch", request, ct);

            ct.ThrowIfCancellationRequested();
            var bitmap = await Task.Run(() =>
            {
                ct.ThrowIfCancellationRequested();
                return RenderHeatmap(SelectedGridSize, range, response, points, HeatmapThreshold, UseProbabilityMode);
            }, ct);
            DecisionBoundaryImage = bitmap;
            _heatmapCache[key] = bitmap;
            HeatmapStatus = $"Heatmap построен: {SelectedGridSize}x{SelectedGridSize}";
        }
        catch (OperationCanceledException)
        {
            HeatmapStatus = "Heatmap отменен.";
        }
        catch (Exception ex)
        {
            HeatmapStatus = $"Ошибка heatmap: {ex.Message}";
            _logger.Append("error", $"Ошибка heatmap: {ex.Message}");
        }
        finally
        {
            HeatmapRunning = false;
        }
    }

    public void ClearHeatmap()
    {
        DecisionBoundaryImage = null;
        HeatmapStatus = "Heatmap очищен.";
    }

    public void CancelHeatmap()
    {
        _heatmapCts?.Cancel();
    }

    private async Task<TrainStartConfig> BuildTrainConfigAsync()
    {
        if (_currentProject == null)
            return DefaultTrainConfig();

        var networkPath = Path.Combine(_currentProject.RootPath, "network.json");
        var paramsPath = Path.Combine(_currentProject.RootPath, "params.json");
        var datasetPath = Path.Combine(_currentProject.RootPath, "dataset.csv");
        var modelPath = Path.Combine(_currentProject.RootPath, "model.json");

        var net = await ReadJsonAsync<TrainNetworkConfig>(networkPath) ?? DefaultTrainConfig().Network;
        TrainTrainConfig train;
        if (UseFileParams)
        {
            train = await ReadJsonAsync<TrainTrainConfig>(paramsPath) ?? BuildTrainConfigFromFields();
        }
        else
        {
            train = BuildTrainConfigFromFields();
        }

        bool shouldResume = ResumeTraining || File.Exists(modelPath);
        if (File.Exists(modelPath) && !ResumeTraining)
            _logger.Append("info", "model.json найден, resume включен автоматически.");

        return new TrainStartConfig
        {
            Network = net,
            Train = train,
            Data = new TrainDataConfig
            {
                Preset = "FILE",
                DatasetPath = datasetPath
            },
            Resume = shouldResume,
            ModelPath = modelPath
        };
    }

    private async Task<(string projectId, string experimentId)> EnsureRemoteExperimentAsync()
    {
        if (_currentProject == null)
            throw new InvalidOperationException("Проект не выбран.");

        if (!string.IsNullOrWhiteSpace(_remoteProjectId) && !string.IsNullOrWhiteSpace(_remoteExperimentId))
            return (_remoteProjectId!, _remoteExperimentId!);

        var projectList = await _client.CallAsync<ProjectDto[]>("project.list", null);
        var project = projectList.FirstOrDefault(p => string.Equals(p.name, _currentProject.Name, StringComparison.OrdinalIgnoreCase));
        if (project == null)
        {
            project = await _client.CallAsync<ProjectDto>("project.create", new { name = _currentProject.Name });
            _logger.Append("info", $"Создан проект в хосте: {project.name}");
        }

        _remoteProjectId = project.projectId;

        var experiments = await _client.CallAsync<ExperimentDto[]>("experiment.list", new { projectId = _remoteProjectId });
        var experiment = experiments.FirstOrDefault(e => string.Equals(e.name, "main", StringComparison.OrdinalIgnoreCase));
        if (experiment == null)
        {
            experiment = await _client.CallAsync<ExperimentDto>("experiment.create", new
            {
                projectId = _remoteProjectId,
                name = "main",
                description = "auto"
            });
            _logger.Append("info", $"Создан эксперимент в хосте: {experiment.name}");
        }

        _remoteExperimentId = experiment.experimentId;
        return (_remoteProjectId, _remoteExperimentId);
    }

    private static async Task<T?> ReadJsonAsync<T>(string path)
    {
        if (!File.Exists(path))
            return default;
        var json = await File.ReadAllTextAsync(path);
        return JsonSerializer.Deserialize<T>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
    }

    private TrainTrainConfig BuildTrainConfigFromFields()
    {
        int epochs = TryParseInt(ParamEpochs, 50, "epochs");
        int batch = TryParseInt(ParamBatchSize, 8, "batchSize");
        double lr = TryParseDoubleFlexible(ParamLearningRate, 0.05, "learningRate");

        return new TrainTrainConfig
        {
            Epochs = epochs,
            BatchSize = batch,
            Shuffle = true,
            DropLast = false,
            LearningRate = lr,
            UiEveryNEpochs = 1
        };
    }

    private static TrainStartConfig DefaultTrainConfig()
    {
        return new TrainStartConfig
        {
            Network = new TrainNetworkConfig
            {
                InputSize = 2,
                OutputSize = 2,
                Hidden = new[] { 4, 4 },
                Activation = "ReLu",
                Seed = 123
            },
            Train = new TrainTrainConfig
            {
                Epochs = 50,
                BatchSize = 8,
                Shuffle = true,
                DropLast = false,
                LearningRate = 0.05,
                UiEveryNEpochs = 1
            },
            Data = new TrainDataConfig
            {
                Preset = "XOR"
            }
        };
    }

    private async Task OpenFileAsync(string fullPath)
    {
        var existing = EditorTabs.FirstOrDefault(t => string.Equals(t.FullPath, fullPath, StringComparison.OrdinalIgnoreCase));
        if (existing != null)
        {
            SelectedTab = existing;
            return;
        }

        var text = await _files.ReadTextAsync(fullPath);
        var tab = new EditorTab(Path.GetFileName(fullPath), fullPath, text);
        EditorTabs.Add(tab);
        SelectedTab = tab;
        BuildJsonTree(text);
        await LoadGraphAsync();
        await LoadStructureAsync();
        _logger.Append("info", $"Файл открыт: {Path.GetFileName(fullPath)}");
    }

    private void BuildProjectTree(StudioProject project)
    {
        ProjectTree.Clear();
        var root = new ExplorerNode(project.Name);
        foreach (var file in project.Files.OrderBy(f => f.RelativePath))
        {
            root.Children.Add(new ExplorerNode(file.DisplayName, file.FullPath));
        }
        ProjectTree.Add(root);
    }

    private string GetRelative(string fullPath)
    {
        return _currentProject == null ? fullPath : Path.GetRelativePath(_currentProject.RootPath, fullPath).Replace("\\", "/");
    }

    private void CloseTabByPath(string fullPath)
    {
        var tab = EditorTabs.FirstOrDefault(t => string.Equals(t.FullPath, fullPath, StringComparison.OrdinalIgnoreCase));
        if (tab != null)
            CloseTab(tab);
    }

    private void UpdateOpenTabPath(string oldRelPath, string newRelPath)
    {
        if (_currentProject == null)
            return;

        var oldFull = Path.Combine(_currentProject.RootPath, oldRelPath);
        var tab = EditorTabs.FirstOrDefault(t => string.Equals(t.FullPath, oldFull, StringComparison.OrdinalIgnoreCase));
        if (tab == null)
            return;

        tab.FullPath = Path.Combine(_currentProject.RootPath, newRelPath);
        tab.Title = Path.GetFileName(newRelPath);
    }

    private static bool IsRequiredFile(string relPath)
    {
        var name = Path.GetFileName(relPath).ToLowerInvariant();
        return name == "dataset.csv" || name == "network.json" || name == "params.json";
    }

    private async Task LoadParamsAsync()
    {
        if (_currentProject == null)
            return;

        var path = Path.Combine(_currentProject.RootPath, "params.json");
        if (!File.Exists(path))
            return;

        try
        {
            var json = await File.ReadAllTextAsync(path);
            if (JsonDocument.Parse(json).RootElement is { } root)
            {
                ParamEpochs = ReadNumberAsString(root, "epochs", ParamEpochs);
                ParamLearningRate = ReadNumberAsString(root, "learningRate", ParamLearningRate);
                ParamBatchSize = ReadNumberAsString(root, "batchSize", ParamBatchSize);
                if (root.TryGetProperty("optimizer", out var opt) && opt.ValueKind == JsonValueKind.String)
                    ParamOptimizer = opt.GetString() ?? ParamOptimizer;
            }
        }
        catch (Exception ex)
        {
            _logger.Append("warn", $"Не удалось прочитать params.json: {ex.Message}");
        }
    }

    private async Task LoadGraphAsync()
    {
        GraphItems.Clear();
        if (_currentProject == null)
            return;

        var path = Path.Combine(_currentProject.RootPath, "network.json");
        if (!File.Exists(path))
            return;

        var json = await File.ReadAllTextAsync(path);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var input = root.TryGetProperty("inputSize", out var inp) ? inp.GetInt32() : 0;
        var output = root.TryGetProperty("outputSize", out var outp) ? outp.GetInt32() : 0;
        GraphItems.Add($"Вход: {input}");
        if (root.TryGetProperty("hidden", out var hidden) && hidden.ValueKind == JsonValueKind.Array)
        {
            int idx = 1;
            foreach (var h in hidden.EnumerateArray())
            {
                GraphItems.Add($"Слой {idx}: {h.GetInt32()} нейронов");
                idx++;
            }
        }
        GraphItems.Add($"Выход: {output}");
    }

    private async Task LoadStructureAsync()
    {
        StructureItems.Clear();
        StructurePreview = "";
        StructureGraph = GraphModel.Empty;
        SelectedGraphNodeId = null;
        if (_currentProject == null)
            return;

        var path = Path.Combine(_currentProject.RootPath, "network.json");
        if (!File.Exists(path))
        {
            StructureItems.Add("network.json не найден");
            return;
        }

        var json = await File.ReadAllTextAsync(path);
        StructurePreview = json;

        var graph = NetworkGraphBuilder.Build(json);
        if (graph.Nodes.Count == 0 && !string.IsNullOrWhiteSpace(json))
        {
            _logger.Append("error", "Некорректный network.json");
            StructureItems.Add("Некорректный network.json");
            StructureGraph = GraphModel.Empty;
            return;
        }

        StructureGraph = graph;
        foreach (var node in graph.Nodes)
            StructureItems.Add(node.Title);
    }

    public async Task ExportMetricsCsvAsync(string fullPath)
    {
        try
        {
            var lines = new List<string> { "epoch,loss,valLoss,accuracy,utc" };
            foreach (var m in Metrics)
            {
                var acc = m.Accuracy.HasValue ? m.Accuracy.Value.ToString("F6") : "";
                var val = m.ValLoss.HasValue ? m.ValLoss.Value.ToString("F6") : "";
                lines.Add($"{m.Epoch},{m.Loss:F6},{val},{acc},{m.Utc:O}");
            }

            await File.WriteAllLinesAsync(fullPath, lines);
            _logger.Append("info", $"Экспорт метрик: {fullPath}");
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка экспорта метрик: {ex.Message}");
        }
    }

    private void BuildJsonTree(string text)
    {
        JsonTree.Clear();
        try
        {
            using var doc = JsonDocument.Parse(text);
            var root = BuildJsonNode("root", doc.RootElement);
            JsonTree.Add(root);
        }
        catch
        {
        }
    }

    private static ExplorerNode BuildJsonNode(string name, JsonElement element)
    {
        var node = new ExplorerNode(name);
        switch (element.ValueKind)
        {
            case JsonValueKind.Object:
                foreach (var prop in element.EnumerateObject())
                    node.Children.Add(BuildJsonNode(prop.Name, prop.Value));
                break;
            case JsonValueKind.Array:
                int index = 0;
                foreach (var item in element.EnumerateArray())
                {
                    node.Children.Add(BuildJsonNode($"[{index}]", item));
                    index++;
                }
                break;
            default:
                node.Children.Add(new ExplorerNode(element.ToString() ?? string.Empty));
                break;
        }
        return node;
    }

    private void OnDisconnected()
    {
        Dispatcher.UIThread.Post(() =>
        {
            Status = "Отключено";
            IsConnected = false;
            ConnectionState = StudioConnectionState.Disconnected;
            TrainingStatus = TrainingState.Idle;
            _activeRunId = null;
            OnPropertyChanged(nameof(ShortRunId));
            OnPropertyChanged(nameof(HasRunId));
            OnPropertyChanged(nameof(StatusLine));
            _logger.Append("warn", "Соединение с хостом разорвано.");
        });
    }

    private void OnEventReceived(RpcEvent evt)
    {
        switch (evt.type)
        {
            case "log.append":
                HandleLog(evt.data);
                break;
            case "metrics.update":
                HandleMetrics(evt.data);
                break;
            case "train.stateChanged":
                HandleTrainState(evt.data);
                break;
        }
    }

    private void HandleLog(object? data)
    {
        if (data is not JsonElement el || el.ValueKind != JsonValueKind.Object)
            return;

        var level = GetString(el, "level") ?? "info";
        var message = GetString(el, "message") ?? "";
        var utc = GetDateTime(el, "utc");
        string prefix = utc.HasValue ? utc.Value.ToLocalTime().ToString("HH:mm:ss") : "--:--:--";
        _logger.Append(level, $"{prefix} {message}");

        if (message.StartsWith("model.saved:", StringComparison.OrdinalIgnoreCase) ||
            message.StartsWith("model.loaded:", StringComparison.OrdinalIgnoreCase))
        {
            Dispatcher.UIThread.Post(async () => await RefreshProjectFilesAsync());
        }
    }

    private void HandleMetrics(object? data)
    {
        if (data is not JsonElement el || el.ValueKind != JsonValueKind.Object)
            return;

        var runId = GetString(el, "runId") ?? "";
        int epoch = GetInt(el, "epoch");
        double loss = GetDoubleAny(el, "loss", "trainLoss");
        double? valLoss = GetNullableDouble(el, "valLoss");
        double? acc = GetNullableDouble(el, "accuracy");
        var utc = GetDateTime(el, "utc") ?? DateTimeOffset.UtcNow;

        if (!ShouldEmitMetric(epoch))
            return;

        if (valLoss.HasValue)
            _lastValLoss = valLoss;
        if (acc.HasValue)
            _lastAcc = acc;

        var point = new MetricPoint(
            epoch,
            loss,
            valLoss ?? _lastValLoss,
            acc ?? _lastAcc,
            utc);
        _metricsQueue.Enqueue(new PendingMetric(runId, point));
    }

    private void HandleTrainState(object? data)
    {
        if (data is not JsonElement el || el.ValueKind != JsonValueKind.Object)
            return;

        var runId = GetString(el, "runId") ?? "";
        var state = GetString(el, "state") ?? "";
        var message = GetString(el, "message") ?? "";
        var utc = GetDateTime(el, "utc");
        string prefix = utc.HasValue ? utc.Value.ToLocalTime().ToString("HH:mm:ss") : "--:--:--";
        var details = string.IsNullOrWhiteSpace(message) ? "" : $" ({message})";
        _logger.Append("info", $"{prefix} state={state} runId={runId}{details}");

        TrainingStatus = state switch
        {
            "running" => TrainingState.Running,
            "finished" => TrainingState.Finished,
            "stopped" => TrainingState.Stopped,
            "failed" => TrainingState.Failed,
            _ => TrainingStatus
        };
        OnPropertyChanged(nameof(StatusLine));
        OnPropertyChanged(nameof(ShortRunId));
        OnPropertyChanged(nameof(HasRunId));

        if (_runStartedUtc.HasValue && (state == "finished" || state == "stopped" || state == "failed"))
        {
            var duration = DateTimeOffset.Now - _runStartedUtc.Value;
            _logger.Append("info", $"Длительность: {duration:g}");
            _runStartedUtc = null;
        }

        if (!string.IsNullOrWhiteSpace(runId))
        {
            AddRunId(runId);
            _ = RefreshRunListAsync();
        }
    }

    public async Task RefreshProjectFilesAsync()
    {
        if (_currentProject == null)
            return;

        var files = Directory.EnumerateFiles(_currentProject.RootPath, "*", SearchOption.TopDirectoryOnly)
            .Select(path =>
            {
                var rel = Path.GetFileName(path);
                var kind = WorkspaceService.GetDefaultKindFor(rel);
                return WorkspaceService.ToItem(_currentProject.RootPath, rel, kind);
            })
            .ToArray();

        _currentProject = _currentProject with { Files = files };
        await _workspace.SaveProjectAsync(_currentProject);
        BuildProjectTree(_currentProject);
        RaiseComputedState();
    }

    private void FlushMetrics()
    {
        if (_metricsQueue.IsEmpty)
            return;

        Dispatcher.UIThread.Post(() =>
        {
            while (_metricsQueue.TryDequeue(out var pending))
                Metrics.Add(pending.Point);

            while (Metrics.Count > 2000)
                Metrics.RemoveAt(0);
        });
    }

    private void ResetCompareState()
    {
        CompareMode = false;
        RunIds.Clear();
        SelectedRunA = null;
        SelectedRunB = null;
        CompareItemsA = null;
        CompareItemsB = null;
        CompareShowTrain = true;
        CompareShowVal = true;
        CompareShowAcc = true;
    }

    private void AddRunId(string runId)
    {
        if (RunIds.Contains(runId))
            return;
        RunIds.Insert(0, runId);
    }

    private async Task RefreshRunListAsync()
    {
        RunIds.Clear();
        if (!IsConnected || _currentProject == null)
            return;

        try
        {
            var projectId = await TryGetRemoteProjectIdAsync();
            if (string.IsNullOrWhiteSpace(projectId))
                return;

            var runs = await _client.CallAsync<RunDto[]>("run.list", new { projectId });
            var ordered = runs
                .OrderByDescending(r => r.startedUtc ?? DateTime.MinValue)
                .ToArray();
            foreach (var run in ordered)
                RunIds.Add(run.runId);

            if (ordered.Length >= 2)
            {
                SelectedRunA = ordered[0].runId;
                SelectedRunB = ordered[1].runId;
            }
            else if (ordered.Length == 1)
            {
                SelectedRunA = ordered[0].runId;
                SelectedRunB = null;
            }
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка загрузки запусков: {ex.Message}");
        }
    }

    private async Task<string?> TryGetRemoteProjectIdAsync()
    {
        if (_currentProject == null)
            return null;

        var projects = await _client.CallAsync<ProjectDto[]>("project.list", null);
        var project = projects.FirstOrDefault(p => string.Equals(p.name, _currentProject.Name, StringComparison.OrdinalIgnoreCase));
        return project?.projectId;
    }

    public async Task CompareRunsAsync()
    {
        if (string.IsNullOrWhiteSpace(SelectedRunA) || string.IsNullOrWhiteSpace(SelectedRunB))
        {
            _logger.Append("warn", "Select two runs.");
            return;
        }
        if (SelectedRunA == SelectedRunB)
        {
            _logger.Append("warn", "Run A and Run B must be different.");
            return;
        }

        try
        {
            var a = await LoadMetricsForRunAsync(SelectedRunA);
            var b = await LoadMetricsForRunAsync(SelectedRunB);

            if (a.Count == 0 || b.Count == 0)
            {
                _logger.Append("warn", "Нет метрик для сравнения.");
                return;
            }

            int maxA = a.Max(m => m.Epoch);
            int maxB = b.Max(m => m.Epoch);
            int maxEpoch = Math.Min(maxA, maxB);
            if (maxA != maxB)
                _logger.Append("info", $"Сравнение по min(epoch)={maxEpoch}.");

            CompareItemsA = a.Where(m => m.Epoch <= maxEpoch).OrderBy(m => m.Epoch).ToArray();
            CompareItemsB = b.Where(m => m.Epoch <= maxEpoch).OrderBy(m => m.Epoch).ToArray();
            CompareMode = true;
        }
        catch (Exception ex)
        {
            _logger.Append("error", $"Ошибка сравнения: {ex.Message}");
        }
    }

    public void ClearCompare()
    {
        CompareMode = false;
        CompareItemsA = null;
        CompareItemsB = null;
    }

    private async Task<IList<MetricPoint>> LoadMetricsForRunAsync(string runId)
    {
        if (IsConnected)
        {
            var points = await _client.CallAsync<RunMetricsPointDto[]>("run.metrics", new { runId });
            return points.Select(p => new MetricPoint(
                p.epoch,
                p.loss,
                valLoss: null,
                accuracy: p.accuracy,
                utc: p.utc)).ToList();
        }

        if (_currentProject == null)
            return Array.Empty<MetricPoint>();

        var root = _currentProject.RootPath;
        var pathByRun = Path.Combine(root, $"metrics-{runId}.csv");
        var path = File.Exists(pathByRun) ? pathByRun : Path.Combine(root, "metrics.csv");
        if (!File.Exists(path))
            return Array.Empty<MetricPoint>();

        return await ParseMetricsCsvAsync(path);
    }

    private static async Task<IList<MetricPoint>> ParseMetricsCsvAsync(string path)
    {
        var list = new List<MetricPoint>();
        var lines = await File.ReadAllLinesAsync(path);
        if (lines.Length == 0)
            return list;

        int start = 0;
        var header = lines[0].Split(',', StringSplitOptions.TrimEntries);
        int idxEpoch = Array.FindIndex(header, h => h.Equals("epoch", StringComparison.OrdinalIgnoreCase));
        int idxLoss = Array.FindIndex(header, h => h.Equals("trainLoss", StringComparison.OrdinalIgnoreCase) || h.Equals("loss", StringComparison.OrdinalIgnoreCase));
        int idxVal = Array.FindIndex(header, h => h.Equals("valLoss", StringComparison.OrdinalIgnoreCase));
        int idxAcc = Array.FindIndex(header, h => h.Equals("accuracy", StringComparison.OrdinalIgnoreCase) || h.Equals("acc", StringComparison.OrdinalIgnoreCase));
        bool hasHeader = idxEpoch >= 0 || idxLoss >= 0 || idxVal >= 0 || idxAcc >= 0;
        if (hasHeader)
            start = 1;
        else
        {
            idxEpoch = 0;
            idxLoss = 1;
            idxVal = 2;
            idxAcc = 3;
        }

        for (int i = start; i < lines.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i]))
                continue;
            var parts = lines[i].Split(',', StringSplitOptions.TrimEntries);
            if (idxEpoch >= parts.Length || idxLoss >= parts.Length)
                continue;

            if (!int.TryParse(parts[idxEpoch], NumberStyles.Integer, CultureInfo.InvariantCulture, out var epoch))
                continue;
            if (!TryParseDouble(parts[idxLoss], out var loss))
                continue;

            double? val = null;
            double? acc = null;
            if (idxVal >= 0 && idxVal < parts.Length && TryParseDouble(parts[idxVal], out var v))
                val = v;
            if (idxAcc >= 0 && idxAcc < parts.Length && TryParseDouble(parts[idxAcc], out var a))
                acc = a;

            list.Add(new MetricPoint(epoch, loss, val, acc, DateTimeOffset.UtcNow));
        }

        return list;
    }

    private static string? GetString(JsonElement el, string name)
    {
        return el.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.String
            ? prop.GetString()
            : null;
    }

    private static int GetInt(JsonElement el, string name)
    {
        return el.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.Number && prop.TryGetInt32(out var value)
            ? value
            : 0;
    }

    private static double GetDouble(JsonElement el, string name)
    {
        return el.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.Number
            ? prop.GetDouble()
            : 0;
    }

    private static double GetDoubleAny(JsonElement el, string primary, string fallback)
    {
        if (el.TryGetProperty(primary, out var prop) && prop.ValueKind == JsonValueKind.Number)
            return prop.GetDouble();
        if (el.TryGetProperty(fallback, out var alt) && alt.ValueKind == JsonValueKind.Number)
            return alt.GetDouble();
        return 0;
    }

    private static double? GetNullableDouble(JsonElement el, string name)
    {
        return el.TryGetProperty(name, out var prop) && prop.ValueKind == JsonValueKind.Number
            ? prop.GetDouble()
            : null;
    }

    private int TryParseInt(string value, int fallback, string field)
    {
        if (int.TryParse(value.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
            return parsed;
        _logger.Append("warn", $"Неверное значение {field}: {value}. Использовано {fallback}.");
        return fallback;
    }

    private double TryParseDoubleFlexible(string value, double fallback, string field)
    {
        var normalized = value.Replace(',', '.').Trim();
        if (double.TryParse(normalized, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed))
            return parsed;
        _logger.Append("warn", $"Неверное значение {field}: {value}. Использовано {fallback}.");
        return fallback;
    }

    public void ResetParams()
    {
        ParamEpochs = "50";
        ParamLearningRate = "0.05";
        ParamBatchSize = "8";
        ParamOptimizer = "adam";
        UseFileParams = true;
        MetricsEveryText = "1";
        ResumeTraining = false;
        _logger.Append("info", "Параметры сброшены.");
    }

    private void ValidateEpochs()
    {
        bool valid = int.TryParse(_paramEpochs.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) && value >= 1;
        if (SetField(ref _epochsValid, valid, nameof(_epochsValid)))
        {
            OnPropertyChanged(nameof(EpochsInvalid));
            if (!valid && !_loggedEpochsInvalid)
            {
                _logger.Append("warn", "epochs должно быть целым числом >= 1.");
                _loggedEpochsInvalid = true;
            }
            if (valid)
                _loggedEpochsInvalid = false;
        }
    }

    private void ValidateLearningRate()
    {
        var normalized = _paramLearningRate.Replace(',', '.');
        bool valid = double.TryParse(normalized.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var value) && value > 0;
        if (SetField(ref _learningRateValid, valid, nameof(_learningRateValid)))
        {
            OnPropertyChanged(nameof(LearningRateInvalid));
            if (!valid && !_loggedLearningRateInvalid)
            {
                _logger.Append("warn", "learningRate должно быть числом > 0.");
                _loggedLearningRateInvalid = true;
            }
            if (valid)
                _loggedLearningRateInvalid = false;
        }
    }

    private void ValidateBatchSize()
    {
        bool valid = int.TryParse(_paramBatchSize.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) && value >= 1;
        if (SetField(ref _batchSizeValid, valid, nameof(_batchSizeValid)))
        {
            OnPropertyChanged(nameof(BatchSizeInvalid));
            if (!valid && !_loggedBatchSizeInvalid)
            {
                _logger.Append("warn", "batchSize должно быть целым числом >= 1.");
                _loggedBatchSizeInvalid = true;
            }
            if (valid)
                _loggedBatchSizeInvalid = false;
        }
    }

    private void ValidateOptimizer()
    {
        if (!OptimizerOptions.Contains(_paramOptimizer))
        {
            _logger.Append("warn", $"Оптимизатор '{_paramOptimizer}' может не поддерживаться хостом.");
        }
        else if (_paramOptimizer != "adam")
        {
            _logger.Append("warn", $"Оптимизатор '{_paramOptimizer}' выбран. Поддержка хостом не подтверждена.");
        }
    }

    private bool EnsureParamsValid()
    {
        ValidateEpochs();
        ValidateLearningRate();
        ValidateBatchSize();
        return _epochsValid && _learningRateValid && _batchSizeValid;
    }

    private static string ReadNumberAsString(JsonElement root, string name, string fallback)
    {
        if (!root.TryGetProperty(name, out var prop))
            return fallback;

        return prop.ValueKind switch
        {
            JsonValueKind.Number => prop.GetDouble().ToString(CultureInfo.InvariantCulture),
            JsonValueKind.String => prop.GetString() ?? fallback,
            _ => fallback
        };
    }

    private static double[] ParseInputVector(string raw)
    {
        var parts = raw.Split(new[] { ',', ' ', ';', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0)
            throw new InvalidOperationException("Input is empty.");

        var values = new List<double>();
        foreach (var part in parts)
        {
            var normalized = part.Replace(',', '.').Trim();
            if (!double.TryParse(normalized, NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                throw new InvalidOperationException($"Invalid number: {part}");
            values.Add(v);
        }

        return values.ToArray();
    }

    private sealed record DatasetSchema(bool HasHeader, int[] InputCols, int LabelCol);

    private static DatasetSchema DetectDatasetSchema(string path)
    {
        if (!File.Exists(path))
            throw new InvalidOperationException("dataset.csv не найден.");

        string? first = null;
        string? second = null;
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;
            if (first == null)
                first = line;
            else
            {
                second = line;
                break;
            }
        }

        if (first == null)
            throw new InvalidOperationException("dataset.csv пуст.");

        bool firstIsNumeric = LineIsNumeric(first);
        bool hasHeader = !firstIsNumeric;
        var dataLine = hasHeader && second != null ? second : first;
        var cols = dataLine.Split(',', StringSplitOptions.TrimEntries);
        if (cols.Length < 2)
            throw new InvalidOperationException("dataset.csv должен содержать минимум 2 колонки.");

        int labelCol = cols.Length - 1;
        var inputCols = Enumerable.Range(0, cols.Length - 1).ToArray();
        return new DatasetSchema(hasHeader, inputCols, labelCol);
    }

    private static bool LineIsNumeric(string line)
    {
        var parts = line.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length == 0)
            return false;
        foreach (var part in parts)
        {
            if (!double.TryParse(part, NumberStyles.Float, CultureInfo.InvariantCulture, out _)
                && !double.TryParse(part, NumberStyles.Float, CultureInfo.CurrentCulture, out _))
                return false;
        }
        return true;
    }

    private sealed record Range2D(double MinX, double MaxX, double MinY, double MaxY);

    private static List<(double x, double y, int label)> LoadDatasetPoints(string path, DatasetSchema schema)
    {
        var points = new List<(double x, double y, int label)>();
        bool first = true;
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;
            if (first && schema.HasHeader)
            {
                first = false;
                continue;
            }
            first = false;

            var parts = line.Split(',', StringSplitOptions.TrimEntries);
            if (parts.Length == 0)
                continue;

            if (schema.InputCols.Length < 2)
                continue;

            if (!TryParseDouble(parts[schema.InputCols[0]], out var x))
                continue;
            if (!TryParseDouble(parts[schema.InputCols[1]], out var y))
                continue;

            int label = 0;
            if (schema.LabelCol >= 0 && schema.LabelCol < parts.Length)
                int.TryParse(parts[schema.LabelCol], NumberStyles.Integer, CultureInfo.InvariantCulture, out label);

            points.Add((x, y, label));
        }

        return points;
    }

    private static Range2D GetRange(List<(double x, double y, int label)> points)
    {
        if (points.Count == 0)
            return new Range2D(0, 1, 0, 1);

        double minX = points.Min(p => p.x);
        double maxX = points.Max(p => p.x);
        double minY = points.Min(p => p.y);
        double maxY = points.Max(p => p.y);
        if (Math.Abs(maxX - minX) < 1e-9)
        {
            minX -= 0.5;
            maxX += 0.5;
        }
        if (Math.Abs(maxY - minY) < 1e-9)
        {
            minY -= 0.5;
            maxY += 0.5;
        }
        return new Range2D(minX, maxX, minY, maxY);
    }

    private Range2D GetManualRange()
    {
        if (!TryParseDouble(RangeXMin, out var minX) ||
            !TryParseDouble(RangeXMax, out var maxX) ||
            !TryParseDouble(RangeYMin, out var minY) ||
            !TryParseDouble(RangeYMax, out var maxY))
            throw new InvalidOperationException("Некорректный диапазон осей.");

        if (maxX <= minX || maxY <= minY)
            throw new InvalidOperationException("Диапазон осей должен быть min < max.");

        return new Range2D(minX, maxX, minY, maxY);
    }

    private static double[][] BuildGridInputs(int gridSize, Range2D range)
    {
        var inputs = new double[gridSize * gridSize][];
        int idx = 0;
        double dx = (range.MaxX - range.MinX) / (gridSize - 1);
        double dy = (range.MaxY - range.MinY) / (gridSize - 1);

        for (int y = 0; y < gridSize; y++)
        {
            double yy = range.MaxY - dy * y;
            for (int x = 0; x < gridSize; x++)
            {
                double xx = range.MinX + dx * x;
                inputs[idx++] = new[] { xx, yy };
            }
        }

        return inputs;
    }

    private static string BuildHeatmapKey(string modelId, int gridSize, Range2D range, double threshold, bool useProbability)
    {
        return $"{modelId}:{gridSize}:{(useProbability ? "prob" : "argmax")}:{threshold:0.###}:{range.MinX:0.####}:{range.MaxX:0.####}:{range.MinY:0.####}:{range.MaxY:0.####}";
    }

    private static Bitmap RenderHeatmap(
        int gridSize,
        Range2D range,
        InferBatchResponse response,
        List<(double x, double y, int label)> points,
        double threshold,
        bool useProbability)
    {
        var bmp = new WriteableBitmap(
            new PixelSize(gridSize, gridSize),
            new Vector(96, 96),
            PixelFormat.Bgra8888,
            AlphaFormat.Premul);

        using var fb = bmp.Lock();
        int stride = fb.RowBytes;
        int height = fb.Size.Height;
        int width = fb.Size.Width;
        var buffer = new byte[stride * height];

        for (int i = 0; i < response.Outputs.Length; i++)
        {
            int x = i % width;
            int y = i / width;
            if (y >= height)
                break;

            double prob1 = 0;
            int predicted = 0;
            if (response.Probabilities != null && response.Probabilities.Length > i && response.Probabilities[i].Length > 1)
            {
                prob1 = response.Probabilities[i][1];
                predicted = ArgMax(response.Probabilities[i]);
            }
            else if (response.PredictedIndices != null && response.PredictedIndices.Length > i)
            {
                predicted = response.PredictedIndices[i];
            }

            byte r;
            byte g;
            byte b;
            if (useProbability)
            {
                var value = prob1;
                r = (byte)Math.Clamp(value * 255, 0, 255);
                b = (byte)Math.Clamp((1.0 - value) * 255, 0, 255);
                g = (byte)0;

                bool isBoundary = Math.Abs(value - threshold) <= 0.03;
                if (isBoundary)
                {
                    r = 255;
                    g = 255;
                    b = 255;
                }
            }
            else
            {
                (r, g, b) = GetClassColor(predicted);
            }

            int offset = y * stride + x * 4;
            buffer[offset + 0] = b;
            buffer[offset + 1] = g;
            buffer[offset + 2] = r;
            buffer[offset + 3] = 255;
        }

        foreach (var (x, y, label) in points)
        {
            int px = (int)Math.Round((x - range.MinX) / (range.MaxX - range.MinX) * (width - 1));
            int py = (int)Math.Round((range.MaxY - y) / (range.MaxY - range.MinY) * (height - 1));
            DrawPoint(buffer, stride, width, height, px, py, label == 1);
        }

        Marshal.Copy(buffer, 0, fb.Address, buffer.Length);
        return bmp;
    }

    private static void DrawPoint(byte[] buffer, int stride, int width, int height, int x, int y, bool positive)
    {
        int radius = 3;
        var fill = positive
            ? (r: (byte)230, g: (byte)60, b: (byte)60)
            : (r: (byte)46, g: (byte)91, b: (byte)255);
        var outline = positive
            ? (r: (byte)255, g: (byte)255, b: (byte)255)
            : (r: (byte)0, g: (byte)0, b: (byte)0);

        if (positive)
        {
            DrawCircle(buffer, stride, width, height, x, y, radius + 1, outline);
            DrawCircle(buffer, stride, width, height, x, y, radius, fill);
        }
        else
        {
            DrawCross(buffer, stride, width, height, x, y, radius + 1, outline);
            DrawCross(buffer, stride, width, height, x, y, radius, fill);
        }
    }

    private static (byte r, byte g, byte b) GetClassColor(int index)
    {
        return index switch
        {
            0 => ((byte)46, (byte)91, (byte)255),
            1 => ((byte)230, (byte)60, (byte)60),
            2 => ((byte)0, (byte)200, (byte)120),
            3 => ((byte)255, (byte)165, (byte)0),
            4 => ((byte)160, (byte)90, (byte)255),
            _ => ((byte)200, (byte)200, (byte)200)
        };
    }

    private static int ArgMax(double[] values)
    {
        if (values.Length == 0)
            return 0;
        int idx = 0;
        double max = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > max)
            {
                max = values[i];
                idx = i;
            }
        }
        return idx;
    }

    private static void DrawCircle(byte[] buffer, int stride, int width, int height, int x, int y, int radius, (byte r, byte g, byte b) color)
    {
        int r2 = radius * radius;
        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                if (dx * dx + dy * dy > r2)
                    continue;
                WritePixel(buffer, stride, width, height, x + dx, y + dy, color);
            }
        }
    }

    private static void DrawCross(byte[] buffer, int stride, int width, int height, int x, int y, int radius, (byte r, byte g, byte b) color)
    {
        for (int d = -radius; d <= radius; d++)
        {
            WritePixel(buffer, stride, width, height, x + d, y + d, color);
            WritePixel(buffer, stride, width, height, x + d, y - d, color);
        }
    }

    private static void WritePixel(byte[] buffer, int stride, int width, int height, int x, int y, (byte r, byte g, byte b) color)
    {
        if (x < 0 || y < 0 || x >= width || y >= height)
            return;
        int offset = y * stride + x * 4;
        buffer[offset + 0] = color.b;
        buffer[offset + 1] = color.g;
        buffer[offset + 2] = color.r;
        buffer[offset + 3] = 255;
    }

    private static bool TryParseColumns(string? raw, out int[] cols)
    {
        cols = Array.Empty<int>();
        if (string.IsNullOrWhiteSpace(raw))
            return false;

        var parts = raw.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries);
        var values = new List<int>();
        foreach (var part in parts)
        {
            if (!int.TryParse(part, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v))
                return false;
            if (v < 0)
                return false;
            values.Add(v);
        }

        if (values.Count == 0)
            return false;

        cols = values.ToArray();
        return true;
    }

    private static bool TryParseColumnIndex(string? raw, out int col)
    {
        col = 0;
        if (string.IsNullOrWhiteSpace(raw))
            return false;
        if (!int.TryParse(raw.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out col))
            return false;
        return col >= 0;
    }

    private static bool IsNetworkFile(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return false;
        return Path.GetFileName(path).Equals("network.json", StringComparison.OrdinalIgnoreCase);
    }

    private void ScheduleStructureReload()
    {
        _structureDebounce.Stop();
        _structureDebounce.Start();
    }

    private void UpdateSelectedLayer()
    {
        _selectedLayerMeta.Clear();
        if (StructureGraph == null || SelectedGraphNodeId == null)
        {
            SelectedLayerTitle = "";
            SelectedLayerSubtitle = "";
            return;
        }

        var node = StructureGraph.Nodes.FirstOrDefault(n => n.Id == SelectedGraphNodeId.Value);
        if (node == null)
        {
            SelectedLayerTitle = "";
            SelectedLayerSubtitle = "";
            return;
        }

        SelectedLayerTitle = node.Title;
        SelectedLayerSubtitle = node.Subtitle;
        foreach (var pair in node.Meta)
            _selectedLayerMeta.Add($"{pair.Key}: {pair.Value}");
    }

    private void InvalidateHeatmapCacheIfNeeded(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return;

        var file = Path.GetFileName(path);
        if (file.Equals("dataset.csv", StringComparison.OrdinalIgnoreCase) ||
            file.Equals("network.json", StringComparison.OrdinalIgnoreCase))
        {
            InvalidateHeatmapCache(file);
        }
    }

    private void InvalidateHeatmapCache(string reason)
    {
        _heatmapCache.Clear();
        DecisionBoundaryImage = null;
        HeatmapStatus = $"Heatmap сброшен ({reason}).";
    }

    public string LoadedModelShortId => string.IsNullOrWhiteSpace(_loadedModelId)
        ? "-"
        : _loadedModelId.Length <= 8
            ? _loadedModelId
            : _loadedModelId[..8];

    public string LoadedModelInfo
    {
        get
        {
            if (string.IsNullOrWhiteSpace(_loadedModelId))
                return "modelId: -";
            var inputSize = _loadedInputSize?.ToString(CultureInfo.InvariantCulture) ?? "-";
            var outputSize = _loadedOutputSize?.ToString(CultureInfo.InvariantCulture) ?? "-";
            return $"modelId: {LoadedModelShortId} | in={inputSize} out={outputSize}";
        }
    }

    public bool HasInferInputs => InferInputs.Count > 0;
    public bool ShowInferCsv => !HasInferInputs;

    private void BuildInferInputs()
    {
        InferInputs.Clear();
        if (!_loadedInputSize.HasValue || _loadedInputSize.Value <= 0)
        {
            OnPropertyChanged(nameof(HasInferInputs));
            OnPropertyChanged(nameof(ShowInferCsv));
            return;
        }

        for (int i = 0; i < _loadedInputSize.Value; i++)
            InferInputs.Add(new InferInputField(i, "0"));
        OnPropertyChanged(nameof(HasInferInputs));
        OnPropertyChanged(nameof(ShowInferCsv));
    }

    private double[] BuildInferInputVector()
    {
        if (_loadedInputSize.HasValue && _loadedInputSize.Value > 0 && InferInputs.Count == _loadedInputSize.Value)
        {
            var values = new double[_loadedInputSize.Value];
            for (int i = 0; i < InferInputs.Count; i++)
            {
                if (!TryParseDouble(InferInputs[i].Value, out var v))
                    throw new InvalidOperationException($"Некорректный ввод x{i}.");
                values[i] = v;
            }
            return values;
        }

        return ParseInputVector(InferInput);
    }

    private static bool TryParseDouble(string? raw, out double value)
    {
        value = 0;
        if (string.IsNullOrWhiteSpace(raw))
            return false;

        var normalized = raw.Replace(',', '.').Trim();
        return double.TryParse(normalized, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    private static string FormatTopProbabilities(double[]? probabilities)
    {
        if (probabilities == null || probabilities.Length == 0)
            return "-";

        var indexed = probabilities
            .Select((v, i) => new { Index = i, Value = v })
            .OrderByDescending(x => x.Value)
            .Take(3)
            .Select(x => $"{x.Index}:{x.Value:0.####}");
        return string.Join("  ", indexed);
    }

    private void AddInferHistory(InferHistoryItem item)
    {
        InferHistory.Insert(0, item);
        while (InferHistory.Count > 20)
            InferHistory.RemoveAt(InferHistory.Count - 1);
    }

    private void UpdateSoftmaxBars(double[]? probabilities)
    {
        SoftmaxBars.Clear();
        if (probabilities == null || probabilities.Length == 0)
        {
            OnPropertyChanged(nameof(ShowSoftmaxBars));
            return;
        }

        for (int i = 0; i < probabilities.Length; i++)
            SoftmaxBars.Add(new SoftmaxBarItem(i, probabilities[i]));

        OnPropertyChanged(nameof(ShowSoftmaxBars));
    }

    private void ValidateMetricsEvery()
    {
        bool valid = int.TryParse(_metricsEveryText.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var value) && value >= 1;
        if (SetField(ref _metricsEveryValid, valid, nameof(_metricsEveryValid)))
        {
            OnPropertyChanged(nameof(MetricsEveryInvalid));
            if (!valid && !_loggedMetricsEveryInvalid)
            {
                _logger.Append("warn", "Период метрик должен быть целым числом >= 1.");
                _loggedMetricsEveryInvalid = true;
            }
            if (valid)
                _loggedMetricsEveryInvalid = false;
        }
    }

    private bool ShouldEmitMetric(int epoch)
    {
        if (!_metricsEveryValid)
            return false;
        if (!int.TryParse(_metricsEveryText.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var every))
            return false;
        if (every <= 1)
            return true;
        return epoch == 1 || epoch % every == 0;
    }

    private static DateTimeOffset? GetDateTime(JsonElement el, string name)
    {
        if (!el.TryGetProperty(name, out var prop))
            return null;
        if (prop.ValueKind == JsonValueKind.String && DateTimeOffset.TryParse(prop.GetString(), out var dto))
            return dto;
        return null;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    private void OnPropertyChanged([CallerMemberName] string? name = null)
    {
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? name = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
            return false;
        field = value;
        OnPropertyChanged(name);
        return true;
    }

    private bool RequiredFilesPresent()
    {
        if (_currentProject == null)
            return false;
        var root = _currentProject.RootPath;
        return File.Exists(Path.Combine(root, "network.json"))
               && File.Exists(Path.Combine(root, "params.json"))
               && File.Exists(Path.Combine(root, "dataset.csv"));
    }

    private void RaiseComputedState()
    {
        OnPropertyChanged(nameof(CanConnect));
        OnPropertyChanged(nameof(CanDisconnect));
        OnPropertyChanged(nameof(CanRun));
        OnPropertyChanged(nameof(CanStop));
        OnPropertyChanged(nameof(StatusLine));
    }

    private static string FormatConnectionState(StudioConnectionState state)
    {
        return state switch
        {
            StudioConnectionState.Disconnected => "Отключено",
            StudioConnectionState.Connecting => "Подключение",
            StudioConnectionState.Connected => "Подключено",
            _ => "Отключено"
        };
    }

    public sealed class InferInputField : INotifyPropertyChanged
    {
        private string _value;

        public int Index { get; }
        public string Label => $"x{Index}";
        public string Value
        {
            get => _value;
            set
            {
                if (_value == value)
                    return;
                _value = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Value)));
            }
        }

        public InferInputField(int index, string value)
        {
            Index = index;
            _value = value;
        }

        public event PropertyChangedEventHandler? PropertyChanged;
    }

    public sealed class InferHistoryItem
    {
        public string Input { get; }
        public string Predicted { get; }
        public string Display => $"{Input} -> {Predicted}";

        public InferHistoryItem(string input, string predicted)
        {
            Input = input;
            Predicted = predicted;
        }
    }

    public sealed class SoftmaxBarItem
    {
        public int Index { get; }
        public double Value { get; }
        public string Label => $"y{Index}";
        public string Display => Value.ToString("0.####", CultureInfo.InvariantCulture);

        public SoftmaxBarItem(int index, double value)
        {
            Index = index;
            Value = value;
        }
    }

    private static string FormatTrainingState(TrainingState state)
    {
        return state switch
        {
            TrainingState.Idle => "Ожидание",
            TrainingState.Running => "Выполняется",
            TrainingState.Finished => "Завершено",
            TrainingState.Stopped => "Остановлено",
            TrainingState.Failed => "Ошибка",
            _ => "Ожидание"
        };
    }
}
