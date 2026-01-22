using ML.Core;
using ML.Core.Serialization;

namespace ML.Host.Services;

public sealed class ModelStoreService
{
    private readonly object _lock = new();
    private Network? _currentNetwork;

    public void SetCurrentNetwork(Network? network)
    {
        lock (_lock)
        {
            _currentNetwork = network;
        }
    }

    public void Save(string? modelName, string? filePath)
    {
        lock (_lock)
        {
            if (_currentNetwork == null)
                throw new InvalidOperationException("No active network to save.");

            if (!string.IsNullOrWhiteSpace(filePath))
                ModelStore.SaveToFile(filePath, _currentNetwork);
            else
                ModelStore.Save(modelName ?? throw new InvalidOperationException("modelName required"), _currentNetwork);
        }
    }

    public Network Load(string? modelName, string? filePath)
    {
        lock (_lock)
        {
            _currentNetwork = !string.IsNullOrWhiteSpace(filePath)
                ? ModelStore.LoadFromFile(filePath)
                : ModelStore.Load(modelName ?? throw new InvalidOperationException("modelName required"));

            return _currentNetwork;
        }
    }
}
