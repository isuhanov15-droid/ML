using System.Runtime.InteropServices;
using ML.Shared.Protocol;

namespace ML.Host.Rpc.Handlers;

public sealed class HostHandlers
{
    public Task<object?> GetInfoAsync()
    {
        var version = typeof(HostHandlers).Assembly.GetName().Version?.ToString() ?? "dev";
        var dto = new HostInfoDto(
            appName: "ML.Host",
            hostVersion: version,
            protocolVersion: Protocol.ProtocolVersion,
            machineName: Environment.MachineName,
            osDescription: RuntimeInformation.OSDescription,
            framework: RuntimeInformation.FrameworkDescription,
            utcNow: DateTimeOffset.UtcNow);
        return Task.FromResult<object?>(dto);
    }

    public Task<object?> GetCapabilitiesAsync()
    {
        var cpu = new CpuInfoDto(Environment.ProcessorCount, processorName: null);
        var dto = new HostCapabilitiesDto(
            cpu: cpu,
            gpuDevices: Array.Empty<GpuDeviceDto>(),
            supportedBackends: new[] { "cpu" });
        return Task.FromResult<object?>(dto);
    }
}
