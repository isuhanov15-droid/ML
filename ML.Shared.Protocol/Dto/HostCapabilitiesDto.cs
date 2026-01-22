namespace ML.Shared.Protocol;

public sealed record HostCapabilitiesDto(
    CpuInfoDto cpu,
    GpuDeviceDto[] gpuDevices,
    string[] supportedBackends);

public sealed record CpuInfoDto(int logicalCores, string? processorName);

public sealed record GpuDeviceDto(string backend, string name, int? vramMb);
