namespace ML.Shared.Protocol;

// RPC contracts (ProtocolVersion = 1):
// - project.list -> ProjectDto[]
// - project.create { name } -> ProjectDto
// - project.update { projectId, name } -> ProjectDto
// - project.delete { projectId } -> { ok }
// - project.files { projectId } -> ProjectFileDto[]
// - file.create { projectId, path, content? } -> { ok }
// - file.read { projectId, path } -> { path, content }
// - file.write { projectId, path, content } -> { ok }
// - file.rename { projectId, path, newPath } -> { ok }
// - file.delete { projectId, path } -> { ok }
// - experiment.list { projectId } -> ExperimentDto[]
// - experiment.create { projectId, name, description?, trainConfig?, computeSpec? } -> ExperimentDto
// - run.list { projectId, experimentId? } -> RunDto[]
// - run.metrics { runId } -> RunMetricsPointDto[]
public static class ProjectContracts
{
}
