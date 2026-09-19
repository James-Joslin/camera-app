using System.Security.Cryptography;
using System.Text.Json;
using Azure.Storage.Blobs;

namespace CameraSoftware.Api.Inference;

public sealed record ModelRefreshStatus(DateTimeOffset? LastCheckedAt,
    DateTimeOffset? LastUpdatedAt, string? Error, bool InProgress = false)
{
    public string Mode => "manual";
}

public sealed record ReleaseArtifact(string Blob, string FileName, long Size, string Sha256);
public sealed record ModelRelease(string ReleaseId, string Status, string Identity, ReleaseArtifact[] Artifacts)
{
    public static ModelRelease Parse(JsonElement pointer, JsonElement manifest, string variant, string container)
    {
        var id = pointer.GetProperty("releaseId").GetString()!;
        var prefix = pointer.GetProperty("storage").GetProperty("releasePrefix").GetString()!;
        if (string.IsNullOrWhiteSpace(id) || prefix != $"person_detector_ssd/releases/{id}" ||
            id.Contains('/') || id.Contains('\\') || id is "." or ".." ||
            manifest.GetProperty("releaseId").GetString() != id ||
            pointer.GetProperty("storage").GetProperty("container").GetString() != container ||
            manifest.GetProperty("storage").GetProperty("container").GetString() != container ||
            manifest.GetProperty("storage").GetProperty("prefix").GetString() != prefix ||
            pointer.GetProperty("storage").GetProperty("releaseManifest").GetString() != prefix + "/release_manifest.json")
            throw new InvalidDataException("Release pointer and manifest do not agree.");
        var release = manifest.GetProperty("release");
        var status = release.GetProperty("status").GetString()!;
        if (!release.GetProperty("accepted").GetBoolean() ||
            status is not ("production" or "experimental") ||
            pointer.GetProperty("releaseStatus").GetString() != status)
            throw new InvalidDataException("Release is not accepted or its status is inconsistent.");
        var artifacts = new List<ReleaseArtifact>();
        foreach (var extension in new[] { "xml", "bin" })
        {
            var name = $"person_detector_{variant}.{extension}";
            var relative = "models/" + name;
            var blob = pointer.GetProperty("models").GetProperty(variant).GetProperty(extension).GetString()!;
            if (blob != prefix + "/" + relative) throw new InvalidDataException("Model is outside the published release.");
            var entry = manifest.GetProperty("artifacts").EnumerateArray().Single(a => a.GetProperty("path").GetString() == relative);
            var hash = entry.GetProperty("sha256").GetString()!;
            var size = entry.GetProperty("size").GetInt64();
            if (hash.Length != 64 || !hash.All(Uri.IsHexDigit) || size <= 0 || size > 512L * 1024 * 1024)
                throw new InvalidDataException("Invalid model artifact checksum or size.");
            artifacts.Add(new(blob, name, size, hash));
        }
        var identity = Convert.ToHexString(SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(
            id + status + string.Join("", artifacts.Select(a => a.Sha256)))));
        return new(id, status, identity, artifacts.ToArray());
    }

    public async Task VerifyAsync(string directory, CancellationToken token)
    {
        foreach (var artifact in Artifacts)
        {
            // Cached metadata is read from disk too: never accept arbitrary paths.
            if (Path.GetFileName(artifact.FileName) != artifact.FileName || artifact.FileName.Contains('\\'))
                throw new InvalidDataException("Invalid cached model filename.");
            await using var stream = File.OpenRead(Path.Combine(directory, artifact.FileName));
            if (stream.Length != artifact.Size || !Convert.ToHexString(await SHA256.HashDataAsync(stream, token))
                    .Equals(artifact.Sha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"Model checksum/size mismatch: {artifact.FileName}");
        }
    }
}

public sealed class ModelReleaseRefreshService(
    IConfiguration config, InferenceService inference, ILogger<ModelReleaseRefreshService> logger) : BackgroundService
{
    private string? activeIdentity;
    private readonly string cache = config["MODEL_CACHE_DIR"] ?? "/app/model-cache";
    private readonly string variant = config["MODEL_VARIANT"] ?? "int8";

    private readonly SemaphoreSlim refreshLock = new(1, 1);
    private CancellationToken stopping;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        stopping = stoppingToken;
        // Restore only the previously selected local release. Remote discovery is
        // exclusively initiated by the authenticated Settings action.
        await refreshLock.WaitAsync(stoppingToken);
        inference.Refresh = inference.Refresh with { InProgress = true };
        await Task.Yield();
        try
        {
            Directory.CreateDirectory(cache);
            await RestoreAsync(stoppingToken);
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested) { }
        catch (Exception ex) { ReportFailure(ex); }
        finally
        {
            inference.Refresh = inference.Refresh with { InProgress = false };
            refreshLock.Release();
        }
    }

    public async Task<IResult> RefreshManuallyAsync(CancellationToken token)
    {
        if (!await refreshLock.WaitAsync(0, token))
            return Results.Conflict(new { error = "A model refresh is already in progress." });
        inference.Refresh = inference.Refresh with {
            InProgress = true, LastCheckedAt = DateTimeOffset.UtcNow, Error = null
        };
        try
        {
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(token, stopping);
            timeout.CancelAfter(TimeSpan.FromMinutes(5));
            var previous = activeIdentity;
            await RefreshAsync(timeout.Token);
            inference.Refresh = inference.Refresh with { InProgress = false, Error = null };
            var updated = previous != activeIdentity;
            return Results.Ok(new {
                updated,
                message = updated ? "The latest published model is now loaded." : "The latest published model is already loaded.",
                status = inference.Status()
            });
        }
        catch (OperationCanceledException) when (token.IsCancellationRequested || stopping.IsCancellationRequested) { throw; }
        catch (Exception ex)
        {
            ReportFailure(ex);
            return Results.Json(new { error = inference.Refresh.Error }, statusCode: 503);
        }
        finally
        {
            inference.Refresh = inference.Refresh with { InProgress = false };
            refreshLock.Release();
        }
    }

    private void ReportFailure(Exception ex)
    {
        logger.LogWarning(ex, "Model refresh failed; retaining the currently available model");
        inference.Refresh = inference.Refresh with { Error = "Model refresh failed; keeping the current model. Check API logs." };
    }

    private async Task RestoreAsync(CancellationToken token)
    {
        var marker = Path.Combine(cache, "active.json");
        if (!File.Exists(marker)) return;
        var release = JsonSerializer.Deserialize<ModelRelease>(await File.ReadAllTextAsync(marker, token))
            ?? throw new InvalidDataException("Invalid cached release.");
        if (release.Identity.Length != 64 || !release.Identity.All(Uri.IsHexDigit))
            throw new InvalidDataException("Invalid cache identity.");
        var directory = Path.Combine(cache, release.Identity);
        await release.VerifyAsync(directory, token);
        await ActivateAsync(release, directory, token);
        activeIdentity = release.Identity;
    }

    private async Task ActivateAsync(ModelRelease release, string directory, CancellationToken token)
    {
        await inference.ReplaceModelAsync(Path.Combine(directory, $"person_detector_{variant}.xml"),
            release.ReleaseId, release.Status, token);
        inference.Refresh = inference.Refresh with { LastUpdatedAt = DateTimeOffset.UtcNow };
        logger.LogInformation("Activated model release {ReleaseId} ({Variant})", release.ReleaseId, variant);
    }

    private async Task RefreshAsync(CancellationToken token)
    {
        if (variant is not ("int8" or "fp16" or "fp32")) throw new InvalidDataException("Unsupported MODEL_VARIANT.");
        var connection = config["AZURITE_CONNECTION_STRING"];
        if (string.IsNullOrWhiteSpace(connection)) throw new InvalidOperationException("AZURITE_CONNECTION_STRING is required for model refresh.");
        // Pin the wire version to one supported by Azurite without skipApiVersionCheck.
        var options = new BlobClientOptions(BlobClientOptions.ServiceVersion.V2023_11_03);
        options.Retry.MaxRetries = 2;
        options.Retry.NetworkTimeout = TimeSpan.FromSeconds(30);
        var containerName = config["AZURITE_MODEL_CONTAINER"] ?? "computer-vision-models";
        var container = new BlobContainerClient(connection, containerName, options);
        var pointerName = config["AZURITE_MODEL_CURRENT_POINTER"] ?? "person_detector_ssd/current.json";
        using var pointer = JsonDocument.Parse((await container.GetBlobClient(pointerName).DownloadContentAsync(token)).Value.Content);
        var manifestName = pointer.RootElement.GetProperty("storage").GetProperty("releaseManifest").GetString()!;
        using var manifest = JsonDocument.Parse((await container.GetBlobClient(manifestName).DownloadContentAsync(token)).Value.Content);
        var release = ModelRelease.Parse(pointer.RootElement, manifest.RootElement, variant, containerName);
        if (release.Identity == activeIdentity) return;
        Directory.CreateDirectory(cache);
        var staging = Path.Combine(cache, ".download-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(staging);
        try
        {
            foreach (var artifact in release.Artifacts)
                await container.GetBlobClient(artifact.Blob).DownloadToAsync(Path.Combine(staging, artifact.FileName), token);
            await release.VerifyAsync(staging, token);
            var directory = Path.Combine(cache, release.Identity);
            if (Directory.Exists(directory) &&
                Path.GetDirectoryName(Path.GetFullPath(inference.ModelPath)) == Path.GetFullPath(directory))
            {
                // Never overwrite files backing a loaded model (including a previous
                // activation whose cache-marker write failed).
                await release.VerifyAsync(directory, token);
            }
            else
            {
                // Replace an inactive cache entry, allowing recovery after disk corruption.
                if (Directory.Exists(directory)) Directory.Delete(directory, true);
                Directory.Move(staging, directory);
            }
            await ActivateAsync(release, directory, token);
            var marker = Path.Combine(cache, "active.json");
            await File.WriteAllTextAsync(marker + ".tmp", JsonSerializer.Serialize(release), token);
            File.Move(marker + ".tmp", marker, true);
            activeIdentity = release.Identity;
            // Retain the active release only; old native workers have finished by now.
            foreach (var old in Directory.EnumerateDirectories(cache))
            {
                var name = Path.GetFileName(old);
                if (old != directory && name.Length == 64 && name.All(Uri.IsHexDigit)) Directory.Delete(old, true);
            }
        }
        finally { if (Directory.Exists(staging)) Directory.Delete(staging, true); }
    }
}
