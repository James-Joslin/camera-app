using System.Collections.Concurrent;
using System.Diagnostics;
using CameraSoftware.Api.Data;
using CameraSoftware.Api.Domain;
using CameraSoftware.Api.Security;

namespace CameraSoftware.Api.Streaming;

public sealed class StreamManager(
    CameraRepository cameras,
    StreamSessionRepository sessions,
    SecretCipher cipher,
    ILogger<StreamManager> logger) : BackgroundService
{
    private sealed record ActiveStream(Process Process, Guid SessionId, string CameraName, DateTimeOffset StartedAt);
    private readonly ConcurrentDictionary<Guid, ActiveStream> active = new();
    private readonly string streamRoot = Path.GetFullPath(Environment.GetEnvironmentVariable("STREAM_ROOT") ?? "/app/streams");

    public IReadOnlyList<StreamView> List() => active.Select(item => new StreamView(
        item.Key, item.Value.CameraName, item.Value.Process.HasExited ? "stopped" : "live",
        $"/streams/{item.Key}/index.m3u8", item.Value.StartedAt, null)).ToList();

    public async Task<(StreamView? Stream, string? Error)> StartAsync(Guid cameraId, Guid? userId, CancellationToken cancellationToken)
    {
        if (active.TryGetValue(cameraId, out var existing) && !existing.Process.HasExited)
            return (new StreamView(cameraId, existing.CameraName, "live", $"/streams/{cameraId}/index.m3u8", existing.StartedAt, null), null);

        var camera = await cameras.GetSecretAsync(cameraId, cancellationToken);
        if (camera is null) return (null, "Camera not found.");
        if (!camera.Enabled) return (null, "Camera is disabled.");

        var cameraDirectory = Path.Combine(streamRoot, cameraId.ToString());
        Directory.CreateDirectory(cameraDirectory);
        ClearStaleStreamArtifacts(cameraDirectory);
        var sessionId = await sessions.StartedAsync(cameraId, userId, cancellationToken);
        var playlist = Path.Combine(cameraDirectory, "index.m3u8");
        var process = BuildProcess(BuildRtspUrl(camera), playlist);

        try
        {
            if (!process.Start()) throw new InvalidOperationException("FFmpeg did not start.");
            var startedAt = DateTimeOffset.UtcNow;
            var activeStream = new ActiveStream(process, sessionId, camera.Name, startedAt);
            active[cameraId] = activeStream;
            await sessions.SetStatusAsync(sessionId, "live");
            _ = ObserveExitAsync(cameraId, activeStream);
            return (new StreamView(cameraId, camera.Name, "live", $"/streams/{cameraId}/index.m3u8", startedAt, null), null);
        }
        catch (Exception exception)
        {
            process.Dispose();
            await sessions.SetStatusAsync(sessionId, "failed", exception.Message);
            return (null, exception.Message);
        }
    }

    public async Task<bool> StopCameraAsync(Guid cameraId)
    {
        if (!active.TryRemove(cameraId, out var stream)) return false;
        if (!stream.Process.HasExited)
        {
            stream.Process.Kill(entireProcessTree: true);
            await stream.Process.WaitForExitAsync();
        }
        await sessions.SetStatusAsync(stream.SessionId, "stopped");
        return true;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try { await Task.Delay(Timeout.Infinite, stoppingToken); }
        catch (OperationCanceledException) { }
    }

    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        foreach (var cameraId in active.Keys) await StopCameraAsync(cameraId);
        await base.StopAsync(cancellationToken);
    }

    private void ClearStaleStreamArtifacts(string cameraDirectory)
    {
        foreach (var file in Directory.EnumerateFiles(cameraDirectory))
        {
            var extension = Path.GetExtension(file);
            if (extension is not (".m3u8" or ".ts" or ".tmp")) continue;
            try
            {
                File.Delete(file);
            }
            catch (IOException exception)
            {
                logger.LogWarning(exception, "Could not remove stale stream artifact {File}", file);
            }
        }
    }

    private static Process BuildProcess(string rtspUrl, string playlist)
    {
        var start = new ProcessStartInfo("ffmpeg")
        {
            UseShellExecute = false,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            CreateNoWindow = true
        };
        foreach (var argument in new[]
        {
            "-hide_banner", "-loglevel", "warning",
            "-fflags", "nobuffer", "-rtsp_transport", "tcp", "-reorder_queue_size", "0",
            "-i", rtspUrl, "-map", "0:v:0", "-an", "-c:v", "copy",
            "-flush_packets", "1", "-muxdelay", "0", "-muxpreload", "0",
            "-f", "hls", "-hls_time", "1", "-hls_list_size", "4",
            "-hls_delete_threshold", "2", "-hls_start_number_source", "epoch",
            "-hls_flags", "delete_segments+omit_endlist+independent_segments+temp_file",
            "-hls_segment_filename", Path.Combine(Path.GetDirectoryName(playlist)!, "segment-%010d.ts"), playlist
        }) start.ArgumentList.Add(argument);
        return new Process { StartInfo = start, EnableRaisingEvents = true };
    }

    private async Task ObserveExitAsync(Guid cameraId, ActiveStream stream)
    {
        var process = stream.Process;
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();
        var wasActive = ((ICollection<KeyValuePair<Guid, ActiveStream>>)active)
            .Remove(new KeyValuePair<Guid, ActiveStream>(cameraId, stream));
        var message = string.IsNullOrWhiteSpace(error) ? null : error[^Math.Min(error.Length, 1000)..];
        message = Sanitize(message);
        await sessions.SetStatusAsync(stream.SessionId, wasActive && process.ExitCode != 0 ? "failed" : "stopped", wasActive ? message : null);
        logger.LogInformation("Camera {CameraId} FFmpeg process exited with {ExitCode}", cameraId, process.ExitCode);
        process.Dispose();
    }

    private static string? Sanitize(string? value) => value is null
        ? null
        : System.Text.RegularExpressions.Regex.Replace(
            value, @"rtsp://[^@\s]+@", "rtsp://***@",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);

    private string BuildRtspUrl(CameraSecret camera)
    {
        var username = Uri.EscapeDataString(cipher.Decrypt(camera.EncryptedUsername));
        var password = Uri.EscapeDataString(cipher.Decrypt(camera.EncryptedPassword));
        return $"rtsp://{username}:{password}@{camera.Host}:{camera.Port}{camera.RtspPath}";
    }
}

