using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using CameraSoftware.Api.Inference;
using Xunit;
using Microsoft.Extensions.Configuration;

namespace CameraApi.Tests;

public class ModelReleaseTests
{
    private static ModelRelease Parse(string id = "release-1", string manifestId = "release-1",
        bool accepted = true, string xmlPath = "person_detector_ssd/releases/release-1/models/person_detector_int8.xml")
    {
        var hash = new string('a', 64);
        using var pointer = JsonDocument.Parse(JsonSerializer.Serialize(new {
            releaseId = id, releaseStatus = "experimental",
            storage = new { container = "models", releasePrefix = $"person_detector_ssd/releases/{id}",
                releaseManifest = $"person_detector_ssd/releases/{id}/release_manifest.json" },
            models = new { int8 = new { xml = xmlPath, bin = $"person_detector_ssd/releases/{id}/models/person_detector_int8.bin" } }
        }));
        using var manifest = JsonDocument.Parse(JsonSerializer.Serialize(new {
            releaseId = manifestId, release = new { accepted, status = "experimental" },
            storage = new { container = "models", prefix = $"person_detector_ssd/releases/{manifestId}" },
            artifacts = new[] {
                new { path = "models/person_detector_int8.xml", size = 12, sha256 = hash },
                new { path = "models/person_detector_int8.bin", size = 12, sha256 = hash }
            }
        }));
        return ModelRelease.Parse(pointer.RootElement, manifest.RootElement, "int8", "models");
    }

    [Fact]
    public void Published_experimental_release_remains_usable() => Assert.Equal("experimental", Parse().Status);

    [Fact]
    public void Rejects_unaccepted_release() => Assert.Throws<InvalidDataException>(() => Parse(accepted: false));

    [Fact]
    public void Rejects_pointer_manifest_mismatch() => Assert.Throws<InvalidDataException>(() => Parse(manifestId: "other"));

    [Fact]
    public void Rejects_artifacts_outside_release() => Assert.Throws<InvalidDataException>(() => Parse(xmlPath: "other/model.xml"));

    [Fact]
    public void Rejects_release_path_traversal() => Assert.Throws<InvalidDataException>(() => Parse(id: "../other", manifestId: "../other"));

    [Fact]
    public async Task Checks_content_even_when_file_size_matches()
    {
        var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var bytes = Encoding.UTF8.GetBytes("good");
            var release = new ModelRelease("id", "experimental", "identity", new[] {
                new ReleaseArtifact("blob", "model.bin", bytes.Length, Convert.ToHexString(SHA256.HashData(bytes)))
            });
            await File.WriteAllBytesAsync(Path.Combine(directory, "model.bin"), bytes);
            await release.VerifyAsync(directory, CancellationToken.None);
            await File.WriteAllTextAsync(Path.Combine(directory, "model.bin"), "evil");
            await Assert.ThrowsAsync<InvalidDataException>(() => release.VerifyAsync(directory, CancellationToken.None));
        }
        finally { Directory.Delete(directory, true); }
    }
}

public class ManualModelRefreshTests
{
    [Fact]
    public async Task Startup_does_not_check_storage_even_with_legacy_polling_settings()
    {
        var directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        var config = new Microsoft.Extensions.Configuration.ConfigurationBuilder()
            .AddInMemoryCollection(new System.Collections.Generic.Dictionary<string, string?> {
                ["MODEL_CACHE_DIR"] = directory,
                ["AZURITE_CONNECTION_STRING"] = "intentionally-invalid",
                ["MODEL_AUTO_REFRESH"] = "true",
                ["MODEL_REFRESH_SECONDS"] = "5"
            }).Build();
        using var inference = new InferenceService(config,
            Microsoft.Extensions.Logging.Abstractions.NullLogger<InferenceService>.Instance);
        using var refresh = new ModelReleaseRefreshService(config, inference,
            Microsoft.Extensions.Logging.Abstractions.NullLogger<ModelReleaseRefreshService>.Instance);
        try
        {
            await refresh.StartAsync(CancellationToken.None);
            // The startup task completes after local restore; no polling loop remains.
            await refresh.ExecuteTask!.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Null(inference.Refresh.LastCheckedAt);
            Assert.Null(inference.Refresh.Error);
            Assert.False(inference.Refresh.InProgress);
            Assert.Equal("manual", inference.Refresh.Mode);
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
    }

    [Fact]
    public async Task Manual_failure_reports_error_and_allows_another_attempt()
    {
        var config = new Microsoft.Extensions.Configuration.ConfigurationBuilder().Build();
        using var inference = new InferenceService(config,
            Microsoft.Extensions.Logging.Abstractions.NullLogger<InferenceService>.Instance);
        using var refresh = new ModelReleaseRefreshService(config, inference,
            Microsoft.Extensions.Logging.Abstractions.NullLogger<ModelReleaseRefreshService>.Instance);
        for (var i = 0; i < 2; i++)
        {
            var result = await refresh.RefreshManuallyAsync(CancellationToken.None);
            Assert.Equal(503, ((Microsoft.AspNetCore.Http.IStatusCodeHttpResult)result).StatusCode);
            Assert.NotNull(inference.Refresh.LastCheckedAt);
            Assert.NotNull(inference.Refresh.Error);
            Assert.False(inference.Refresh.InProgress);
        }
    }
}
