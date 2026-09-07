using System;
using Xunit;
using CameraSoftware.Api.Security;

namespace CameraApi.Tests;

public sealed class HealthTests
{
    [Fact]
    public void Camera_credentials_round_trip_through_authenticated_encryption()
    {
        Environment.SetEnvironmentVariable(
            "CAMERA_CREDENTIAL_KEY",
            "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=");
        var cipher = new SecretCipher();

        var encrypted = cipher.Encrypt("camera-password");

        Assert.NotEqual("camera-password", encrypted);
        Assert.Equal("camera-password", cipher.Decrypt(encrypted));
    }
}
