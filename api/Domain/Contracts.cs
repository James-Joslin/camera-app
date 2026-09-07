namespace CameraSoftware.Api.Domain;

public sealed record UserView(Guid Id, string Email, string DisplayName, DateTimeOffset CreatedAt);
public sealed record RegisterRequest(string Email, string DisplayName, string Password);
public sealed record LoginRequest(string Email, string Password);
public sealed record AuthResponse(string Token, DateTimeOffset ExpiresAt, UserView User);

public sealed record CameraView(
    Guid Id, string Name, string Location, string Host, int Port, string RtspPath,
    bool Enabled, bool CredentialsConfigured, DateTimeOffset CreatedAt);

public sealed record CameraCreateRequest(
    string Name, string Location, string Host, int Port, string RtspPath,
    string Username, string Password, bool Enabled = true);

public sealed record CameraSecret(
    Guid Id, string Name, string Host, int Port, string RtspPath,
    string EncryptedUsername, string EncryptedPassword, bool Enabled);

public sealed record StreamView(
    Guid CameraId, string CameraName, string Status, string? StreamUrl,
    DateTimeOffset? StartedAt, string? Error);

