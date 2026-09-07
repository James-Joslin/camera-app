using System.Security.Cryptography;
using CameraSoftware.Api.Data;
using CameraSoftware.Api.Domain;
using Npgsql;

namespace CameraSoftware.Api.Security;

public sealed class AuthService(UserRepository users)
{
    private const int Iterations = 210_000;

    public async Task<(AuthResponse? Response, string? Error)> RegisterAsync(RegisterRequest request, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request.Email) || !request.Email.Contains('@')) return (null, "A valid email is required.");
        if (request.Password.Length < 12) return (null, "Password must be at least 12 characters.");
        if (string.IsNullOrWhiteSpace(request.DisplayName)) return (null, "Display name is required.");
        if (await users.FindByEmailAsync(request.Email, cancellationToken) is not null) return (null, "An account already exists for this email.");
        var salt = RandomNumberGenerator.GetBytes(32);
        try
        {
            var user = await users.CreateAsync(request, HashPassword(request.Password, salt), salt, cancellationToken);
            return (await CreateSessionAsync(user, cancellationToken), null);
        }
        catch (PostgresException exception) when (exception.SqlState == PostgresErrorCodes.UniqueViolation)
        {
            return (null, "An account already exists for this email.");
        }
    }

    public async Task<(AuthResponse? Response, string? Error)> LoginAsync(LoginRequest request, CancellationToken cancellationToken)
    {
        var user = await users.FindByEmailAsync(request.Email, cancellationToken);
        if (user is null || !CryptographicOperations.FixedTimeEquals(user.PasswordHash, HashPassword(request.Password, user.PasswordSalt)))
            return (null, "Invalid email or password.");
        return (await CreateSessionAsync(user, cancellationToken), null);
    }

    public async Task<UserView?> AuthenticateAsync(HttpRequest request, CancellationToken cancellationToken)
    {
        var authorization = request.Headers.Authorization.ToString();
        if (!authorization.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase)) return null;
        try
        {
            var tokenBytes = Convert.FromBase64String(authorization[7..].Trim());
            return await users.FindBySessionAsync(SHA256.HashData(tokenBytes), cancellationToken);
        }
        catch (FormatException) { return null; }
    }

    private async Task<AuthResponse> CreateSessionAsync(StoredUser user, CancellationToken cancellationToken)
    {
        var tokenBytes = RandomNumberGenerator.GetBytes(48);
        var token = Convert.ToBase64String(tokenBytes);
        var expiresAt = DateTimeOffset.UtcNow.AddDays(7);
        await users.StoreSessionAsync(user.Id, SHA256.HashData(tokenBytes), expiresAt, cancellationToken);
        return new AuthResponse(token, expiresAt, new UserView(user.Id, user.Email, user.DisplayName, user.CreatedAt));
    }

    private static byte[] HashPassword(string password, byte[] salt) =>
        Rfc2898DeriveBytes.Pbkdf2(password, salt, Iterations, HashAlgorithmName.SHA512, 64);
}

