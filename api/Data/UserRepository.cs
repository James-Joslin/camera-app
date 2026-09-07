using CameraSoftware.Api.Domain;
using Npgsql;

namespace CameraSoftware.Api.Data;

public sealed record StoredUser(Guid Id, string Email, string DisplayName, byte[] PasswordHash, byte[] PasswordSalt, DateTimeOffset CreatedAt);

public sealed class UserRepository(NpgsqlDataSource dataSource)
{
    public async Task<StoredUser?> FindByEmailAsync(string email, CancellationToken cancellationToken)
    {
        const string sql = "SELECT id, email, display_name, password_hash, password_salt, created_at FROM app_users WHERE email = @email";
        await using var command = dataSource.CreateCommand(sql);
        command.Parameters.AddWithValue("email", email.Trim().ToLowerInvariant());
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken)) return null;
        return new StoredUser(reader.GetGuid(0), reader.GetString(1), reader.GetString(2), (byte[])reader[3], (byte[])reader[4], reader.GetFieldValue<DateTimeOffset>(5));
    }

    public async Task<StoredUser> CreateAsync(RegisterRequest request, byte[] hash, byte[] salt, CancellationToken cancellationToken)
    {
        const string sql = """
            INSERT INTO app_users (email, display_name, password_hash, password_salt)
            VALUES (@email, @name, @hash, @salt)
            RETURNING id, email, display_name, password_hash, password_salt, created_at
            """;
        await using var command = dataSource.CreateCommand(sql);
        command.Parameters.AddWithValue("email", request.Email.Trim().ToLowerInvariant());
        command.Parameters.AddWithValue("name", request.DisplayName.Trim());
        command.Parameters.AddWithValue("hash", hash);
        command.Parameters.AddWithValue("salt", salt);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        await reader.ReadAsync(cancellationToken);
        return new StoredUser(reader.GetGuid(0), reader.GetString(1), reader.GetString(2), (byte[])reader[3], (byte[])reader[4], reader.GetFieldValue<DateTimeOffset>(5));
    }

    public async Task StoreSessionAsync(Guid userId, byte[] tokenHash, DateTimeOffset expiresAt, CancellationToken cancellationToken)
    {
        await using var command = dataSource.CreateCommand("INSERT INTO user_sessions (user_id, token_hash, expires_at) VALUES (@userId, @tokenHash, @expiresAt)");
        command.Parameters.AddWithValue("userId", userId);
        command.Parameters.AddWithValue("tokenHash", tokenHash);
        command.Parameters.AddWithValue("expiresAt", expiresAt);
        await command.ExecuteNonQueryAsync(cancellationToken);
    }

    public async Task<UserView?> FindBySessionAsync(byte[] tokenHash, CancellationToken cancellationToken)
    {
        const string sql = """
            SELECT u.id, u.email, u.display_name, u.created_at
            FROM user_sessions s JOIN app_users u ON u.id = s.user_id
            WHERE s.token_hash = @tokenHash AND s.expires_at > now()
            """;
        await using var command = dataSource.CreateCommand(sql);
        command.Parameters.AddWithValue("tokenHash", tokenHash);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken)) return null;
        return new UserView(reader.GetGuid(0), reader.GetString(1), reader.GetString(2), reader.GetFieldValue<DateTimeOffset>(3));
    }
}

