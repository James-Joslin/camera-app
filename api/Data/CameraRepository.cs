using CameraSoftware.Api.Domain;
using Npgsql;

namespace CameraSoftware.Api.Data;

public sealed class CameraRepository(NpgsqlDataSource dataSource)
{
    public async Task<IReadOnlyList<CameraView>> ListAsync(CancellationToken cancellationToken)
    {
        const string sql = """
            SELECT id, name, location, host, port, rtsp_path, enabled,
                   encrypted_username IS NOT NULL AND encrypted_password IS NOT NULL, created_at
            FROM cameras ORDER BY name
            """;
        await using var command = dataSource.CreateCommand(sql);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        var cameras = new List<CameraView>();
        while (await reader.ReadAsync(cancellationToken))
            cameras.Add(new CameraView(
                reader.GetGuid(0), reader.GetString(1), reader.GetString(2), reader.GetString(3),
                reader.GetInt32(4), reader.GetString(5), reader.GetBoolean(6), reader.GetBoolean(7),
                reader.GetFieldValue<DateTimeOffset>(8)));
        return cameras;
    }

    public async Task<CameraSecret?> GetSecretAsync(Guid id, CancellationToken cancellationToken)
    {
        const string sql = """
            SELECT id, name, host, port, rtsp_path, encrypted_username, encrypted_password, enabled
            FROM cameras WHERE id = @id
            """;
        await using var command = dataSource.CreateCommand(sql);
        command.Parameters.AddWithValue("id", id);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        if (!await reader.ReadAsync(cancellationToken)) return null;
        return new CameraSecret(
            reader.GetGuid(0), reader.GetString(1), reader.GetString(2), reader.GetInt32(3),
            reader.GetString(4), reader.GetString(5), reader.GetString(6), reader.GetBoolean(7));
    }

    public async Task<CameraView> CreateAsync(
        CameraCreateRequest request, string encryptedUsername, string encryptedPassword,
        CancellationToken cancellationToken)
    {
        const string sql = """
            INSERT INTO cameras (name, location, host, port, rtsp_path, encrypted_username, encrypted_password, enabled)
            VALUES (@name, @location, @host, @port, @path, @username, @password, @enabled)
            RETURNING id, created_at
            """;
        await using var command = dataSource.CreateCommand(sql);
        command.Parameters.AddWithValue("name", request.Name.Trim());
        command.Parameters.AddWithValue("location", request.Location.Trim());
        command.Parameters.AddWithValue("host", request.Host.Trim());
        command.Parameters.AddWithValue("port", request.Port);
        command.Parameters.AddWithValue("path", NormalizePath(request.RtspPath));
        command.Parameters.AddWithValue("username", encryptedUsername);
        command.Parameters.AddWithValue("password", encryptedPassword);
        command.Parameters.AddWithValue("enabled", request.Enabled);
        await using var reader = await command.ExecuteReaderAsync(cancellationToken);
        await reader.ReadAsync(cancellationToken);
        return new CameraView(
            reader.GetGuid(0), request.Name.Trim(), request.Location.Trim(), request.Host.Trim(), request.Port,
            NormalizePath(request.RtspPath), request.Enabled, true, reader.GetFieldValue<DateTimeOffset>(1));
    }

    public async Task<bool> DeleteAsync(Guid id, CancellationToken cancellationToken)
    {
        await using var command = dataSource.CreateCommand("DELETE FROM cameras WHERE id = @id");
        command.Parameters.AddWithValue("id", id);
        return await command.ExecuteNonQueryAsync(cancellationToken) > 0;
    }

    private static string NormalizePath(string path) => path.StartsWith('/') ? path : $"/{path}";
}

