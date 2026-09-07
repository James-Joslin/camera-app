using Npgsql;

namespace CameraSoftware.Api.Data;

public sealed class StreamSessionRepository(NpgsqlDataSource dataSource)
{
    public async Task<Guid> StartedAsync(Guid cameraId, Guid? userId, CancellationToken cancellationToken)
    {
        await using var command = dataSource.CreateCommand("INSERT INTO stream_sessions (camera_id, started_by, status) VALUES (@cameraId, @userId, 'starting') RETURNING id");
        command.Parameters.AddWithValue("cameraId", cameraId);
        command.Parameters.AddWithValue("userId", userId.HasValue ? userId.Value : DBNull.Value);
        return (Guid)(await command.ExecuteScalarAsync(cancellationToken))!;
    }

    public async Task SetStatusAsync(Guid sessionId, string status, string? error = null)
    {
        const string sql = """
            UPDATE stream_sessions SET status = @status, error = @error,
                stopped_at = CASE WHEN @status IN ('stopped', 'failed') THEN now() ELSE stopped_at END
            WHERE id = @id
            """;
        await using var command = dataSource.CreateCommand(sql);
        command.Parameters.AddWithValue("id", sessionId);
        command.Parameters.AddWithValue("status", status);
        command.Parameters.AddWithValue("error", error is null ? DBNull.Value : error);
        await command.ExecuteNonQueryAsync();
    }
}

