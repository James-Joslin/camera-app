using CameraSoftware.Api.Security;
using CameraSoftware.Api.Streaming;

namespace CameraSoftware.Api.Endpoints;

public static class StreamEndpoints
{
    public static void MapStreamEndpoints(this WebApplication app)
    {
        var streams = app.MapGroup("/api/streams").WithTags("Streams");
        streams.MapGet("/", (StreamManager manager) => Results.Ok(manager.List()));
        streams.MapPost("/{cameraId:guid}/start", async (Guid cameraId, HttpRequest request, StreamManager manager, AuthService auth, CancellationToken token) =>
        {
            var user = await auth.AuthenticateAsync(request, token);
            if (user is null) return Results.Unauthorized();
            var result = await manager.StartAsync(cameraId, user.Id, token);
            return result.Stream is null ? Results.BadRequest(new { error = result.Error }) : Results.Ok(result.Stream);
        });
        streams.MapPost("/{cameraId:guid}/stop", async (Guid cameraId, HttpRequest request, StreamManager manager, AuthService auth, CancellationToken token) =>
        {
            if (await auth.AuthenticateAsync(request, token) is null) return Results.Unauthorized();
            return await manager.StopCameraAsync(cameraId) ? Results.NoContent() : Results.NotFound();
        });
    }
}

