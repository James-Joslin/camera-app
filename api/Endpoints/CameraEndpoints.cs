using CameraSoftware.Api.Data;
using CameraSoftware.Api.Domain;
using CameraSoftware.Api.Security;

namespace CameraSoftware.Api.Endpoints;

public static class CameraEndpoints
{
    public static void MapCameraEndpoints(this WebApplication app)
    {
        var cameras = app.MapGroup("/api/cameras").WithTags("Cameras");
        cameras.MapGet("/", (CameraRepository repository, CancellationToken token) => repository.ListAsync(token));
        cameras.MapPost("/", async (CameraCreateRequest body, HttpRequest request, CameraRepository repository, AuthService auth, SecretCipher cipher, CancellationToken token) =>
        {
            if (await auth.AuthenticateAsync(request, token) is null) return Results.Unauthorized();
            if (string.IsNullOrWhiteSpace(body.Name) || string.IsNullOrWhiteSpace(body.Host) || body.Port is < 1 or > 65535)
                return Results.BadRequest(new { error = "Name, host, and a valid port are required." });
            var camera = await repository.CreateAsync(body, cipher.Encrypt(body.Username), cipher.Encrypt(body.Password), token);
            return Results.Created($"/api/cameras/{camera.Id}", camera);
        });
        cameras.MapDelete("/{id:guid}", async (Guid id, HttpRequest request, CameraRepository repository, AuthService auth, CancellationToken token) =>
        {
            if (await auth.AuthenticateAsync(request, token) is null) return Results.Unauthorized();
            return await repository.DeleteAsync(id, token) ? Results.NoContent() : Results.NotFound();
        });
    }
}

