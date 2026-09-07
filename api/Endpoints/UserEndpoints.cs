using CameraSoftware.Api.Domain;
using CameraSoftware.Api.Security;

namespace CameraSoftware.Api.Endpoints;

public static class UserEndpoints
{
    public static void MapUserEndpoints(this WebApplication app)
    {
        var users = app.MapGroup("/api/users").WithTags("Users");
        users.MapPost("/register", async (RegisterRequest request, AuthService auth, CancellationToken token) =>
        {
            var result = await auth.RegisterAsync(request, token);
            return result.Response is null ? Results.BadRequest(new { error = result.Error }) : Results.Ok(result.Response);
        });
        users.MapPost("/login", async (LoginRequest request, AuthService auth, CancellationToken token) =>
        {
            var result = await auth.LoginAsync(request, token);
            return result.Response is null ? Results.Json(new { error = result.Error }, statusCode: 401) : Results.Ok(result.Response);
        });
        users.MapGet("/me", async (HttpRequest request, AuthService auth, CancellationToken token) =>
        {
            var user = await auth.AuthenticateAsync(request, token);
            return user is null ? Results.Unauthorized() : Results.Ok(user);
        });
    }
}

