using CameraSoftware.Api.Data;
using CameraSoftware.Api.Endpoints;
using CameraSoftware.Api.Security;
using CameraSoftware.Api.Streaming;
using Microsoft.Extensions.FileProviders;
using Npgsql;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var connectionString = $"Host={builder.Configuration["POSTGRES_HOST"] ?? "db"};" +
    $"Port={builder.Configuration["POSTGRES_PORT"] ?? "5432"};" +
    $"Database={builder.Configuration["POSTGRES_DB"] ?? "cameras"};" +
    $"Username={builder.Configuration["POSTGRES_USER"] ?? "camera_app"};" +
    $"Password={builder.Configuration["POSTGRES_PASSWORD"] ?? "change-me"};";

builder.Services.AddSingleton(NpgsqlDataSource.Create(connectionString));
builder.Services.AddSingleton<CameraRepository>();
builder.Services.AddSingleton<UserRepository>();
builder.Services.AddSingleton<StreamSessionRepository>();
builder.Services.AddSingleton<SecretCipher>();
builder.Services.AddSingleton<AuthService>();
builder.Services.AddSingleton<StreamManager>();
builder.Services.AddHostedService<StreamManager>(provider => provider.GetRequiredService<StreamManager>());

var app = builder.Build();
var streamRoot = Path.GetFullPath(Environment.GetEnvironmentVariable("STREAM_ROOT") ?? "/app/streams");
Directory.CreateDirectory(streamRoot);

app.UseSwagger();
app.UseSwaggerUI();
app.UseStaticFiles(new StaticFileOptions
{
    FileProvider = new PhysicalFileProvider(streamRoot),
    RequestPath = "/streams",
    ServeUnknownFileTypes = true,
    DefaultContentType = "application/octet-stream",
    OnPrepareResponse = context =>
    {
        context.Context.Response.Headers.CacheControl = "no-store";
        context.Context.Response.Headers.AccessControlAllowOrigin = "*";
    }
});

app.MapGet("/", () => Results.Ok(new { service = "camera-api", status = "ok" }));
app.MapGet("/status/live", () => Results.Ok(new { status = "ok" }));
app.MapGet("/status/ready", async (NpgsqlDataSource dataSource, CancellationToken cancellationToken) =>
{
    try
    {
        await using var command = dataSource.CreateCommand("SELECT 1");
        await command.ExecuteScalarAsync(cancellationToken);
        return Results.Ok(new { status = "ready", postgres = "ok" });
    }
    catch (Exception exception)
    {
        return Results.Json(new { status = "not_ready", error = exception.Message }, statusCode: 503);
    }
});

app.MapUserEndpoints();
app.MapCameraEndpoints();
app.MapStreamEndpoints();
app.Run();

public partial class Program { }

