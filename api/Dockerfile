FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY cameraApi.csproj .
RUN dotnet restore
COPY . .
RUN dotnet publish -c Release -o /out --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV ASPNETCORE_URLS=http://+:8080
COPY --from=build /out .
RUN mkdir -p /app/streams && chown -R $APP_UID:$APP_UID /app
USER $APP_UID
EXPOSE 8080
ENTRYPOINT ["dotnet", "cameraApi.dll"]

