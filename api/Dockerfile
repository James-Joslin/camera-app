FROM python:3.12-slim-bookworm AS native
RUN apt-get update && apt-get install -y --no-install-recommends cmake g++ libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends make && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --no-deps openvino==2025.3.0
COPY Inference/native /native
RUN cmake -S /native -B /native/build -DCMAKE_BUILD_TYPE=Release \
      -DOpenVINO_DIR=/usr/local/lib/python3.12/site-packages/openvino/cmake \
      -DCMAKE_INSTALL_PREFIX=/native/out \
    && cmake --build /native/build -j2 && cmake --install /native/build \
    && cp /usr/local/lib/python3.12/site-packages/openvino/libs/*.so* /native/out/lib/

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY cameraApi.csproj .
RUN dotnet restore
COPY . .
RUN dotnet publish -c Release -o /out --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl libopencv-core406 libopencv-imgproc406 libopencv-imgcodecs406 libtbb12 && rm -rf /var/lib/apt/lists/*
COPY --from=native /native/out/lib/ /opt/camera-inference/
ENV LD_LIBRARY_PATH=/opt/camera-inference
WORKDIR /app
ENV ASPNETCORE_URLS=http://+:8080
COPY --from=build /out .
RUN mkdir -p /app/streams && chown -R $APP_UID:$APP_UID /app
USER $APP_UID
EXPOSE 8080
ENTRYPOINT ["dotnet", "cameraApi.dll"]

