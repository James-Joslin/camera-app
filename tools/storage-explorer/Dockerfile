FROM mcr.microsoft.com/dotnet/runtime:10.0 AS dotnet
FROM lscr.io/linuxserver/webtop:ubuntu-xfce
ARG STORAGE_EXPLORER_VERSION=1.45.0
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnome-keyring dbus-x11 libsecret-1-0 libnss3 libgbm1 libasound2t64 libgtk-3-0 libkrb5-3 xdg-utils \
    && mkdir -p /opt/storage-explorer \
    && curl -fL "https://github.com/microsoft/AzureStorageExplorer/releases/download/v${STORAGE_EXPLORER_VERSION}/StorageExplorer-linux-x64.tar.gz" | tar -xz -C /opt/storage-explorer \
    && rm -rf /var/lib/apt/lists/*
COPY --from=dotnet /usr/share/dotnet /usr/share/dotnet
ENV DOTNET_ROOT=/usr/share/dotnet
ENV PATH="/usr/share/dotnet:${PATH}"
COPY storage-explorer.desktop /usr/share/applications/storage-explorer.desktop
COPY init/10-storage-explorer /custom-cont-init.d/10-storage-explorer
COPY start-storage-explorer /usr/local/bin/start-storage-explorer
