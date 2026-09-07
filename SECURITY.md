# Security policy

Do not report secrets, camera footage, credentials, or private model artifacts in public issues. Send a private report to the repository maintainers with a reproducible description and impact.

The sample environment files contain development placeholders only. Replace all secrets before production use, keep Azurite on private networks, and mount Kaggle credentials at runtime rather than baking them into images.


Camera credentials are encrypted with the `CAMERA_CREDENTIAL_KEY` AES key, passwords use salted PBKDF2, and bearer tokens are stored only as hashes. Treat the camera key as durable production key material: rotating it invalidates existing encrypted camera credentials. RTSP credentials must never be placed in logs, command strings, issues, or screenshots.
