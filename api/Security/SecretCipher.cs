using System.Security.Cryptography;

namespace CameraSoftware.Api.Security;

public sealed class SecretCipher
{
    private readonly byte[] key;

    public SecretCipher()
    {
        var encoded = Environment.GetEnvironmentVariable("CAMERA_CREDENTIAL_KEY")
            ?? throw new InvalidOperationException("CAMERA_CREDENTIAL_KEY must be a base64-encoded 32-byte key.");
        key = Convert.FromBase64String(encoded);
        if (key.Length != 32) throw new InvalidOperationException("CAMERA_CREDENTIAL_KEY must decode to exactly 32 bytes.");
    }

    public string Encrypt(string plaintext)
    {
        var nonce = RandomNumberGenerator.GetBytes(12);
        var source = System.Text.Encoding.UTF8.GetBytes(plaintext);
        var ciphertext = new byte[source.Length];
        var tag = new byte[16];
        using var aes = new AesGcm(key, 16);
        aes.Encrypt(nonce, source, ciphertext, tag);
        return Convert.ToBase64String([.. nonce, .. tag, .. ciphertext]);
    }

    public string Decrypt(string encoded)
    {
        var payload = Convert.FromBase64String(encoded);
        var nonce = payload[..12];
        var tag = payload[12..28];
        var ciphertext = payload[28..];
        var plaintext = new byte[ciphertext.Length];
        using var aes = new AesGcm(key, 16);
        aes.Decrypt(nonce, ciphertext, tag, plaintext);
        return System.Text.Encoding.UTF8.GetString(plaintext);
    }
}

