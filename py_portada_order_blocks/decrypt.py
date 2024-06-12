import base64
import hashlib
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


def removing_padding_pkcs7(data):
    padding_len = data[-1]
    bret = data[:-padding_len]
    while bret[-1]==10 or bret[-1]==13:
        bret = bret[:-1]
    return bret


def decrypt_file_openssl(file_name_enc, key, iterations=100000):
    # Leer el fichero encriptado
    with open(file_name_enc, "rb") as file_to_decrypt:
        encrypted_content = file_to_decrypt.read()

    # Extract the salt from the beginning of the file (assuming 16 bytes after the "Salted__" header)
    salt = encrypted_content[8:16]
    encrypted_content = encrypted_content[16:]

    # Derive the key and IV using the password and salt with PBKDF2
    dk = hashlib.pbkdf2_hmac('sha256', key.encode('utf-8', errors='surrogateescape'), salt, iterations, dklen=32 + 16)
    key = dk[:32]
    iv = dk[32:]

    # Create the cipher object and decrypt the data
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decrypt = cipher.decryptor()
    decrypted_content = decrypt.update(encrypted_content) + decrypt.finalize()

    return removing_padding_pkcs7(decrypted_content).decode("utf-8")
