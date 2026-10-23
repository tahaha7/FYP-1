import json
from pathlib import Path

from cryptography.fernet import Fernet


def get_key(key_path):
    key_path = Path(key_path)
    if key_path.exists():
        return key_path.read_bytes()
    # first run, make a new key
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    key_path.write_bytes(key)
    return key


def _fernet(key_path):
    return Fernet(get_key(key_path))


def encrypt_file(in_path, out_path, key_path):
    f = _fernet(key_path)
    data = Path(in_path).read_bytes()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(f.encrypt(data))
    return str(out_path)


def decrypt_file(in_path, out_path, key_path):
    f = _fernet(key_path)
    data = Path(in_path).read_bytes()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(f.decrypt(data))
    return str(out_path)


def encrypt_bytes(data, key_path):
    return _fernet(key_path).encrypt(data)


def decrypt_bytes(data, key_path):
    return _fernet(key_path).decrypt(data)


def save_json_encrypted(obj, out_path, key_path):
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(_fernet(key_path).encrypt(raw))
    return str(out_path)


def load_json_encrypted(in_path, key_path):
    raw = _fernet(key_path).decrypt(Path(in_path).read_bytes())
    return json.loads(raw.decode("utf-8"))
