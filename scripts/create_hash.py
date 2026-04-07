import getpass
import os

import bcrypt


def main():
    password = os.getenv("RAG_PASSWORD_TO_HASH")

    if not password:
        password = getpass.getpass("Введите пароль для хеширования: ")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    print(hashed.decode())


if __name__ == "__main__":
    main()
