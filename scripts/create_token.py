
import argparse
import getpass
import sys

from plagiatguard.security import authenticate


def main():
    parser = argparse.ArgumentParser(
        description="Générer un JWT pour un utilisateur existant."
    )

    parser.add_argument(
        "--username",
        required=True,
        help="Nom de l'utilisateur"
    )

    args = parser.parse_args()

    password = getpass.getpass(
        "Mot de passe (12 caractères minimum) : "
    )

    token = authenticate(args.username, password)

    if token is None:
        print("Erreur : utilisateur inexistant, inactif ou mot de passe incorrect.")
        sys.exit(1)

    print("\nAuthentification réussie.")
    print("Jeton JWT :")
    print(token)


if __name__ == "__main__":
    main()
