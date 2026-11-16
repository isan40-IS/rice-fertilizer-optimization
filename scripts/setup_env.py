import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print(f"\n[EXECUTE] {' '.join(cmd)}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Perintah gagal: {' '.join(cmd)}")
        sys.exit(result.returncode)


def check_venv_exists(env_name: str) -> bool:
    venv_dir = Path(env_name)
    if not venv_dir.exists():
        return False

    python_bin = (
        venv_dir / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else venv_dir / "bin" / "python"
    )
    return python_bin.exists()


def check_conda_exists(env_name: str) -> bool:
    try:
        output = subprocess.check_output(
            ["conda", "env", "list"], text=True, stderr=subprocess.DEVNULL
        )
        for line in output.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if parts[0] == env_name:
                return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return False


def setup_venv(
    env_name: str, req_file: str, mode: str, env_type: str, python_ver: str
):
    if env_type == "venv":
        venv_dir = Path(env_name)
        
        if check_venv_exists(env_name):
            print(f" Environment 'venv' di '{venv_dir.resolve()}' sudah ada.")
        else:
            print(f"1. Membuat Python venv di '{venv_dir.resolve()}'")
            run_cmd([sys.executable, "-m", "venv", env_name])

        pip_bin = (
            str(venv_dir / "Scripts" / "pip.exe")
            if sys.platform == "win32"
            else str(venv_dir / "bin" / "pip")
        )
        python_bin = (
            str(venv_dir / "Scripts" / "python.exe")
            if sys.platform == "win32"
            else str(venv_dir / "bin" / "python")
        )

        print("\n2. Mengupgrade Pip")
        run_cmd([python_bin, "-m", "pip", "install", "--upgrade", "pip"])

        if Path(req_file).exists():
            print(f"\n3. Menginstall/Sync dependencies dari '{req_file}'")
            run_cmd([pip_bin, "install", "-r", req_file])

        if mode in ["dev", "test"] and Path("requirements-dev.txt").exists():
            print("\n4. Menginstall/Sync dependencies Dev/Test")
            run_cmd([pip_bin, "install", "-r", "requirements-dev.txt"])

        print("\n5. Menginstall Package Lokal (-e .)")
        run_cmd([pip_bin, "install", "-e", "."])

        print("\n" + " Setup Selesai!".center(50))
        if sys.platform == "win32":
            print(f" Aktifkan dengan: .\\{env_name}\\Scripts\\activate")
        else:
            print(f" Aktifkan dengan: source {env_name}/bin/activate")

    elif env_type == "conda":
        if check_conda_exists(env_name):
            print(f" Environment Conda '{env_name}' sudah ada.")
        else:
            print(f"1. Membuat Conda Env '{env_name}' (Python {python_ver})")
            run_cmd(["conda", "create", "-n", env_name, f"python={python_ver}", "-y"])

        run_cmd(
            [
                "conda",
                "run",
                "-n",
                env_name,
                "python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ]
        )

        if Path(req_file).exists():
            run_cmd(
                ["conda", "run", "-n", env_name, "pip", "install", "-r", req_file]
            )

        if mode in ["dev", "test"] and Path("requirements-dev.txt").exists():
            run_cmd(
                [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "pip",
                    "install",
                    "-r",
                    "requirements-dev.txt",
                ]
            )

        run_cmd(["conda", "run", "-n", env_name, "pip", "install", "-e", "."])
        print(f"\nAktifkan dengan: conda activate {env_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script Otomatisasi Environment Setup"
    )
    parser.add_argument(
        "--type",
        "-t",
        choices=["venv", "conda"],
        default="venv",
        help="Tipe env: venv atau conda (default: venv)",
    )
    parser.add_argument(
        "--name",
        "-n",
        default=".rfo",
        help="Nama folder venv atau env conda (default: .rfo)",
    )
    parser.add_argument(
        "--python",
        "-p",
        default="3.14",
        help="Versi Python (khusus conda, default: 3.14)",
    )
    parser.add_argument(
        "--req",
        "-r",
        default="requirements.txt",
        help="File requirements (default: requirements.txt)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["biasa", "dev", "test"],
        default="biasa",
        help="Mode instalasi (default: biasa)",
    )

    args = parser.parse_args()
    setup_venv(args.name, args.req, args.mode, args.type, args.python)