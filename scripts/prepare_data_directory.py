import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir / "src"))

from rice_fertilizer.config import load_all_configs, resolve_path

def main() -> None:
    configs = load_all_configs()
    input_path = resolve_path(configs["paths"]["input_csv"])
    input_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input directory ready: {input_path.parent}")
    print(f"Place the CSV at: {input_path}")


if __name__ == "__main__":
    main()
