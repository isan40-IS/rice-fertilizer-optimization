from .common import initialize_run
from ..modeling.trainer import run_training
from ..utils import configure_logging

def main(run_id: str | None = None):
    configure_logging()
    configs, run_dir = initialize_run(run_id=run_id)
    outputs = run_training(configs, run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Selected model: {outputs['selected_model']}")
    return run_dir, outputs

if __name__ == "__main__":
    main()
