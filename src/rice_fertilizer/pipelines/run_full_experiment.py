from .common import initialize_run
from ..data.preprocessing_pipeline import run_preprocessing
from ..modeling.trainer import run_training
from ..utils import configure_logging

def main(run_id: str | None = None):
    configure_logging()
    configs, run_dir = initialize_run(run_id=run_id)
    preprocessing = run_preprocessing(configs, run_dir)
    training = run_training(configs, run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Processed dataset: {preprocessing['processed_dataset']}")
    print(f"Selected model: {training['selected_model']}")
    print(
        "Optimization is intentionally separate. Fill optimization decisions, then run "
        "scripts/optimize_scenarios.py with the selected bundle."
    )
    return run_dir

if __name__ == "__main__":
    main()
