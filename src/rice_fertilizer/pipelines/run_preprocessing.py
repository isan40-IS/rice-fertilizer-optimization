from .common import initialize_run
from ..data.preprocessing_pipeline import run_preprocessing
from ..utils import configure_logging

def main(run_id: str | None = None):
    configure_logging()
    configs, run_dir = initialize_run(run_id=run_id)
    outputs = run_preprocessing(configs, run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Processed dataset: {outputs['processed_dataset']}")
    return run_dir, outputs

if __name__ == "__main__":
    main()
