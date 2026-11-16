import argparse
from rice_fertilizer.pipelines.run_evaluation import main

parser = argparse.ArgumentParser()
parser.add_argument("--model-bundle", required=True)
parser.add_argument("--optimization-results", required=True)
parser.add_argument("--run-id", default=None)
args = parser.parse_args()
main(
    run_id=args.run_id,
    model_bundle_path=args.model_bundle,
    optimization_results_path=args.optimization_results,
)
