
from xrtm.eval.kit.eval.metrics import ExpectedCalibrationErrorEvaluator
from xrtm.eval.core.eval.definitions import EvaluationResult

def test_ece_basic():
    evaluator = ExpectedCalibrationErrorEvaluator(num_bins=10)
    results = [
        EvaluationResult(subject_id="1", score=0, ground_truth=1, prediction=0.9, metadata={}), # Bin 9
        EvaluationResult(subject_id="2", score=0, ground_truth=0, prediction=0.1, metadata={}), # Bin 1
    ]
    ece, bins = evaluator.compute_calibration_data(results)
    # Bin 9: 1 item, pred 0.9, gt 1. acc 1. mean_conf 0.9. abs(1 - 0.9) = 0.1
    # Bin 1: 1 item, pred 0.1, gt 0. acc 0. mean_conf 0.1. abs(0 - 0.1) = 0.1
    # ECE = (1/2)*0.1 + (1/2)*0.1 = 0.1
    assert abs(ece - 0.1) < 1e-6

def test_ece_mixed_types():
    evaluator = ExpectedCalibrationErrorEvaluator(num_bins=2)
    results = [
        EvaluationResult(subject_id="1", score=0, ground_truth="yes", prediction=0.8, metadata={}),
        EvaluationResult(subject_id="2", score=0, ground_truth="no", prediction="0.2", metadata={}),
        EvaluationResult(subject_id="3", score=0, ground_truth=True, prediction=0.9, metadata={}),
        EvaluationResult(subject_id="4", score=0, ground_truth=False, prediction=0.1, metadata={}),
    ]
    # Bin 0 (0-0.5): Items 2 (0.2), 4 (0.1).
    # Item 2: gt "no" -> 0.0. pred 0.2.
    # Item 4: gt False -> 0.0. pred 0.1.
    # Bin 0 mean_conf = (0.2 + 0.1)/2 = 0.15. mean_acc = 0.
    # Bin 1 (0.5-1.0): Items 1 (0.8), 3 (0.9).
    # Item 1: gt "yes" -> 1.0. pred 0.8.
    # Item 3: gt True -> 1.0. pred 0.9.
    # Bin 1 mean_conf = (0.8 + 0.9)/2 = 0.85. mean_acc = 1.0.

    # ECE = (2/4)*abs(0 - 0.15) + (2/4)*abs(1 - 0.85) = 0.5 * 0.15 + 0.5 * 0.15 = 0.075 + 0.075 = 0.15
    ece, bins = evaluator.compute_calibration_data(results)
    assert abs(ece - 0.15) < 1e-6

def test_ece_out_of_bounds():
    evaluator = ExpectedCalibrationErrorEvaluator(num_bins=10)
    results = [
        EvaluationResult(subject_id="1", score=0, ground_truth=1, prediction=1.5, metadata={}),
        EvaluationResult(subject_id="2", score=0, ground_truth=0, prediction=-0.5, metadata={}),
    ]
    # Prediction 1.5 -> Clamped to 1.0 -> Bin 9 (last bin)
    # Prediction -0.5 -> Clamped to 0.0 -> Bin 0

    # Bin 9: 1 item. pred 1.5. gt 1. mean_conf 1.5. mean_acc 1. abs(1 - 1.5) = 0.5
    # Bin 0: 1 item. pred -0.5. gt 0. mean_conf -0.5. mean_acc 0. abs(0 - -0.5) = 0.5

    # ECE = 0.5 * 0.5 + 0.5 * 0.5 = 0.5

    ece, bins = evaluator.compute_calibration_data(results)
    assert abs(ece - 0.5) < 1e-6

    # Check stored bins for correct values
    # The last bin should have mean_prediction 1.5
    assert abs(bins[9].mean_prediction - 1.5) < 1e-6
    # The first bin should have mean_prediction -0.5
    assert abs(bins[0].mean_prediction + 0.5) < 1e-6

if __name__ == "__main__":
    test_ece_basic()
    test_ece_mixed_types()
    test_ece_out_of_bounds()
    print("All tests passed!")
