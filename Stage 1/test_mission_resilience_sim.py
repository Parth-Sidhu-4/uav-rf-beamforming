
import importlib.util, sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("mrs", str(Path(__file__).with_name("mission_resilience_sim.py")))
mrs = importlib.util.module_from_spec(SPEC)
sys.modules["mrs"] = mrs
SPEC.loader.exec_module(mrs)

def test_layer1():
    assert mrs.run_layer1_unit_test()["passed"]

def test_layer2():
    assert mrs.run_layer2_unit_test()["passed"]

def test_layer3():
    assert mrs.run_layer3_unit_test()["passed"]

def test_layer4():
    assert mrs.run_layer4_unit_test()["passed"]

def test_layer5():
    assert mrs.run_layer5_unit_test()["passed"]

def test_experiment2():
    assert mrs.run_experiment_2()["validated"]
