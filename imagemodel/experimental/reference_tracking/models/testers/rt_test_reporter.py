import platform
from typing import Dict

from imagemodel.common.reporter import TestReporter
from imagemodel.common.setup import TestExperimentSetup
from imagemodel.experimental.reference_tracking.models.testers.tester import Tester


class RTTestReporter(TestReporter):
    def __init__(self, setup: TestExperimentSetup, tester: Tester):
        super().__init__(setup=setup, tester=tester)
    
    def test_report_text(self, setup: TestExperimentSetup, tester: Tester) -> str:
        info: str = """
# Information ---------------------------
Hostname: {}
Test ID: {}
Test Dataset: {}
Test Data Folder: {}/{}
-----------------------------------------
        """.format(
                platform.node(),
                setup.experiment_id,
                tester.test_dataset_description,
                setup.base_data_folder,
                setup.experiment_id)
        return info
    
    def test_result_report_text(self, test_result: Dict[str, float]) -> str:
        result: str = """
# Result ---------------------------
Test Loss: {}
Test Accuracy: {}
-----------------------------------------
        """.format(
                test_result, test_result)
        return result
# test_loss, test_acc = model.evaluate(
#         test_dataset, workers=8, use_multiprocessing=True
# )
# result: str = """
# # Result ---------------------------
# Test Loss: {}
# Test Accuracy: {}
# -----------------------------------------
# """.format(
#         test_loss, test_acc
# )
# print(result)
# tmp_result = "/tmp/result.txt"
# f = open(tmp_result, "w")
# f.write(result)
# f.close()
# upload_blob(
#         bucket_name,
#         tmp_result,
#         os.path.join("data", test_id, os.path.basename(tmp_result)),
# )
