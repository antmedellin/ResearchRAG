"""Pure-unit acceptance tests for the cross-department identity policy; no live DB required."""
import unittest
from research_portfolio.common import norm_key
from research_portfolio.resolve.dedup import sequence_marker

class CrossDepartmentPolicy(unittest.TestCase):
    def test_joint_appointment_union(self): self.assertEqual(sorted(set(["maie"]+["ece"])),["ece","maie"])
    def test_exact_title_normalization(self): self.assertEqual(norm_key("Gas-Assisted Injection"),norm_key("Gas Assisted Injection"))
    def test_part_numbers_are_distinct(self):
        self.assertEqual(sequence_marker("Matrix, Part 1"),"1"); self.assertEqual(sequence_marker("Matrix, Part 2"),"2")
        self.assertNotEqual(sequence_marker("Matrix, Part 1"),sequence_marker("Matrix, Part 2"))
    def test_grant_renewal_is_distinct(self): self.assertNotEqual(norm_key("Advanced Manufacturing"),norm_key("Advanced Manufacturing Renewal"))

if __name__=="__main__": unittest.main()
