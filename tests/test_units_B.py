import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import unittest

from ths_zero.units_B import rpm_to_radps, radps_to_rpm, sign, clamp


class TestUnitsB(unittest.TestCase):
    def test_rpm_radps_roundtrip(self):
        for rpm in [0.0, 60.0, 1234.5, -500.0]:
            radps = rpm_to_radps(rpm)
            rpm2 = radps_to_rpm(radps)
            self.assertAlmostEqual(rpm, rpm2, places=9)

    def test_sign(self):
        self.assertEqual(sign(1.0), 1)
        self.assertEqual(sign(-1.0), -1)
        self.assertEqual(sign(0.0), 0)

    def test_clamp(self):
        self.assertEqual(clamp(5.0, 0.0, 1.0), 1.0)
        self.assertEqual(clamp(-1.0, 0.0, 1.0), 0.0)
        self.assertEqual(clamp(0.5, 0.0, 1.0), 0.5)


if __name__ == "__main__":
    unittest.main()
