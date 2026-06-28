import unittest

from doit_agent.shell import run_shell


class ShellTests(unittest.TestCase):
    def test_run_shell_captures_stdout(self):
        result = run_shell("printf hello")
        self.assertEqual(result.stdout, "hello")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.returncode, 0)

    def test_run_shell_captures_stderr_and_returncode(self):
        result = run_shell("printf err >&2; exit 7")
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "err")
        self.assertEqual(result.returncode, 7)

    def test_run_shell_timeout(self):
        result = run_shell("sleep 2", timeout=1)
        self.assertEqual(result.returncode, 124)
        self.assertIn("timed out", result.stderr)


if __name__ == "__main__":
    unittest.main()

